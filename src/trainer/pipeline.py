"""Dispatch orchestration for the v1 trainer.

A ``Pipeline`` owns a device, the compute kernels, and the GPU resources.
For inference, call :meth:`Pipeline.render`. For training, set a target
image with :meth:`Pipeline.set_target_image` and call
:meth:`Pipeline.train_step`.
"""

from __future__ import annotations

import contextlib
import hashlib
from functools import lru_cache

import numpy as np
import slangpy as spy

from trainer import buffers
from trainer import densify as _densify
from trainer.buffers import (
    CLUSTER_SIZE,
    GRAD_SCALAR_COMPONENTS,
    RADIX_GLOBAL_HIST_SIZE,
    RADIX_PART_SIZE,
    SCAN_BLOCK_SIZE,
)
from trainer.config import AdamLRs, RenderConfig
from trainer.dataset import Camera
from trainer.densify import DensifyConfig
from trainer.device import SHADERS_DIR
from trainer.kernels import load_kernels

# DensifyConfig / AdamLRs / RenderConfig are imported (not defined) here and
# re-exported for back-compat: existing callers do
# ``from trainer.pipeline import DensifyConfig, AdamLRs``.
__all__ = ["Pipeline", "DensifyConfig", "AdamLRs", "RenderConfig"]


# DeviceRadixSort dispatch dims — must match shaders/sort/device_radix_sort.slang.
# (The layout-sizing constants RADIX_PART_SIZE / RADIX_GLOBAL_HIST_SIZE /
# SCAN_BLOCK_SIZE are imported from buffers.py; these three are used only in
# thread_count computations here, not in allocation.)
_RADIX_US_DIM = 128
_RADIX_SCAN_DIM = 128
_RADIX_D_DIM = 256


@lru_cache(maxsize=None)
def _marker_color(name: str) -> tuple[float, float, float]:
    """Deterministic pseudo-random RGB for a debug-marker/group name, so the
    same phase gets the same color across runs (profiler captures stay
    diff-able). Uses blake2s, NOT builtin hash() — CPython salts str hashing
    per-process, which would recolor every run."""
    digest = hashlib.blake2s(name.encode("utf-8"), digest_size=3).digest()
    # Bias into [0.35, 1.0] so colors stay legible (never near-black).
    return tuple(0.35 + (b / 255.0) * 0.65 for b in digest)  # type: ignore[return-value]


# TODO: bad swe, we are importing adam, which idiomatically does not explain what we are importing
# ideally we should be importing the shared data module, importing adam because adam imports the shared data module + its own structs is bad.
def _module_for_reflection(device: spy.Device) -> spy.Module:
    """Load a shader file that includes all the shared struct types so
    buffers.py can reflect ``Gaussian3D`` / ``Projected2DGaussian`` /
    ``AdamState``."""
    return spy.Module.load_from_file(
        device, str(SHADERS_DIR / "optimizers" / "adam.slang")
    )


class Pipeline:
    def __init__(
        self,
        device: spy.Device,
        width: int,
        height: int,
        max_splats: int = 500_000,
        max_sort_pairs: int | None = None,
        gpu_lock: "contextlib.AbstractContextManager | None" = None,
        cfg: RenderConfig | None = None,
        viewer_only: bool = False,
    ) -> None:
        self.device = device
        # Serializes GPU queue access (submit/present/wait_for_idle) against
        # any OTHER thread sharing this device — notably the live Viewer, which
        # runs its render loop on a background thread. Vulkan requires queue
        # submission to be externally synchronized; without this, the training
        # thread and viewer thread race on the shared VkQueue, which
        # non-deterministically either faults the device ("Vulkan returned a
        # failure") or thrashes (both threads stall fighting for the queue).
        # Defaults to a no-op so single-threaded use (tests, --no-viewer) is
        # unchanged.
        self.gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()
        self.width = width
        self.height = height
        self.max_splats = max_splats
        if max_sort_pairs is None:
            max_sort_pairs = max(buffers.DEFAULT_MAX_SORT_PAIRS, max_splats * 2)
        self.max_sort_pairs = max_sort_pairs

        self.shared_data_module = _module_for_reflection(device)
        self.viewer_only = viewer_only

        # --- kernels ---
        # Compiled from the declarative KERNEL_SPECS table (kernels.py); each
        # sets self.k_<attr>. Debug-marker names are passed explicitly at each
        # _dispatch call site, so the returned label map is unused here.
        # viewer_only mode compiles just the ~14 kernels the render path needs.
        load_kernels(device, self, viewer_only=viewer_only)

        # --- buffers ---
        # In viewer_only mode, skip the ~3.5 GB (@max_splats=7M) of training-
        # only buffers: grads, adam state, densify scratch, loss textures,
        # SSIM maps, importance state, eta accumulators, split factors.
        # The shared _run_forward touches ONLY the resources allocate_viewer
        # provides — reading anything else would raise AttributeError.
        allocate_fn = buffers.allocate_viewer if viewer_only else buffers.allocate
        self.buffers = allocate_fn(
            device=device, shared_data_module=self.shared_data_module,
            max_splats=max_splats, width=width, height=height,
            max_sort_pairs=max_sort_pairs,
        )

        # Camera buffer: raw 144-byte buffer (136 data + 8 padding to 16-byte align).
        # Using a raw Buffer avoids reflection dependency (survives hot-reload).
        self._camera_buffer = device.create_buffer(
            size=144,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label="camera_data",
        )

        # Training step counter (Python-side, for Adam bias correction).
        self.iteration = 0

        # When set, all dispatches route through this encoder (for debug markers).
        self._encoder: spy.CommandEncoder | None = None

        # ID of the most recent submit_command_buffer call. Callers (the train
        # loop) can wait on this via device.wait_for_submit(id) to bound queue
        # depth without a full wait_for_idle. Kept up to date by every submit
        # site in this file (train_step, densify_simple, _dispatch_indirect
        # transient submits).
        self.last_submit_id: int = 0

        # Render/loss knobs (near-plane cull, regularization weights, alpha
        # loss weight, SSIM mix). Owned by the trainer and passed in; read deep
        # in the dispatch path (bin/loss/regularization). Frozen — see
        # trainer.config.RenderConfig for the fields and their defaults.
        self.cfg = cfg if cfg is not None else RenderConfig()
        # The viewer pipeline deliberately does NOT support cluster-cull: it
        # skips those kernels (see kernels.VIEWER_KERNEL_ATTRS) and never
        # allocates the cluster AABB buffers (buffers.allocate_viewer seeds
        # cluster_visible=all-ones only). If cfg leaves cluster_cull on (the
        # RenderConfig default), _run_forward would dispatch the uncompiled
        # k_build_cluster_aabb and raise AttributeError before Present, which
        # then floods the swapchain with acquireNextImage errors. Force it off
        # here so the render path takes the all-visible no-op branch.
        if viewer_only and self.cfg.cluster_cull:
            import dataclasses
            self.cfg = dataclasses.replace(self.cfg, cluster_cull=False)

        # Backward variant selector. True = transposed per-splat
        # backward_persplat.slang (default; splat-parallel with checkpoint
        # reconstruction, validated gradient-equivalent to the per-pixel
        # kernel in tests/test_backward_persplat.py). False = the original
        # per-pixel backward.slang, kept for side-by-side comparison / fallback.
        self.use_persplat_backward = True

        # Rotating phase for the transposed backward's stochastic pixel
        # subsampling (BACKWARD_PIXEL_STRIDE in backward_persplat.slang).
        # Incremented each backward so the visited pixel subset alternates;
        # a plain int is fine (the shader takes it mod STRIDE). Irrelevant
        # when STRIDE == 1.
        self._backward_pixel_phase = 0

        # Active loss target texture. Defaults to the pipeline's own
        # target_image; set_target_image / set_target_texture swap it. Kept
        # initialized here so _run_loss reads self._active_target directly.
        # viewer_only skips: no loss path exists on this pipeline.
        self._active_target = None if viewer_only else self.buffers.target_image

        # The radix ping-pong slot holding the sorted gaussian-id buffer after
        # the forward pass. _run_forward caches the data-dependent winner here;
        # the backward + scoring reads it. Defaults to slot A (the sort source).
        self._sorted_values = self.buffers.sort_values_a

    def on_hot_reload(self) -> None:
        """Refresh reflection-dependent objects after device.reload_all_programs().
        Camera buffer is raw bytes so it survives reload. Only the module
        reference needs refreshing (for any code that uses reflected types)."""
        self.shared_data_module = _module_for_reflection(self.device)

    def _dispatch(self, kernel, marker: str, **kwargs):
        """Dispatch a kernel, tagging it with a flat debug marker ``marker``
        (only when an encoder is active). ``marker`` is the phase/kernel name
        shown in profiler captures; its color is derived deterministically
        from the name (see _marker_color)."""
        if self._encoder:
            self._encoder.insert_debug_marker(marker, spy.math.float3(*_marker_color(marker)))
        kernel.dispatch(command_encoder=self._encoder, **kwargs)

    #TODO: double check this function, im not sure about the setup
    def _dispatch_indirect(self, kernel, marker: str, args_buf: spy.NDBuffer,
                           offset: int = 0, **kwargs) -> None:
        """Dispatch a compute kernel indirectly (group count read from args_buf
        at `offset` bytes as three uint32s: groups_x, groups_y, groups_z).

        Opens a scoped compute pass on the active encoder, binds the pipeline,
        writes entry-point parameters via a ShaderCursor (matching
        kernel.dispatch(**kwargs) semantics), then dispatch_compute_indirect.
        If no encoder is active (e.g. from render()) a transient encoder is
        created and submitted so this helper works from any caller.
        """
        enc = self._encoder
        transient = enc is None
        if transient:
            enc = self.device.create_command_encoder()
        enc.insert_debug_marker(marker, spy.math.float3(*_marker_color(marker)))
        # HAZARD: the args_buf was written by a preceding kernel dispatch
        # (e.g. prepareDispatchArgs). dispatch_compute_indirect reads it as
        # driver-consumed indirect arguments; without a barrier the read
        # may race the write and pick up garbage → a catastrophic group
        # count. Insert a global barrier to force write-visibility before
        # the indirect fetch. Cheap compared to the cost of the failure
        # mode (a 32-bit garbage groups_x can queue billions of workgroups).
        enc.global_barrier() #TODO: im honestly not sure I need this barrier
        pass_enc = enc.begin_compute_pass()
        try:
            shader_obj = pass_enc.bind_pipeline(kernel.pipeline)
            cursor = spy.ShaderCursor(shader_obj).find_entry_point(0)
            cursor.write(kwargs)
            pass_enc.dispatch_compute_indirect(
                spy.BufferOffsetPair(args_buf.storage, offset)
            )
        finally:
            pass_enc.end()
        if transient:
            cmd = enc.finish()
            with self.gpu_lock:
                self.last_submit_id = self.device.submit_command_buffer(cmd)

    @contextlib.contextmanager
    def _marker_group(self, label: str):
        """Nested debug-group scope for profiling captures (e.g. Nsight
        Graphics). No-op when no encoder is active, so it costs nothing outside
        an encoder-backed train_step. Groups nest, so wrapping a sub-stage of a
        phase (bin / scan / sort / rasterize) gives a drill-down tree under the
        top-level phase group instead of one flat run of per-kernel markers.

        The group color is derived deterministically from ``label`` (see
        _marker_color), so a phase keeps its color across runs and callers no
        longer hand-pick RGB tuples."""
        enc = self._encoder
        if enc is None:
            yield
            return
        enc.push_debug_group(label, spy.math.float3(*_marker_color(label)))
        try:
            yield
        finally:
            enc.pop_debug_group()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def set_gaussians_from_numpy(
        self,
        means: np.ndarray,
        log_scales: np.ndarray,
        rot_quats: np.ndarray,
        opacities: np.ndarray,
        color_logits: np.ndarray,
    ) -> None:
        buffers.write_gaussians(
            self.buffers,
            means=means, log_scales=log_scales, rot_quats=rot_quats,
            opacities=opacities, color_logits=color_logits,
        )
        buffers.zero_adam_state(self.buffers)
        self.iteration = 0

    def set_gaussians_raw(self, floats: np.ndarray, count: int) -> None:
        """Upload a snapshot already in the packed 14-float layout (from the
        splat stream). Render-only path — does not touch Adam state."""
        buffers.write_gaussians_raw(self.buffers, floats, count)

    def set_target_image(self, rgb: np.ndarray) -> None:
        """Upload a target image into the pipeline's owned texture. Call
        this once per iter if you don't want to manage per-camera target
        textures yourself.

        The loss kernel performs ``srgb_to_linear`` on read. We pad the
        image to RGBA because our texture format is rgba32_float.

        For training loops, prefer :meth:`create_target_texture` to
        pre-upload every target once and pass the texture reference
        into :meth:`set_target_texture` each iter — avoids an 8 MB
        upload per step.
        """
        assert rgb.shape == (self.height, self.width, 3), (
            f"target shape {rgb.shape} != ({self.height}, {self.width}, 3)"
        )
        rgba = np.concatenate(
            [rgb.astype(np.float32), np.ones((self.height, self.width, 1), np.float32)],
            axis=-1,
        )
        self.buffers.target_image.copy_from_numpy(np.ascontiguousarray(rgba))
        self._active_target = self.buffers.target_image

    def create_target_texture(self, img: np.ndarray) -> spy.Texture:
        """Create a new rgba32_float target texture and upload into it.

        Accepts (H, W, 3) RGB or (H, W, 4) RGBA. RGB is padded with
        alpha=1.

        For RGBA inputs the RGB target is the STRAIGHT (un-premultiplied)
        sRGB color and the alpha channel is the raw GT coverage. This
        matches gsplat's objective: it compares the premultiplied-over-black
        render against the straight gt_rgb via L1, with a separate alpha
        (coverage) loss. Premultiplying the RGB target instead (target = a*c)
        is satisfiable at LOW coverage, so the RGB loss stops pushing the
        silhouette outward — measured as ~0.15 sRGB darker edges and a
        6-8% coverage shortfall (smaller silhouette) vs gsplat. Keeping the
        target straight lets the RGB residual drive edge coverage/brightness
        up to full, as gsplat does.
        """
        if img.ndim == 3 and img.shape[2] == 4:
            assert img.shape[:2] == (self.height, self.width)
            rgba = img.astype(np.float32).copy()
            # Straight sRGB RGB + raw alpha — no premultiply (see docstring).
        else:
            assert img.shape == (self.height, self.width, 3), (
                f"target shape {img.shape} != ({self.height}, {self.width}, 3)"
            )
            rgba = np.concatenate(
                [img.astype(np.float32), np.ones((self.height, self.width, 1), np.float32)],
                axis=-1,
            )
        tex = self.device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=self.width, height=self.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="target_image_cached",
        )
        tex.copy_from_numpy(np.ascontiguousarray(rgba))
        return tex

    def set_target_texture(self, tex: spy.Texture) -> None:
        """Use a pre-uploaded target texture for subsequent forward+loss
        calls. Cheap: just a reference swap — no copy, no upload."""
        self._active_target = tex

    # ------------------------------------------------------------------
    # Camera + forward pipeline
    # ------------------------------------------------------------------

    def _upload_camera(self, camera: Camera) -> None:
        if camera.image_size != (self.width, self.height):
            raise ValueError(
                f"camera resolution {camera.image_size} does not match pipeline "
                f"resolution {(self.width, self.height)}"
            )
        data = np.empty(36, dtype=np.float32)
        data[0:16] = camera.view_matrix.flatten()
        data[16:32] = camera.proj_matrix.flatten()
        data[32:34] = np.array([self.width, self.height], dtype=np.float32)
        data[34:36] = 0.0  # padding
        self._camera_buffer.copy_from_numpy(np.ascontiguousarray(data.view(np.uint8)))

    def _radix_sort_pairs(self) -> tuple[spy.NDBuffer, spy.NDBuffer, spy.NDBuffer]:
        """DeviceRadixSort (ported from GPUSorting, MIT). Returns the
        (keys_lo, keys_hi, values) triplet holding the sorted result.

        Per-pass structure (see shaders/sort/device_radix_sort.slang):
          upsweep    — per-partition digit histogram + atomic add to
                       b_global_hist for this pass.
          scan       — one workgroup per digit, exclusive scan across
                       active partitions in b_pass_hist.
          downsweep  — wave-multisplit rank + scatter of the 3-word
                       record to the *_out buffers.

        ``n`` (the actual pair count) lives in ``scan_total[0]`` on the
        GPU; Python never reads it back in the hot path.
        """
        b = self.buffers
        src_lo, src_hi, src_v = b.sort_keys_lo_a, b.sort_keys_hi_a, b.sort_values_a
        dst_lo, dst_hi, dst_v = b.sort_keys_lo_b, b.sort_keys_hi_b, b.sort_values_b
        max_blocks = buffers._ceil_div(self.max_sort_pairs, RADIX_PART_SIZE)

        # Prepare indirect-dispatch args from the GPU-resident count
        # scan_total[0]. Upsweep and downsweep are then launched with
        # active_blocks groups instead of max_blocks — dispatch cost scales
        # with the frame's actual pair count, not the max_sort_pairs cap.
        self._dispatch(self.k_radix_prepare_dispatch, "radix_prepare_dispatch",
            thread_count=[1, 1, 1],
            n_buf=b.scan_total.storage,
            dispatch_args=b.radix_dispatch_args.storage,
            max_blocks=max_blocks,
            # Clamp scan_total[0] to buffer capacity in place, so the sort and
            # tile-range stages never read past what bin_emit wrote (which is
            # itself capped at max_sort_pairs). See prepareDispatchArgs.
            max_sort_pairs=self.max_sort_pairs,
        )

        # Clear b_global_hist once at the start of the whole sort;
        # subsequent atomic adds during upsweep accumulate into disjoint
        # per-pass regions (offset = pass_idx * 256).
        self._dispatch(self.k_radix_init_global_hist, "radix_init_global_hist",
            thread_count=[RADIX_GLOBAL_HIST_SIZE, 1, 1],
            b_global_hist=b.radix_global_hist.storage,
        )

        for pass_idx in range(8):
            radix_shift = (pass_idx % 4) * 8
            key_word = 0 if pass_idx < 4 else 1
            with self._marker_group(f"pass_{pass_idx}"):
                self._dispatch_indirect(self.k_radix_upsweep, "radix_upsweep",
                    b.radix_dispatch_args,
                    keys_lo=src_lo.storage, keys_hi=src_hi.storage,
                    b_pass_hist=b.radix_pass_hist.storage,
                    b_global_hist=b.radix_global_hist.storage,
                    n_buf=b.scan_total.storage,
                    radix_shift=radix_shift, key_word=key_word,
                    pass_idx=pass_idx, max_blocks=max_blocks,
                )
                self._dispatch(self.k_radix_scan, "radix_scan",
                    thread_count=[256 * _RADIX_SCAN_DIM, 1, 1],
                    b_pass_hist=b.radix_pass_hist.storage,
                    n_buf=b.scan_total.storage,
                    max_blocks=max_blocks,
                )
                self._dispatch_indirect(self.k_radix_downsweep, "radix_downsweep",
                    b.radix_dispatch_args,
                    keys_lo_in=src_lo.storage, keys_hi_in=src_hi.storage, values_in=src_v.storage,
                    keys_lo_out=dst_lo.storage, keys_hi_out=dst_hi.storage, values_out=dst_v.storage,
                    b_pass_hist=b.radix_pass_hist.storage,
                    b_global_hist=b.radix_global_hist.storage,
                    n_buf=b.scan_total.storage,
                    radix_shift=radix_shift, key_word=key_word,
                    pass_idx=pass_idx, max_blocks=max_blocks,
                )
            src_lo, dst_lo = dst_lo, src_lo
            src_hi, dst_hi = dst_hi, src_hi
            src_v, dst_v = dst_v, src_v
        return src_lo, src_hi, src_v

    def _run_forward(self, camera: Camera) -> None:
        self._upload_camera(camera)
        b = self.buffers
        num_tiles = b.num_tiles

        # Clear output textures + tile ranges.
        with self._marker_group("clear"):
            self._dispatch(self.k_clear_textures, "clear_textures",
                thread_count=[self.width, self.height, 1],
                output_image=b.output_image,
                gaussian_count_per_pixel=b.gaussian_count_per_pixel,
                frame_dim=spy.uint2(self.width, self.height),
            )
            self._dispatch(self.k_clear_tile_ranges, "clear_tile_ranges",
                thread_count=[num_tiles, 1, 1],
                tile_start=b.tile_start.storage,
                tile_end=b.tile_end.storage,
                num_tiles=num_tiles,
            )

        # Phase 0.5: cluster-frustum cull (Laine 2011, gated by
        # cfg.cluster_cull). Rebuild per-cluster AABBs from the current
        # gaussians and mark clusters fully outside the frustum. bin_count
        # below reads cluster_visible as its first early-out. When off,
        # cluster_visible stays seeded to all-ones (see buffers.allocate)
        # so bin_count's check is a no-op.
        #TODO: This should ideally also be gpu driven by an indirect dispatch
        if self.cfg.cluster_cull:
            num_clusters = (self.max_splats + CLUSTER_SIZE - 1) // CLUSTER_SIZE
            with self._marker_group("cluster_cull"):
                self._dispatch(self.k_build_cluster_aabb, "build_cluster_aabb",
                    thread_count=[num_clusters * CLUSTER_SIZE, 1, 1],
                    gaussians=b.gaussians.storage,
                    num_gaussians_counter=b.num_gaussians_counter.storage,
                    cluster_aabb_min=b.cluster_aabb_min.storage,
                    cluster_aabb_max=b.cluster_aabb_max.storage,
                    max_splats=self.max_splats,
                    num_clusters=num_clusters,
                )
                self._dispatch(self.k_cull_clusters_frustum, "cull_clusters_frustum",
                    thread_count=[num_clusters, 1, 1],
                    cluster_aabb_min=b.cluster_aabb_min.storage,
                    cluster_aabb_max=b.cluster_aabb_max.storage,
                    camera=self._camera_buffer,
                    cluster_visible=b.cluster_visible.storage,
                    num_clusters=num_clusters,
                )

        # Phase 1: per-gaussian tile count.
        with self._marker_group("bin_count"):
            self._dispatch(self.k_bin_count, "bin_count",
                thread_count=[self.max_splats, 1, 1],
                gaussians=b.gaussians.storage,
                num_gaussians_counter=b.num_gaussians_counter.storage,
                camera=self._camera_buffer,
                projected_2d=b.projected_2d.storage,
                blend_data=b.blend_data.storage,
                tile_count_per_gaussian=b.tile_count_per_gaussian.storage,
                cluster_visible=b.cluster_visible.storage,
                z_near=self.cfg.z_near_cull,
                max_splats=self.max_splats,
            )

        # Phase 2: exclusive scan → tile_offset_per_gaussian + total count.
        # scan_total[0] holds the total pair count on the GPU; no readback.
        with self._marker_group("scan_tile_offsets"):
            self._scan(
                n=self.max_splats,
                in_buf=b.tile_count_per_gaussian,
                out_buf=b.tile_offset_per_gaussian,
            )

        # Phase 3: emit (tile_id, depth, gaussian_idx) pairs into A buffers.
        # If the total_pair count is 0, bin_emit does nothing (per-gaussian
        # tile_count==0 → no writes). The downstream sort + tile_ranges
        # kernels early-out on i >= scan_total[0], so zero-pair frames are
        # correct without a CPU branch.
        with self._marker_group("bin_emit"):
            self._dispatch(self.k_bin_emit, "bin_emit",
                thread_count=[self.max_splats, 1, 1],
                gaussians=b.gaussians.storage,
                num_gaussians_counter=b.num_gaussians_counter.storage,
                camera=self._camera_buffer,
                projected_2d=b.projected_2d.storage,
                tile_offset_per_gaussian=b.tile_offset_per_gaussian.storage,
                keys_lo=b.sort_keys_lo_a.storage,
                keys_hi=b.sort_keys_hi_a.storage,
                values=b.sort_values_a.storage,
                cluster_visible=b.cluster_visible.storage,
                z_near=self.cfg.z_near_cull,
                max_sort_pairs=self.max_sort_pairs,
            )

        # Phase 4: global radix sort. Reads n from scan_total[0].
        with self._marker_group("radix_sort"):
            sorted_lo, sorted_hi, sorted_v = self._radix_sort_pairs()

        # Phase 5: identify per-tile start/end in the sorted array.
        with self._marker_group("identify_tile_ranges"):
            self._dispatch(self.k_identify_tile_ranges, "identify_tile_ranges",
                thread_count=[self.max_sort_pairs, 1, 1],
                keys_hi=sorted_hi.storage,
                tile_start=b.tile_start.storage,
                tile_end=b.tile_end.storage,
                n_buf=b.scan_total.storage,
                num_tiles=num_tiles,
            )

        # Stash the sorted-values buffer for backward/forward to consume.
        # Which side of the ping-pong won depends on the last pass; we
        # cached it in self._sorted_values for the backward.
        self._sorted_values = sorted_v

        # Phase 5b: per-tile bucket count + CSR offset scan for the ragged
        # forward-checkpoint buffer consumed by the transposed backward.
        # bucket_offsets[t] = exclusive scan of ceil(tile_len[t] / P);
        # its total (buckets used this frame) goes to bucket_offset_total.
        with self._marker_group("bucket_scan"):
            self._dispatch(self.k_bucket_count, "bucket_count",
                thread_count=[num_tiles, 1, 1],
                tile_start=b.tile_start.storage,
                tile_end=b.tile_end.storage,
                bucket_count=b.bucket_count.storage,
                num_tiles=num_tiles,
            )
            self._scan(
                n=num_tiles,
                in_buf=b.bucket_count,
                out_buf=b.bucket_offsets,
                total_buf=b.bucket_offset_total,
            )

        # Phase 6: forward rasterize.
        with self._marker_group("rasterize"):
            self._dispatch(self.k_forward, "forward",
                thread_count=[b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
                gaussians=b.gaussians.storage,
                projected_2d=b.projected_2d.storage,
                tile_start=b.tile_start.storage,
                tile_end=b.tile_end.storage,
                sorted_gaussian_ids=sorted_v.storage,
                camera=self._camera_buffer,
                bucket_offsets=b.bucket_offsets.storage,
                raster_checkpoints=b.raster_checkpoints.storage,
                output_image=b.output_image,
                gaussian_count_per_pixel=b.gaussian_count_per_pixel,
            )

    def render(self, camera: Camera) -> None:
        # Hold gpu_lock across the whole forward + drain: when an in-process
        # viewer shares this device, its render thread submits on the same
        # VkQueue. Without the lock, this forward + wait_for_idle races the
        # viewer's submits/presents (non-deterministic stall or device fault).
        # No-op lock in single-threaded use.
        with self.gpu_lock:
            self._run_forward(camera)
            self.device.wait_for_idle()

    def read_output_image(self) -> np.ndarray:
        with self.gpu_lock:
            return self.buffers.output_image.to_numpy()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _clear_grads(self) -> None:
        """Zero the 14 scalar param grads + 5 2D grad seeds. Full clear.

        Used by tests that need a clean grad state before an isolated
        backward() (see tests/test_gradients.py, tests/test_backward_persplat.py).
        Also used at pipeline setup via the training loop before iter 1
        (see :meth:`_clear_2d_grad_seeds` for the per-iter hot path)."""
        b = self.buffers
        for name in GRAD_SCALAR_COMPONENTS:
            self._dispatch(self.k_grad_clear, "grad_clear",
                thread_count=[self.max_splats, 1, 1],
                grad_buffer=b.grad_buffers[name].storage,
                max_splats=self.max_splats,
            )
        for name in b.grad_2d_seed_buffers:
            self._dispatch(self.k_grad_clear, "grad_clear",
                thread_count=[self.max_splats, 1, 1],
                grad_buffer=b.grad_2d_seed_buffers[name].storage,
                max_splats=self.max_splats,
            )

    def _clear_2d_grad_seeds(self) -> None:
        """Per-iter grad-clear for the HOT PATH: clears only the 5 2D grad
        seeds. Those are atomic-accumulated by the blend backward every iter
        and there's no other zeroing site, so they MUST be re-zeroed each iter.

        The 14 scalar param grad buffers (grad_mean_x ... grad_color_b) are
        NOT cleared here because the Adam step at end-of-iter already zeros
        them per-thread (adam.slang:140-144) as part of its per-gaussian
        update, so they're guaranteed zero at start-of-iter. This saves 14
        max_splats-sized dense-write dispatches per iter.

        Invariant: the caller must have run :meth:`_clear_grads` once at
        setup (before the first train_step) to seed the invariant, because
        NDBuffer allocator contents aren't guaranteed zero on construction."""
        b = self.buffers
        for name in b.grad_2d_seed_buffers:
            self._dispatch(self.k_grad_clear, "grad_clear",
                thread_count=[self.max_splats, 1, 1],
                grad_buffer=b.grad_2d_seed_buffers[name].storage,
                max_splats=self.max_splats,
            )

    def _run_loss(self) -> None:
        b = self.buffers
        target = self._active_target
        if self.cfg.lambda_ssim > 0:
            # Two-pass SSIM: pass 1 builds the m1/m2/m3 back-projection maps,
            # pass 2 convolves them and blends with the L1 + alpha gradient.
            # The 121-tap window loops are cache-bound, not tap-bound; a
            # separable 4-pass rewrite measured ~70% slower (global-scratch
            # round-trips beat the tap savings). See ssim_loss.slang header.
            with self._marker_group("ssim_maps"):
                self._dispatch(self.k_ssim_maps, "ssim_maps",
                    thread_count=[self.width, self.height, 1],
                    output_image=b.output_image,
                    target_image=target,
                    ssim_m1=b.ssim_m1,
                    ssim_m2=b.ssim_m2,
                    ssim_m3=b.ssim_m3,
                    frame_dim=spy.uint2(self.width, self.height),
                )
            with self._marker_group("ssim_grad"):
                self._dispatch(self.k_ssim_grad, "ssim_grad",
                    thread_count=[self.width, self.height, 1],
                    output_image=b.output_image,
                    target_image=target,
                    ssim_m1=b.ssim_m1,
                    ssim_m2=b.ssim_m2,
                    ssim_m3=b.ssim_m3,
                    dL_dC=b.dL_dC_texture,
                    frame_dim=spy.uint2(self.width, self.height),
                    lambda_ssim=self.cfg.lambda_ssim,
                    w_alpha=self.cfg.w_alpha,
                    lambda_l2=self.cfg.lambda_l2,
                )
        else:
            with self._marker_group("l1_l2_loss"):
                self._dispatch(self.k_loss, "loss",
                    thread_count=[self.width, self.height, 1],
                    output_image=b.output_image,
                    target_image=target,
                    dL_dC=b.dL_dC_texture,
                    frame_dim=spy.uint2(self.width, self.height),
                    w_alpha=self.cfg.w_alpha,
                    lambda_l2=self.cfg.lambda_l2,
                )

    def _run_backward(self, structure_tensor=None,
                      structure_kwargs: dict | None = None) -> None:
        b = self.buffers
        gb = b.grad_buffers
        # The forward left the sorted gaussian-id buffer on one of the two
        # ping-pong slots. We cache it during _run_forward.
        sorted_values = self._sorted_values
        with self._marker_group("rasterize_backward"):
            if self.use_persplat_backward:
                # Transposed per-splat backward: one workgroup (256 threads)
                # per tile, splat-parallel with checkpoint reconstruction.
                num_tiles = b.num_tiles_x * b.num_tiles_y
                self._dispatch(self.k_backward_persplat, "backward_persplat",
                    thread_count=[num_tiles * 256, 1, 1],
                    blend_data=b.blend_data.storage,
                    tile_start=b.tile_start.storage,
                    tile_end=b.tile_end.storage,
                    sorted_gaussian_ids=sorted_values.storage,
                    bucket_offsets=b.bucket_offsets.storage,
                    raster_checkpoints=b.raster_checkpoints.storage,
                    camera=self._camera_buffer,
                    output_image=b.output_image,
                    gaussian_count_per_pixel=b.gaussian_count_per_pixel,
                    dL_dC_texture=b.dL_dC_texture,
                    **b.grad_seed_kwargs(),
                    grad_opacity=gb["opacity"].storage,
                    grad_color_r=gb["color_r"].storage,
                    grad_color_g=gb["color_g"].storage,
                    grad_color_b=gb["color_b"].storage,
                    grad_2d_x_iter=b.grad_2d_x_iter.storage,
                    grad_2d_y_iter=b.grad_2d_y_iter.storage,
                    # SGD pixel-subsampling phase; rotates each backward so the
                    # visited subset alternates (no-op at BACKWARD_PIXEL_STRIDE=1).
                    pixel_phase=self._backward_pixel_phase,
                )
                self._backward_pixel_phase += 1
                # Per-gaussian projection backward: turn the summed 2D grads
                # into 3D mean/scale/rot grads, ONCE per gaussian. Must run
                # before regularization (which InterlockedAdds onto the 3D
                # grads this kernel direct-stores).
                with self._marker_group("projection_backward"):
                    self._dispatch(self.k_projection_backward, "projection_backward",
                        thread_count=[self.max_splats, 1, 1],
                        gaussians=b.gaussians.storage,
                        num_gaussians_counter=b.num_gaussians_counter.storage,
                        camera=self._camera_buffer,
                        **b.grad_seed_kwargs(),
                        grad_mean_x=gb["mean_x"].storage,
                        grad_mean_y=gb["mean_y"].storage,
                        grad_mean_z=gb["mean_z"].storage,
                        grad_log_scale_x=gb["log_scale_x"].storage,
                        grad_log_scale_y=gb["log_scale_y"].storage,
                        grad_log_scale_z=gb["log_scale_z"].storage,
                        grad_rot_w=gb["rot_w"].storage,
                        grad_rot_x=gb["rot_x"].storage,
                        grad_rot_y=gb["rot_y"].storage,
                        grad_rot_z=gb["rot_z"].storage,
                        cluster_visible=b.cluster_visible.storage,
                        max_splats=self.max_splats,
                    )
            else:
                self._dispatch(self.k_backward, "backward",
                    thread_count=[b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
                    gaussians=b.gaussians.storage,
                    projected_2d=b.projected_2d.storage,
                    tile_start=b.tile_start.storage,
                    tile_end=b.tile_end.storage,
                    sorted_gaussian_ids=sorted_values.storage,
                    camera=self._camera_buffer,
                    output_image=b.output_image,
                    gaussian_count_per_pixel=b.gaussian_count_per_pixel,
                    dL_dC_texture=b.dL_dC_texture,
                    **b.grad_kwargs(),
                    grad_2d_x_iter=b.grad_2d_x_iter.storage,
                    grad_2d_y_iter=b.grad_2d_y_iter.storage,
                )
        # SAD-GS online eta accumulation (must be HERE: after backward filled
        # grad_2d_*_iter, before accumulate_iter clears them, so the per-view
        # grad gate in score_structure_aware sees this view's gradient).
        if structure_tensor is not None:
            with self._marker_group("structure_accum"):
                _densify.accumulate_structure_scores(
                    self, structure_tensor, **(structure_kwargs or {}))

        # Fold the per-iter 2D grad vector into the window-level
        # mag + visibility accumulators. Gsplat semantics: the signal
        # driving densification is the NORM OF THE SUM of per-pixel
        # gradients per iter, averaged over iters the gaussian was
        # visible — not the sum of per-pixel norms.
        with self._marker_group("accumulate_iter"):
            self._dispatch(self.k_accumulate_iter, "accumulate_iter",
                thread_count=[self.max_splats, 1, 1],
                grad_2d_x_iter=b.grad_2d_x_iter.storage,
                grad_2d_y_iter=b.grad_2d_y_iter.storage,
                grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
                visibility_count_accum=b.visibility_count_accum.storage,
                frame_dim=spy.float2(self.width, self.height),
                max_splats=self.max_splats,
            )

    def _run_regularization(self) -> None:
        if self.cfg.opacity_reg <= 0 and self.cfg.scale_reg <= 0:
            return
        b = self.buffers
        gb = b.grad_buffers
        self._dispatch(self.k_regularization, "regularization",
            thread_count=[self.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            grad_opacity=gb["opacity"].storage,
            grad_log_scale_x=gb["log_scale_x"].storage,
            grad_log_scale_y=gb["log_scale_y"].storage,
            grad_log_scale_z=gb["log_scale_z"].storage,
            cluster_visible=b.cluster_visible.storage,
            w_opacity=self.cfg.opacity_reg,
            w_scale=self.cfg.scale_reg,
            inv_n=1.0,
        )

    def clear_tracking(self) -> None:
        """Zero the v2 densification tracking accumulators. Called at the
        start of each densification window (every N iters)."""
        b = self.buffers
        self._dispatch(self.k_clear_tracking, "clear_tracking",
            thread_count=[self.max_splats, 1, 1],
            grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
            visibility_count_accum=b.visibility_count_accum.storage,
            grad_2d_x_iter=b.grad_2d_x_iter.storage,
            grad_2d_y_iter=b.grad_2d_y_iter.storage,
            max_splats=self.max_splats,
        )

    def read_tracking(self) -> dict[str, np.ndarray]:
        """Read window-level tracking buffers (int64 fixed-point)."""
        b = self.buffers
        raw_mag = b.grad_mean_2d_mag_accum.to_numpy()
        raw_vis = b.visibility_count_accum.to_numpy()
        mag_int = np.frombuffer(raw_mag.tobytes(), dtype=np.int64)
        vis = np.frombuffer(raw_vis.tobytes(), dtype=np.uint32)
        return {
            "grad_mean_2d_mag": mag_int.astype(np.float64) / 65536.0,
            "visibility_count": vis.astype(np.int64),
        }

    def _run_adam(self, lrs: AdamLRs) -> None:
        self.iteration += 1
        step = float(self.iteration)
        bc1 = 1.0 - 0.9 ** step
        bc2 = 1.0 - 0.999 ** step

        b = self.buffers

        # Populate adam_dispatch_args from num_gaussians_counter so we dispatch
        # only ceil(count/256) workgroups instead of ceil(max_splats/256).
        # Reuses the existing prepareDensifyArgs shader (same math).
        self._dispatch(self.k_densify_prepare_args, "adam_prepare_args",
            thread_count=[1, 1, 1],
            count_buf=b.num_gaussians_counter.storage,
            dispatch_args=b.adam_dispatch_args.storage,
            max_splats=self.max_splats,
        )

        # Indirect: the driver reads (groups_x, 1, 1) from adam_dispatch_args
        # at submit time — so if densify grows the count between iters, next
        # iter's dispatch automatically covers the new range without any
        # host-side count read.
        self._dispatch_indirect(self.k_adam, "adam", b.adam_dispatch_args,
            gaussians=b.gaussians.storage,
            adam_state=b.adam_state.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            **b.grad_kwargs(),
            cluster_visible=b.cluster_visible.storage,
            cfg={
                "lr_mean": lrs.mean,
                "lr_scale": lrs.scale,
                "lr_rot": lrs.rot,
                "lr_opacity": lrs.opacity,
                "lr_color": lrs.color,
                "bias_correction1": bc1,
                "bias_correction2": bc2,
            },
            pixels_per_step=float(self.width * self.height),
        )

    def train_step(self, camera: Camera, lrs: AdamLRs | None = None,
                   structure_tensor=None, structure_kwargs: dict | None = None) -> None:
        """Run one training step: forward → loss → backward → regularization → adam.

        The target image must have been uploaded via
        :meth:`set_target_image` before calling.

        ``structure_tensor`` (SAD-GS online accumulation): when provided (the
        current view's structure-tensor field), a per-gaussian eta sample is
        accumulated into the eta buffers DURING backward — after the per-view
        2D gradient is computed (so the SADGS per-view grad gate can see it) and
        before accumulate_iter clears it. Over a densify window this builds the
        multiview N_high/N_low/eta_max statistics the paper accumulates (Section
        3.6). None (default) = no SAD overhead. ``structure_kwargs`` forwards
        per-call options (pass_idx, opacity_threshold, per_view_grad_thresh) to
        accumulate_structure_scores; the caller gates cadence by simply passing
        structure_tensor=None on iters it wants to skip.

        NO CPU/GPU sync here — training submits kernels and returns
        immediately. This keeps the GPU saturated; the queue drains
        when the training loop next reads back data (loss sample,
        preview render, densify).
        """
        if lrs is None:
            lrs = AdamLRs()
        enc = self.device.create_command_encoder()
        self._encoder = enc
        with self._marker_group("clear_grads"):
            # Hot path: only zero the 5 2D grad seeds. The 14 scalar param
            # grad buffers are already zero because Adam zero'd them per-thread
            # at the end of the previous iter (see _clear_2d_grad_seeds docs).
            # Callers must ensure the scalar-grad invariant is seeded by
            # running _clear_grads() once at pipeline setup.
            self._clear_2d_grad_seeds()
        with self._marker_group("forward"):
            self._run_forward(camera)
        with self._marker_group("loss"):
            self._run_loss()
        with self._marker_group("backward"):
            self._run_backward(structure_tensor=structure_tensor,
                               structure_kwargs=structure_kwargs)
        with self._marker_group("regularization"):
            self._run_regularization()
        with self._marker_group("adam"):
            self._run_adam(lrs)
        cmd = enc.finish()
        with self.gpu_lock:
            self.last_submit_id = self.device.submit_command_buffer(cmd)
        self._encoder = None

    # ------------------------------------------------------------------
    # Densification (strategies + scoring live in trainer.densify; these
    # thin methods delegate, keeping the call surface on Pipeline while the
    # heavy logic sits next to the buffer/scan structures it orchestrates).
    # ------------------------------------------------------------------

    def compute_importance_scores(self, cameras, target_textures, cfg) -> None:
        return _densify.compute_importance_scores(self, cameras, target_textures, cfg)

    def densify(self, cfg, densify_step, n_densify_steps, init_count) -> dict:
        return _densify.densify(self, cfg, densify_step, n_densify_steps, init_count)

    def densify_simple(self, cfg, grad_thresh=0.0002, scale_thresh=None,
                       prune_big_scale=None, allow_scale_prune=True) -> dict:
        return _densify.densify_simple(self, cfg, grad_thresh, scale_thresh,
                                       prune_big_scale, allow_scale_prune)

    def reset_opacity(self, cfg) -> None:
        return _densify.reset_opacity(self, cfg)

    def morton_sort(self, world_min, world_max) -> None:
        """Reorder gaussians + adam_state by 3D Morton code (Z-order).
        See :func:`trainer.densify.morton_sort`."""
        return _densify.morton_sort(self, world_min, world_max)

    def compute_structure_scores(self, cameras, structure_textures, cfg) -> None:
        return _densify.compute_structure_scores(self, cameras, structure_textures, cfg)

    def clear_structure_scores(self) -> None:
        return _densify.clear_structure_scores(self)

    def densify_structure_aware(self, cfg) -> dict:
        return _densify.densify_structure_aware(self, cfg)

    def _scan(self, n: int,
              in_buf: spy.NDBuffer | None = None,
              out_buf: spy.NDBuffer | None = None,
              total_buf: spy.NDBuffer | None = None) -> None:
        """Run a multi-level Blelloch scan on ``in_buf`` into ``out_buf``,
        writing the total into ``total_buf[0]`` (default ``scan_total[0]``).

        ``total_buf`` lets a caller redirect the scan total to a dedicated
        buffer instead of the shared ``scan_total`` (which holds num_rendered
        during the forward pass and must not be clobbered by a side scan).

        Supports up to SCAN_BLOCK_SIZE^3 (~1 billion) elements via 3 levels.
        """
        b = self.buffers
        if in_buf is None:
            in_buf = b.valid_mask_buffer
        if out_buf is None:
            out_buf = b.scan_buffer
        if total_buf is None:
            total_buf = b.scan_total

        num_blocks = max(1, (n + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE)

        # Level 1: per-block scan, producing block_sums[num_blocks].
        self._dispatch(self.k_scan_block, "scan_block",
            thread_count=[num_blocks * SCAN_BLOCK_SIZE, 1, 1],
            in_buf=in_buf.storage,
            out_buf=out_buf.storage,
            block_sums=b.scan_block_sums.storage,
            n=n,
        )

        if num_blocks <= SCAN_BLOCK_SIZE:
            # 2-level: single-workgroup scan of block sums.
            self._dispatch(self.k_scan_block_sums, "scan_block_sums",
                thread_count=[SCAN_BLOCK_SIZE, 1, 1],
                block_sums=b.scan_block_sums.storage,
                total_out=total_buf.storage,
                num_blocks=num_blocks,
            )
        else:
            # 3-level: scan block_sums using the same 2-level pattern.
            num_l2_blocks = max(1, (num_blocks + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE)
            # Scan block_sums → block_sums (in-place prefix), with l2 block sums.
            self._dispatch(self.k_scan_block, "scan_block",
                thread_count=[num_l2_blocks * SCAN_BLOCK_SIZE, 1, 1],
                in_buf=b.scan_block_sums.storage,
                out_buf=b.scan_block_sums.storage,
                block_sums=b.scan_block_sums_l2.storage,
                n=num_blocks,
            )
            self._dispatch(self.k_scan_block_sums, "scan_block_sums",
                thread_count=[SCAN_BLOCK_SIZE, 1, 1],
                block_sums=b.scan_block_sums_l2.storage,
                total_out=total_buf.storage,
                num_blocks=num_l2_blocks,
            )
            self._dispatch(self.k_add_block_offsets, "add_block_offsets",
                thread_count=[num_l2_blocks * SCAN_BLOCK_SIZE, 1, 1],
                out_buf=b.scan_block_sums.storage,
                block_sums=b.scan_block_sums_l2.storage,
                n=num_blocks,
            )

        # Final: add block offsets to per-element results.
        self._dispatch(self.k_add_block_offsets, "add_block_offsets",
            thread_count=[num_blocks * SCAN_BLOCK_SIZE, 1, 1],
            out_buf=out_buf.storage,
            block_sums=b.scan_block_sums.storage,
            n=n,
        )

