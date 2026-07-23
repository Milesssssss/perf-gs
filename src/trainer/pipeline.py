"""Dispatch orchestration for the v1 trainer.

A ``Pipeline`` owns a device, the compute kernels, and the GPU resources.
For inference, call :meth:`Pipeline.render`. For training, set a target
image with :meth:`Pipeline.set_target_image` and call
:meth:`Pipeline.train_step`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import slangpy as spy

from trainer import buffers
from trainer import densify as _densify
from trainer.buffers import (
    CLUSTER_SIZE,
    GRAD_SCALAR_COMPONENTS,
    RADIX_GLOBAL_HIST_SIZE,
    RADIX_PART_SIZE,
)
from trainer.config import (
    AdamLRs,
    CompositorConfig,
    LossConfig,
    RasterConfig,
    RegularizationConfig,
)
from trainer.dataset import Camera
from trainer.densify import DensifyConfig
from trainer.device import SHADERS_DIR
from trainer.gpu import CommandBatch, GpuRuntime
from trainer.gpu_scan import GpuScan, ScanKernels
from trainer.kernels import load_kernels
from trainer.losses import ImageLoss
from trainer.rasterizer import RasterizationContext
from trainer.simple_densifier import (
    CommandInterface,
    SimpleDensifier,
    SimpleDensifyKernels,
)

# DensifyConfig / AdamLRs are imported (not defined) here and re-exported for
# back-compat: existing callers do ``from trainer.pipeline import ...``.
__all__ = ["Pipeline", "DensifyConfig", "AdamLRs"]


# DeviceRadixSort dispatch dims — must match shaders/sort/device_radix_sort.slang.
# (The layout-sizing constants RADIX_PART_SIZE / RADIX_GLOBAL_HIST_SIZE /
# SCAN_BLOCK_SIZE are imported from buffers.py; these two are used only in
# thread_count computations here, not in allocation.)
_RADIX_US_DIM = 128
_RADIX_SCAN_DIM = 128


class _RecordingTransaction:
    """A recorded batch plus host-state commits that run after submit succeeds."""

    def __init__(self, batch: CommandBatch) -> None:
        self.batch = batch
        self._commits: list[Callable[[], None]] = []

    def after_submit(self, commit: Callable[[], None]) -> None:
        self._commits.append(commit)

    def commit(self) -> None:
        for commit in self._commits:
            commit()


def _module_for_reflection(device: spy.Device) -> spy.Module:
    """Load the NEUTRAL shared-layout module (``common/types.slang``) so
    buffers.py can reflect the primitive/projected/optimizer struct layouts
    (``Gaussian3D`` / ``Projected2DGaussian`` / ``BlendData`` / ``AdamState``)
    without depending on an optimizer file. types.slang is the single source of
    those layouts — every shader that needs them includes it — and buffers sizes
    every NDBuffer from this reflection so Python never duplicates the layout.
    (Reflecting through optimizers/adam.slang worked only because it transitively
    includes types.slang; pointing straight at the shared module removes that
    accidental optimizer coupling. Layout is byte-identical — see
    tests/test_reflection_layout.py.)"""
    return spy.Module.load_from_file(
        device, str(SHADERS_DIR / "common" / "types.slang")
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
        raster_config: RasterConfig | None = None,
        compositor_config: CompositorConfig | None = None,
        loss_config: LossConfig | None = None,
        regularization_config: RegularizationConfig | None = None,
        viewer_only: bool = False,
    ) -> None:
        self.device = device
        # Serializes GPU queue access (submit/present/wait_for_idle) against
        # any OTHER thread sharing this device, notably the live Viewer, which
        # runs its render loop on a background thread.
        self.gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()
        # GPU runtime: owns the device + queue-lock discipline and hands out CommandBatches
        self.runtime = GpuRuntime(device, self.gpu_lock)
        self.width = width
        self.height = height
        self.max_splats = max_splats
        if max_sort_pairs is None:
            max_sort_pairs = max(buffers.DEFAULT_MAX_SORT_PAIRS, max_splats * 2)
        self.max_sort_pairs = max_sort_pairs

        self.shared_data_module = _module_for_reflection(device)
        self.viewer_only = viewer_only
        self.raster_config = raster_config or RasterConfig()
        if viewer_only and self.raster_config.cluster_cull:
            self.raster_config = replace(self.raster_config, cluster_cull=False)
        self.compositor_config = compositor_config or CompositorConfig()
        self.regularization_config = (
            regularization_config or RegularizationConfig())
        resolved_loss_config = loss_config or LossConfig()

        load_kernels(device, self, viewer_only=viewer_only)

        allocate_fn = buffers.allocate_viewer if viewer_only else buffers.allocate

        self.buffers = allocate_fn(
            device=device, shared_data_module=self.shared_data_module,
            max_splats=max_splats, width=width, height=height,
            max_sort_pairs=max_sort_pairs,
        )

        # Image-space loss (SSIM / L1+L2), recorded independently of how the
        # pixels were produced. viewer_only skips it: allocate_viewer builds no
        # loss kernels, SSIM maps, or dL_dC texture, and _run_loss never runs.
        self.image_loss = None if viewer_only else ImageLoss(
            ssim_maps_kernel=self.k_ssim_maps,
            ssim_grad_kernel=self.k_ssim_grad,
            loss_kernel=self.k_loss,
            ssim_m1=self.buffers.ssim_m1,
            ssim_m2=self.buffers.ssim_m2,
            ssim_m3=self.buffers.ssim_m3,
            image_gradient=self.buffers.dL_dC_texture,
            width=width, height=height, runtime=self.runtime,
            config=resolved_loss_config,
        )

        # Camera buffer: raw 144-byte buffer (136 data + 8 padding to 16-byte align).
        # Using a raw Buffer avoids reflection dependency (survives hot-reload).
        self._camera_buffer = device.create_buffer(
            size=144,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label="camera_data",
        )

        # Training step counter (Python-side, for Adam bias correction).
        #TODO: should the iteration be part of the Pipeline? I doubt it
        self.iteration = 0

        # Active command batch during a recorded step. When set, _dispatch /
        # _marker_group record into it; None => immediate dispatch. External
        # encoder owners (densify, viewer) assign a raw encoder via the
        # `_encoder` compat property below, which wraps it in a CommandBatch.
        self._batch: CommandBatch | None = None
        self.scan_service = GpuScan(
            resources=self.buffers.scan,
            kernels=ScanKernels(
                scan_block=self.k_scan_block,
                scan_block_sums=self.k_scan_block_sums,
                add_block_offsets=self.k_add_block_offsets,
            ),
            runtime=self.runtime,
            active_batch=lambda: self._batch,
        )
        self.simple_densifier = SimpleDensifier(
            primitive=self.buffers.primitive,
            adam=self.buffers._require_adam(),
            resources=self.buffers._require_densification(),
            eta=self.buffers.eta,
            scan=self.scan_service,
            kernels=SimpleDensifyKernels(
                decide_actions=self.k_decide_actions_simple,
                snapshot_count=self.k_densify_snapshot,
                prepare_dispatch=self.k_densify_prepare_args,
                split_and_clone=self.k_split_and_clone,
                build_prune_mask=self.k_build_prune_mask,
                scatter_compact=self.k_scatter_compact,
                scatter_compact_eta=self.k_scatter_compact_eta,
            ),
            commands=CommandInterface(
                dispatch=self.dispatch,
                dispatch_indirect=self.dispatch_indirect,
                recording=self.recording,
                marker_group=self.marker_group,
            ),
            width=self.width,
            height=self.height,
            max_primitives=self.max_splats,
        ) if not viewer_only else None

        self.last_submit_id: int = 0

        self.use_persplat_backward = True
        self.use_flat_bucket_backward = False

        if self.raster_config.mip_enabled and not self.use_persplat_backward:
            raise RuntimeError(
                "mip_enabled requires use_persplat_backward=True: the per-pixel "
                "backward does not propagate opacity-compensation gradients.")

        if (self.compositor_config.extinction_hardness != 1.0
                and not self.use_persplat_backward):
            raise RuntimeError(
                "extinction_hardness != 1 requires use_persplat_backward=True: "
                "the per-pixel backward does not implement hardened extinction.")

        #starts at 0, increments after each bwd step
        self._backward_pixel_phase = 0

        self._extinction_hardness = float(
            self.compositor_config.extinction_hardness)

        self._active_target = None if viewer_only else self.buffers.target_image

        self._sorted_values = self.buffers.sort_values_a

        self._mip_cameras_buffer: spy.Buffer | None = None
        self._mip_num_cameras: int = 0
        self._mip_filter_coef: float = 0.0

    def on_hot_reload(self) -> None:
        """Refresh reflection-dependent objects after device.reload_all_programs().
        Camera buffer is raw bytes so it survives reload. Only the module
        reference needs refreshing (for any code that uses reflected types)."""
        self.shared_data_module = _module_for_reflection(self.device)

    @property
    def _encoder(self) -> "spy.CommandEncoder | None":
        """Compat view of the active command batch's raw encoder. External
        encoder owners (densify, viewer) assign a raw encoder here; the setter
        wraps it in a CommandBatch so ALL dispatch routing goes through one
        path. Internal code uses ``self._batch`` directly."""
        return self._batch.encoder if self._batch is not None else None

    @_encoder.setter
    def _encoder(self, encoder: "spy.CommandEncoder | None") -> None:
        self._batch = CommandBatch(encoder) if encoder is not None else None

    def _dispatch(self, kernel, marker: str, **kwargs):
        """Dispatch a kernel, tagging it with a flat debug marker ``marker`` when
        a command batch is active (train_step / densify / viewer); otherwise an
        immediate dispatch. Recording is delegated to the active CommandBatch."""
        if self._batch is not None:
            self._batch.dispatch(kernel, marker, **kwargs)
        else:
            transient = self.runtime.command_batch(marker)
            transient.dispatch(kernel, marker, **kwargs)
            self.last_submit_id = self.runtime.submit(transient).submit_id

    def _dispatch_indirect(self, kernel, marker: str, args_buf: spy.NDBuffer,
                           offset: int = 0, **kwargs) -> None:
        """Indirect compute dispatch (group count read from ``args_buf`` at
        ``offset`` as three uint32s). Records into the active batch; if none is
        active (e.g. from render()) a transient one-shot batch is created and
        submitted so this works from any caller. The write-visibility global
        barrier before the indirect fetch lives in CommandBatch.dispatch_indirect."""
        if self._batch is not None:
            self._batch.dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)
        else:
            transient = self.runtime.command_batch(marker)
            transient.dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)
            self.last_submit_id = self.runtime.submit(transient).submit_id

    def _marker_group(self, label: str):
        """Nested debug-group scope for profiler captures (Nsight etc.). No-op
        (nullcontext) when no batch is active, so it costs nothing outside a
        recorded step; otherwise delegates to CommandBatch.marker_group."""
        if self._batch is None:
            return contextlib.nullcontext()
        return self._batch.marker_group(label)

    # Public command contract for extracted GPU services. The private aliases
    # remain while Pipeline's internal implementation is migrated incrementally.
    def dispatch(self, kernel, marker: str, **kwargs) -> None:
        self._dispatch(kernel, marker, **kwargs)

    def dispatch_indirect(self, kernel, marker: str, args_buf: spy.NDBuffer,
                          offset: int = 0, **kwargs) -> None:
        self._dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)

    def marker_group(self, label: str):
        return self._marker_group(label)

    def recording(self, label: str):
        return self._recording(label)

    @property
    def camera_buffer(self):
        """Read-only uploaded camera resource for specialized GPU services."""
        return self._camera_buffer

    def rasterize(self, camera: Camera) -> RasterizationContext:
        """Record one forward rasterization and return its explicit handoff."""
        return self._run_forward(camera)

    def radix_sort_pairs(self) -> tuple[spy.NDBuffer, spy.NDBuffer, spy.NDBuffer]:
        """Record the shared 64-bit-key radix sort service."""
        return self._radix_sort_pairs()

    @contextlib.contextmanager
    def _recording(self, label: str):
        """Record GPU work into a fresh command batch installed as the active
        batch, then submit it once on success.

        The active-batch install is cleared in a ``finally`` so a mid-recording
        exception can't leave the pipeline wedged with a stale batch; the submit
        (and the resulting ``last_submit_id``) happens only if the body completed
        without raising. Used by ``train_step`` and the densify passes, which
        record via ``_dispatch`` / ``_marker_group`` into the installed batch.
        """
        batch = self.runtime.command_batch(label)
        transaction = _RecordingTransaction(batch)
        self._batch = batch
        try:
            yield transaction
            self.last_submit_id = self.runtime.submit(batch).submit_id
            transaction.commit()
        finally:
            self._batch = None

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
        # Zero the Adam moments device-side (one clear kernel over the AdamState
        # buffer). Replaces a full GPU->CPU readback that existed only to size a
        # host zero-array — wasteful (>1 GB at a large --max-splats).
        self._dispatch(self.k_clear_adam_state, "clear_adam_state",
            thread_count=[self.max_splats, 1, 1],
            adam_state=self.buffers.adam_state.storage,
            max_splats=self.max_splats)
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

    def _run_forward(self, camera: Camera) -> RasterizationContext:
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
        if self.raster_config.cluster_cull:
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
                filter_3D=b.filter_3D.storage,
                mip_enabled=1 if self.raster_config.mip_enabled else 0,
                kernel_size=self.raster_config.kernel_size,
                z_near=self.raster_config.z_near_cull,
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
                z_near=self.raster_config.z_near_cull,
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
                blend_data=b.blend_data.storage,
                tile_start=b.tile_start.storage,
                tile_end=b.tile_end.storage,
                sorted_gaussian_ids=sorted_v.storage,
                camera=self._camera_buffer,
                bucket_offsets=b.bucket_offsets.storage,
                raster_checkpoints=b.raster_checkpoints.storage,
                output_image=b.output_image,
                gaussian_count_per_pixel=b.gaussian_count_per_pixel,
                extinction_hardness=float(self._extinction_hardness),
            )

        # Explicit forward -> backward handoff (was the hidden self._sorted_values,
        # which is still set above for importance scoring + introspection). The
        # backward takes this context so the dependency is a passed value.
        return RasterizationContext(sorted_ids=sorted_v)

    def set_extinction_hardness(self, hardness: float) -> None:
        """Set the compositor extinction hardness used by subsequent forward /
        backward dispatches. Lets the trainer SCHEDULE k across training (e.g.
        hold at 1.0 until the first densify, then ramp to the target) without
        rebuilding the frozen CompositorConfig. Its extinction_hardness stays
        the run TARGET; the construction guard already required the per-splat
        backward for a target != 1, so any scheduled value in [1, target] is
        honored by that path."""
        self._extinction_hardness = float(hardness)

    def render(self, camera: Camera) -> None:
        with self.recording("render"):
            self.rasterize(camera)
        self.runtime.wait_for_idle()

    def read_output_image(self) -> np.ndarray:
        with self.gpu_lock:
            return self.buffers.output_image.to_numpy()

    def read_bin_debug(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Read the binning-debug buffers the LAST forward produced, for the
        --debug-bin-viz visualization. Returns:
          (tile_counts[num_tiles], overlap[H, W], num_tiles_x, num_tiles_y)
        where tile_counts[t] = tile_end[t] - tile_start[t] (splats binned into
        tile t) and overlap = gaussian_count_per_pixel (splats blended per pixel
        before early-break). Both are already filled every frame — no extra
        kernel. Call after render()/wait_for_idle()."""
        b = self.buffers
        with self.gpu_lock:
            starts = b.tile_start.to_numpy().view(np.uint32)
            ends = b.tile_end.to_numpy().view(np.uint32)
            overlap = b.gaussian_count_per_pixel.to_numpy().view(np.uint32)
        n = b.num_tiles_x * b.num_tiles_y
        tile_counts = (ends[:n].astype(np.int64) - starts[:n].astype(np.int64))
        tile_counts = np.clip(tile_counts, 0, None).astype(np.uint32)
        overlap = overlap.reshape(self.height, self.width)
        return tile_counts, overlap, b.num_tiles_x, b.num_tiles_y

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
        self.image_loss.record(
            self._batch,
            rendered=self.buffers.output_image,
            target=self._active_target,
        )

    def _run_backward(self, context: "RasterizationContext | None" = None,
                      structure_tensor=None,
                      structure_kwargs: dict | None = None) -> None:
        b = self.buffers
        gb = b.grad_buffers
        # Explicit forward -> backward handoff: the sorted gaussian-id buffer (the
        # radix ping-pong winner) comes from the forward's RasterizationContext.
        # Falls back to self._sorted_values for callers that split forward/backward
        # without threading the context (e.g. the test kit's backward()).
        sorted_values = context.sorted_ids if context is not None else self._sorted_values
        with self._marker_group("rasterize_backward"):
            if self.use_persplat_backward:
                num_tiles = b.num_tiles_x * b.num_tiles_y
                if self.use_flat_bucket_backward:
                    # Flat warp-per-bucket backward. Build the compacted
                    # live-bucket list (only buckets a pixel actually reached),
                    # then dispatch ONE WARP PER BUCKET indirectly. Same grads
                    # as the per-tile path, no straggler tail / idle warps.
                    with self._marker_group("live_bucket_compaction"):
                        self._dispatch(self.k_tile_max_processed, "tile_max_processed",
                            thread_count=[num_tiles * 256, 1, 1],
                            gaussian_count_per_pixel=b.gaussian_count_per_pixel,
                            live_bucket_count=b.live_bucket_count.storage,
                            frame_dim_px=spy.uint2(self.width, self.height),
                            num_tiles_x=b.num_tiles_x,
                            num_tiles=num_tiles,
                        )
                        self._scan(
                            n=num_tiles,
                            in_buf=b.live_bucket_count,
                            out_buf=b.live_bucket_offsets,
                            total_buf=b.live_bucket_total,
                        )
                        self._dispatch(self.k_scatter_live_buckets, "scatter_live_buckets",
                            thread_count=[num_tiles, 1, 1],
                            live_bucket_count=b.live_bucket_count.storage,
                            live_bucket_offsets=b.live_bucket_offsets.storage,
                            live_buckets=b.live_buckets.storage,
                            num_tiles=num_tiles,
                        )
                        max_buckets = (
                            buffers._ceil_div(self.max_sort_pairs, buffers.RASTER_BUCKET_SIZE)
                            + num_tiles)
                        self._dispatch(self.k_bucket_prepare_dispatch, "bucket_prepare_dispatch",
                            thread_count=[1, 1, 1],
                            live_bucket_total=b.live_bucket_total.storage,
                            dispatch_args=b.bucket_dispatch_args.storage,
                            max_buckets=max_buckets,
                        )
                    self._dispatch_indirect(self.k_backward_persbucket, "backward_persbucket",
                        args_buf=b.bucket_dispatch_args,
                        blend_data=b.blend_data.storage,
                        tile_start=b.tile_start.storage,
                        tile_end=b.tile_end.storage,
                        sorted_gaussian_ids=sorted_values.storage,
                        bucket_offsets=b.bucket_offsets.storage,
                        live_buckets=b.live_buckets.storage,
                        live_bucket_total=b.live_bucket_total.storage,
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
                        pixel_phase=self._backward_pixel_phase,
                        extinction_hardness=float(self._extinction_hardness),
                    )
                else:
                  # Transposed per-splat backward: one workgroup (256 threads)
                  # per tile, splat-parallel with checkpoint reconstruction.
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
                    extinction_hardness=float(self._extinction_hardness),
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
                        filter_3D=b.filter_3D.storage,
                        mip_enabled=1 if self.raster_config.mip_enabled else 0,
                        kernel_size=self.raster_config.kernel_size,
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
        config = self.regularization_config
        if config.opacity_weight <= 0 and config.scale_weight <= 0:
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
            w_opacity=config.opacity_weight,
            w_scale=config.scale_weight,
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
        """Run one training step: forward -> loss -> backward -> regularization -> adam.

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

        NO CPU/GPU sync here, training submits kernels and returns
        immediately. This keeps the GPU saturated; the queue drains
        when the training loop next reads back data (loss sample,
        preview render, densify).
        """
        if lrs is None:
            lrs = AdamLRs()
        with self._recording("train_step"):
            with self._marker_group("clear_grads"):
                # Hot path: only zero the 5 2D grad seeds. The 14 scalar param
                # grad buffers are already zero because Adam zero'd them per-thread
                # at the end of the previous iter (see _clear_2d_grad_seeds docs).
                # Callers must ensure the scalar-grad invariant is seeded by
                # running _clear_grads() once at pipeline setup.
                self._clear_2d_grad_seeds()
            with self._marker_group("forward"):
                raster_ctx = self._run_forward(camera)
            with self._marker_group("loss"):
                self._run_loss()
            with self._marker_group("backward"):
                self._run_backward(raster_ctx, structure_tensor=structure_tensor,
                                   structure_kwargs=structure_kwargs)
            with self._marker_group("regularization"):
                self._run_regularization()
            with self._marker_group("adam"):
                self._run_adam(lrs)

    # ------------------------------------------------------------------
    # Densification (structure/reset/Morton helpers live in trainer.densify;
    # simple densification is an injected service. These
    # thin methods delegate, keeping the call surface on Pipeline while the
    # heavy logic sits next to the buffer/scan structures it orchestrates).
    # ------------------------------------------------------------------

    def densify_simple(self, cfg, grad_thresh=0.0002, scale_thresh=None,
                       prune_big_scale=None, allow_scale_prune=True,
                       carry_eta=False) -> dict:
        if self.simple_densifier is None:
            raise RuntimeError("densification is unavailable in viewer-only mode")
        return self.simple_densifier.run(
            cfg, grad_thresh, scale_thresh, prune_big_scale,
            allow_scale_prune, carry_eta)

    def set_cameras(self, cameras: list) -> None:
        """Mip-Splatting setup: upload the (static) training-camera array and
        derive the sqrt(0.2)/max_focal coefficient ONCE. Must be called before
        :meth:`compute_3d_filter`.

        Cameras are fixed for a run (intrinsics + extrinsics never change), so
        this is a one-time upload; compute_3d_filter then only re-dispatches the
        per-gaussian kernel each densify tick (the means move, the cameras do
        not). Pass the TRAIN split to match mip-splatting / SADGS (held-out test
        views must not influence the filter).

        `cameras` are the same Camera objects the trainer renders; each packs to
        a 144-byte CameraData (view 16f + proj 16f + image_size 2f + 2f pad).
        """
        n = len(cameras)
        # CameraData packs to 36 float32s: view 4x4 (16) + proj 4x4 (16) +
        # image_size (2) + 2 pad-to-16B-alignment. Buffer size derives from the
        # array itself (data.nbytes) — no hardcoded byte count.
        data = np.zeros((n, 36), dtype=np.float32)
        max_focal = 0.0
        for i, cam in enumerate(cameras):
            data[i, 0:16] = cam.view_matrix.flatten()
            data[i, 16:32] = cam.proj_matrix.flatten()
            data[i, 32:34] = np.array(cam.image_size, dtype=np.float32)
            # focal_x in pixels: proj[0][0] = 2*fx/W  ->  fx = proj[0][0]*W/2.
            fx = abs(float(cam.proj_matrix[0, 0])) * float(cam.image_size[0]) * 0.5
            max_focal = max(max_focal, fx)
        buf = self.device.create_buffer(
            size=data.nbytes,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label="mip_cameras",
        )
        buf.copy_from_numpy(np.ascontiguousarray(data.reshape(-1).view(np.uint8)))
        self._mip_cameras_buffer = buf
        self._mip_num_cameras = n
        self._mip_filter_coef = float(np.sqrt(0.2) / max(max_focal, 1e-8))

    def compute_3d_filter(self) -> None:
        """Mip-Splatting: recompute the per-gaussian 3D smoothing-filter size
        (buffers.filter_3D) from the cameras uploaded by :meth:`set_cameras`.
        Run under no_grad — filter_3D is a geometric constant, never a gradient
        target. Call once at init and after every densify+morton so new children
        get a valid filter.

        Only the per-gaussian dispatch runs here (means change every step); the
        static camera array + filter coefficient were built once by set_cameras.
        Heavy (O(n_gaussians * n_cameras)) but only fires on the ~17 densify
        ticks of a run, off the hot path.
        """
        if getattr(self, "_mip_cameras_buffer", None) is None:
            raise RuntimeError(
                "compute_3d_filter() requires set_cameras(cameras) to be called first")
        b = self.buffers
        self._dispatch(self.k_compute_3d_filter, "compute_3d_filter",
            thread_count=[self.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            cameras=self._mip_cameras_buffer,
            filter_3D=b.filter_3D.storage,
            num_cameras=self._mip_num_cameras,
            filter_coef=self._mip_filter_coef,
            max_splats=self.max_splats,
        )
        self.runtime.wait_for_idle()

    def reset_opacity(self, cfg) -> None:
        return _densify.reset_opacity(self, cfg)

    def morton_sort(self, world_min, world_max, carry_eta=False) -> None:
        """Reorder gaussians + adam_state by 3D Morton code (Z-order).
        See :func:`trainer.densify.morton_sort`."""
        return _densify.morton_sort(self, world_min, world_max, carry_eta)

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
        """Compatibility delegate; new components receive ``scan_service``."""
        if in_buf is None:
            densification = self.buffers._require_densification()
            in_buf = densification.valid_mask
        self.scan_service.scan(n, in_buf, out_buf, total_buf)

