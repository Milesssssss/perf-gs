"""Dispatch orchestration for the v1 trainer.

A ``Pipeline`` owns a device, the compute kernels, and the GPU resources.
For inference, call :meth:`Pipeline.render`. For training, set a target
image with :meth:`Pipeline.set_target_image` and call
:meth:`Pipeline.train_step`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import slangpy as spy

from trainer import buffers
from trainer.buffers import GRAD_SCALAR_COMPONENTS
from trainer.dataset import Camera
from trainer.device import SHADERS_DIR


SCAN_BLOCK_SIZE = 1024  # must match shaders/common/scan.slang

# DeviceRadixSort constants — must match shaders/sort/device_radix_sort.slang.
_RADIX_PART_SIZE = 2048
_RADIX_US_DIM = 128
_RADIX_SCAN_DIM = 128
_RADIX_D_DIM = 256
_RADIX_GLOBAL_HIST_SIZE = 256 * 8


@dataclass
class DensifyConfig:
    """Threshold-free score-ranked densification with budget-scheduled growth.

    Growth follows a sqrt schedule from initial count to max_splats over
    the densification window. The budget at each step is deterministic.

    Growth selection is purely rank-based: the top-B gaussians by
    composite score get densified each step. No signal thresholds — the
    schedule determines how many, the ranking determines which ones.

    Score = importance * avg_grad * opacity * surface_area_proxy.
    Split/clone classification: median surface area within the selected set.
    """
    n_scoring_views: int = 10       # cameras sampled for importance scoring
    loss_thresh: float = 0.1        # L1 threshold for binary error mask
    t_opacity: float = 0.005        # prune if sigmoid(opacity) < this. MUST stay BELOW
                                    # t_opacity_reset_value (0.01): reset clamps opacity down
                                    # to the reset value, so a floor ABOVE it prunes every
                                    # splat the instant reset fires (total wipeout). 3DGS uses
                                    # prune 0.005 < reset 0.01 so reset splats survive and only
                                    # genuinely-dead ones decaying below 0.005 get culled.
    t_big_scale: float = 0.1        # prune if max_scale > this (world-space floater removal)
    opacity_cap: float = 0.8        # post-densify opacity cap (softer than reset)
    t_opacity_reset_value: float = 0.01  # reset opacity to this value


@dataclass
class AdamLRs:
    """Adam learning rates, matching gsplat defaults:
        means_lr     = 1.6e-4
        scales_lr    = 5e-3
        quats_lr     = 1e-3
        opacities_lr = 2.5e-2
        sh0_lr       = 2.5e-3
    Only ``mean`` has LR decay applied (PLAN.md Q15).
    """
    mean: float = 1.6e-4
    scale: float = 5e-3
    rot: float = 1e-3
    opacity: float = 2.5e-2
    color: float = 2.5e-3 # in reality this is the SH0 LR

def _make_kernel(device: spy.Device, path: Path, entry: str) -> spy.ComputeKernel:
    program = device.load_program(str(path), [entry])
    return device.create_compute_kernel(program)

# Maps kernel id -> name for debug markers.
_kernel_labels: dict[int, str] = {}


def _label_kernel(kernel: spy.ComputeKernel, name: str) -> spy.ComputeKernel:
    _kernel_labels[id(kernel)] = name
    return kernel


def _module_for_reflection(device: spy.Device) -> spy.Module:
    """Load a shader file that includes all the shared struct types so
    buffers.py can reflect ``Gaussian3D`` / ``Projected2DGaussian`` /
    ``AdamState``."""
    return spy.Module.load_from_file(
        device, str(SHADERS_DIR / "optimizers" / "adam.slang")
    )


import contextlib


class Pipeline:
    def __init__(
        self,
        device: spy.Device,
        width: int,
        height: int,
        max_splats: int = 500_000,
        max_sort_pairs: int | None = None,
        gpu_lock: "contextlib.AbstractContextManager | None" = None,
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
            max_sort_pairs = max(buffers.DEFAULT_MAX_SORT_PAIRS, max_splats * 16)
        self.max_sort_pairs = max_sort_pairs

        self.module = _module_for_reflection(device)

        # --- kernels ---
        tb = SHADERS_DIR / "rasterize" / "tile_binned"
        opt = SHADERS_DIR / "optimizers"
        loss = SHADERS_DIR / "losses"

        def mk(path, entry):
            k = _make_kernel(device, path, entry)
            _kernel_labels[id(k)] = entry
            return k

        self.k_clear_textures = mk(tb / "clear.slang", "clearTextures")
        self.k_bin_count = mk(tb / "bin_count.slang", "binCount")
        self.k_bin_emit = mk(tb / "bin_emit.slang", "binEmit")
        self.k_clear_tile_ranges = mk(tb / "tile_ranges.slang", "clearTileRanges")
        self.k_identify_tile_ranges = mk(tb / "tile_ranges.slang", "identifyTileRanges")
        self.k_forward = mk(tb / "forward.slang", "forwardRasterize")
        self.k_loss = mk(loss / "l2_linear.slang", "computeLossGrad")
        self.k_ssim_maps = mk(loss / "ssim_loss.slang", "ssimComputeMaps")
        self.k_ssim_grad = mk(loss / "ssim_loss.slang", "ssimGradient")
        self.k_regularization = mk(loss / "regularization.slang", "addRegularization")
        self.k_backward = mk(tb / "backward.slang", "backwardRasterize")

        sort = SHADERS_DIR / "sort"
        self.k_radix_init_global_hist = mk(sort / "device_radix_sort.slang", "initGlobalHist")
        self.k_radix_prepare_dispatch = mk(sort / "device_radix_sort.slang", "prepareDispatchArgs")
        self.k_radix_upsweep = mk(sort / "device_radix_sort.slang", "upsweep")
        self.k_radix_scan = mk(sort / "device_radix_sort.slang", "scan")
        self.k_radix_downsweep = mk(sort / "device_radix_sort.slang", "downsweep")
        self.k_adam = mk(opt / "adam.slang", "adamStep")
        self.k_grad_clear = mk(opt / "grad_utils.slang", "clearGradBuffer")
        self.k_saturation_check = mk(opt / "grad_utils.slang", "checkSaturation")
        self.k_saturation_clear = mk(opt / "grad_utils.slang", "clearSaturationFlag")

        densify = SHADERS_DIR / "densify"
        self.k_clear_tracking = mk(densify / "clear_tracking.slang", "clearTracking")
        self.k_accumulate_iter = mk(densify / "accumulate_iter.slang", "accumulateIter")
        self.k_decide_actions = mk(densify / "decide_actions.slang", "decideActions")
        self.k_decide_actions_simple = mk(densify / "decide_actions_simple.slang", "decideActionsSimple")
        self.k_split_and_clone = mk(densify / "split_and_clone.slang", "splitAndClone")
        self.k_build_prune_mask = mk(densify / "build_prune_mask.slang", "buildPruneMask")
        self.k_scatter_compact = mk(densify / "scatter_compact.slang", "scatterCompact")
        self.k_opacity_reset = mk(densify / "opacity_reset.slang", "resetOpacity")
        self.k_clear_importance = mk(densify / "clear_importance.slang", "clearImportance")
        self.k_compute_l1_minmax = mk(densify / "compute_error_mask.slang", "computeL1MinMax")
        self.k_compute_error_mask = mk(densify / "compute_error_mask.slang", "computeErrorMask")
        self.k_score_forward = mk(densify / "score_forward.slang", "scoreForward")
        self.k_densify_snapshot = mk(densify / "densify_dispatch.slang", "snapshotCount")
        self.k_densify_prepare_args = mk(densify / "densify_dispatch.slang", "prepareDensifyArgs")

        scan = SHADERS_DIR / "common" / "scan.slang"
        self.k_scan_block = mk(scan, "scanBlock")
        self.k_scan_block_sums = mk(scan, "scanBlockSums")
        self.k_add_block_offsets = mk(scan, "addBlockOffsets")

        # --- buffers ---
        self.buffers = buffers.allocate(
            device=device, module=self.module,
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

        # Near-plane cull distance used by the bin kernel.
        self.z_near_cull = 0.01

        # Regularization weights (gsplat defaults: both 0; MCMC uses 0.001).
        self.opacity_reg = 0.0
        self.scale_reg = 0.0

        # Alpha loss weight. Set > 0 when target images have meaningful alpha.
        self.w_alpha = 0.0

        # SSIM loss mixing. 0 = pure L2, >0 = L1+SSIM blend (3DGS default 0.2).
        self.lambda_ssim = 0.2

    def on_hot_reload(self) -> None:
        """Refresh reflection-dependent objects after device.reload_all_programs().
        Camera buffer is raw bytes so it survives reload. Only the module
        reference needs refreshing (for any code that uses reflected types)."""
        self.module = _module_for_reflection(self.device)

    def _dispatch(self, kernel, **kwargs):
        """Dispatch a kernel, inserting a debug marker if encoder is active."""
        if self._encoder:
            label = _kernel_labels.get(id(kernel), "unknown")
            self._encoder.insert_debug_marker(label, spy.math.float3(0.6, 0.8, 1.0))
        kernel.dispatch(command_encoder=self._encoder, **kwargs)

    def _dispatch_indirect(self, kernel, args_buf: spy.NDBuffer, offset: int = 0,
                           **kwargs) -> None:
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
        label = _kernel_labels.get(id(kernel), "unknown")
        enc.insert_debug_marker(label, spy.math.float3(0.6, 0.8, 1.0))
        # HAZARD: the args_buf was written by a preceding kernel dispatch
        # (e.g. prepareDispatchArgs). dispatch_compute_indirect reads it as
        # driver-consumed indirect arguments; without a barrier the read
        # may race the write and pick up garbage → a catastrophic group
        # count. Insert a global barrier to force write-visibility before
        # the indirect fetch. Cheap compared to the cost of the failure
        # mode (a 32-bit garbage groups_x can queue billions of workgroups).
        enc.global_barrier()
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
    def _marker_group(self, label: str, color=(0.5, 0.7, 0.9)):
        """Nested debug-group scope for profiling captures (e.g. Nsight
        Graphics). No-op when no encoder is active, so it costs nothing outside
        an encoder-backed train_step. Groups nest, so wrapping a sub-stage of a
        phase (bin / scan / sort / rasterize) gives a drill-down tree under the
        top-level phase group instead of one flat run of per-kernel markers."""
        enc = self._encoder
        if enc is None:
            yield
            return
        enc.push_debug_group(label, spy.math.float3(*color))
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
        max_blocks = (self.max_sort_pairs + _RADIX_PART_SIZE - 1) // _RADIX_PART_SIZE

        # Prepare indirect-dispatch args from the GPU-resident count
        # scan_total[0]. Upsweep and downsweep are then launched with
        # active_blocks groups instead of max_blocks — dispatch cost scales
        # with the frame's actual pair count, not the max_sort_pairs cap.
        self._dispatch(self.k_radix_prepare_dispatch,
            thread_count=[1, 1, 1],
            n_buf=b.scan_total.storage,
            dispatch_args=b.radix_dispatch_args.storage,
            max_blocks=max_blocks,
        )

        # Clear b_global_hist once at the start of the whole sort;
        # subsequent atomic adds during upsweep accumulate into disjoint
        # per-pass regions (offset = pass_idx * 256).
        self._dispatch(self.k_radix_init_global_hist,
            thread_count=[_RADIX_GLOBAL_HIST_SIZE, 1, 1],
            b_global_hist=b.radix_global_hist.storage,
        )

        for pass_idx in range(8):
            radix_shift = (pass_idx % 4) * 8
            key_word = 0 if pass_idx < 4 else 1
            with self._marker_group(f"pass_{pass_idx}", (0.9, 0.55, 0.2)):
                self._dispatch_indirect(self.k_radix_upsweep,
                    b.radix_dispatch_args,
                    keys_lo=src_lo.storage, keys_hi=src_hi.storage,
                    b_pass_hist=b.radix_pass_hist.storage,
                    b_global_hist=b.radix_global_hist.storage,
                    n_buf=b.scan_total.storage,
                    radix_shift=radix_shift, key_word=key_word,
                    pass_idx=pass_idx, max_blocks=max_blocks,
                )
                self._dispatch(self.k_radix_scan,
                    thread_count=[256 * _RADIX_SCAN_DIM, 1, 1],
                    b_pass_hist=b.radix_pass_hist.storage,
                    n_buf=b.scan_total.storage,
                    max_blocks=max_blocks,
                )
                self._dispatch_indirect(self.k_radix_downsweep,
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
        with self._marker_group("clear", (0.35, 0.35, 0.4)):
            self._dispatch(self.k_clear_textures,
                thread_count=[self.width, self.height, 1],
                output_image=b.output_image,
                output_transmittance=b.output_transmittance,
                gaussian_count_per_pixel=b.gaussian_count_per_pixel,
                frame_dim=spy.uint2(self.width, self.height),
            )
            self._dispatch(self.k_clear_tile_ranges,
                thread_count=[num_tiles, 1, 1],
                tile_start=b.tile_start.storage,
                tile_end=b.tile_end.storage,
                num_tiles=num_tiles,
            )

        # Phase 1: per-gaussian tile count.
        with self._marker_group("bin_count", (0.3, 0.7, 0.5)):
            self._dispatch(self.k_bin_count,
                thread_count=[self.max_splats, 1, 1],
                gaussians=b.gaussians.storage,
                num_gaussians_counter=b.num_gaussians_counter.storage,
                camera=self._camera_buffer,
                projected_2d=b.projected_2d.storage,
                tile_count_per_gaussian=b.tile_count_per_gaussian.storage,
                z_near=self.z_near_cull,
                max_splats=self.max_splats,
            )

        # Phase 2: exclusive scan → tile_offset_per_gaussian + total count.
        # scan_total[0] holds the total pair count on the GPU; no readback.
        with self._marker_group("scan_tile_offsets", (0.3, 0.6, 0.7)):
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
        with self._marker_group("bin_emit", (0.3, 0.7, 0.5)):
            self._dispatch(self.k_bin_emit,
                thread_count=[self.max_splats, 1, 1],
                gaussians=b.gaussians.storage,
                num_gaussians_counter=b.num_gaussians_counter.storage,
                camera=self._camera_buffer,
                projected_2d=b.projected_2d.storage,
                tile_offset_per_gaussian=b.tile_offset_per_gaussian.storage,
                keys_lo=b.sort_keys_lo_a.storage,
                keys_hi=b.sort_keys_hi_a.storage,
                values=b.sort_values_a.storage,
                z_near=self.z_near_cull,
            )

        # Phase 4: global radix sort. Reads n from scan_total[0].
        with self._marker_group("radix_sort", (0.9, 0.5, 0.2)):
            sorted_lo, sorted_hi, sorted_v = self._radix_sort_pairs()

        # Phase 5: identify per-tile start/end in the sorted array.
        with self._marker_group("identify_tile_ranges", (0.3, 0.6, 0.7)):
            self._dispatch(self.k_identify_tile_ranges,
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

        # Phase 6: forward rasterize.
        with self._marker_group("rasterize", (0.2, 0.8, 0.3)):
            self._dispatch(self.k_forward,
                thread_count=[b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
                gaussians=b.gaussians.storage,
                projected_2d=b.projected_2d.storage,
                tile_start=b.tile_start.storage,
                tile_end=b.tile_end.storage,
                sorted_gaussian_ids=sorted_v.storage,
                camera=self._camera_buffer,
                output_image=b.output_image,
                output_transmittance=b.output_transmittance,
                gaussian_count_per_pixel=b.gaussian_count_per_pixel,
            )

    def render(self, camera: Camera) -> None:
        self._run_forward(camera)
        self.device.wait_for_idle()

    def read_output_image(self) -> np.ndarray:
        return self.buffers.output_image.to_numpy()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _clear_grads(self) -> None:
        b = self.buffers
        for name in GRAD_SCALAR_COMPONENTS:
            self._dispatch(self.k_grad_clear,
                thread_count=[self.max_splats, 1, 1],
                grad_buffer=b.grad_buffers[name].storage,
                max_splats=self.max_splats,
            )

    def _run_loss(self) -> None:
        b = self.buffers
        target = getattr(self, "_active_target", None) or b.target_image
        if self.lambda_ssim > 0:
            # Two-pass SSIM: pass 1 builds the m1/m2/m3 back-projection maps,
            # pass 2 convolves them and blends with the L1 + alpha gradient.
            with self._marker_group("ssim_maps", (0.8, 0.8, 0.3)):
                self._dispatch(self.k_ssim_maps,
                    thread_count=[self.width, self.height, 1],
                    output_image=b.output_image,
                    target_image=target,
                    ssim_m1=b.ssim_m1,
                    ssim_m2=b.ssim_m2,
                    ssim_m3=b.ssim_m3,
                    frame_dim=spy.uint2(self.width, self.height),
                )
            with self._marker_group("ssim_grad", (0.9, 0.75, 0.2)):
                self._dispatch(self.k_ssim_grad,
                    thread_count=[self.width, self.height, 1],
                    output_image=b.output_image,
                    target_image=target,
                    ssim_m1=b.ssim_m1,
                    ssim_m2=b.ssim_m2,
                    ssim_m3=b.ssim_m3,
                    dL_dC=b.dL_dC_texture,
                    frame_dim=spy.uint2(self.width, self.height),
                    lambda_ssim=self.lambda_ssim,
                    w_alpha=self.w_alpha,
                )
        else:
            with self._marker_group("l2_loss", (0.9, 0.8, 0.2)):
                self._dispatch(self.k_loss,
                    thread_count=[self.width, self.height, 1],
                    output_image=b.output_image,
                    target_image=target,
                    dL_dC=b.dL_dC_texture,
                    frame_dim=spy.uint2(self.width, self.height),
                    w_alpha=self.w_alpha,
                )

    def _run_backward(self) -> None:
        b = self.buffers
        gb = b.grad_buffers
        # The forward left the sorted gaussian-id buffer on one of the two
        # ping-pong slots. We cache it during _run_forward.
        sorted_values = getattr(self, "_sorted_values", b.sort_values_a)
        with self._marker_group("rasterize_backward", (0.9, 0.3, 0.3)):
            self._dispatch(self.k_backward,
                thread_count=[b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
                gaussians=b.gaussians.storage,
                projected_2d=b.projected_2d.storage,
                tile_start=b.tile_start.storage,
                tile_end=b.tile_end.storage,
                sorted_gaussian_ids=sorted_values.storage,
                camera=self._camera_buffer,
                output_image=b.output_image,
                output_transmittance=b.output_transmittance,
                gaussian_count_per_pixel=b.gaussian_count_per_pixel,
                dL_dC_texture=b.dL_dC_texture,
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
                grad_opacity=gb["opacity"].storage,
                grad_color_r=gb["color_r"].storage,
                grad_color_g=gb["color_g"].storage,
                grad_color_b=gb["color_b"].storage,
                grad_2d_x_iter=b.grad_2d_x_iter.storage,
                grad_2d_y_iter=b.grad_2d_y_iter.storage,
            )
        # Fold the per-iter 2D grad vector into the window-level
        # mag + visibility accumulators. Gsplat semantics: the signal
        # driving densification is the NORM OF THE SUM of per-pixel
        # gradients per iter, averaged over iters the gaussian was
        # visible — not the sum of per-pixel norms.
        with self._marker_group("accumulate_iter", (0.7, 0.4, 0.5)):
            self._dispatch(self.k_accumulate_iter,
                thread_count=[self.max_splats, 1, 1],
                grad_2d_x_iter=b.grad_2d_x_iter.storage,
                grad_2d_y_iter=b.grad_2d_y_iter.storage,
                grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
                visibility_count_accum=b.visibility_count_accum.storage,
                frame_dim=spy.float2(self.width, self.height),
                max_splats=self.max_splats,
            )

    def _run_regularization(self) -> None:
        if self.opacity_reg <= 0 and self.scale_reg <= 0:
            return
        b = self.buffers
        gb = b.grad_buffers
        n = int(b.num_gaussians_counter.to_numpy().view(np.uint32)[0]) if False else self.max_splats
        self._dispatch(self.k_regularization,
            thread_count=[self.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            grad_opacity=gb["opacity"].storage,
            grad_log_scale_x=gb["log_scale_x"].storage,
            grad_log_scale_y=gb["log_scale_y"].storage,
            grad_log_scale_z=gb["log_scale_z"].storage,
            w_opacity=self.opacity_reg,
            w_scale=self.scale_reg,
            inv_n=1.0,
        )

    def clear_tracking(self) -> None:
        """Zero the v2 densification tracking accumulators. Called at the
        start of each densification window (every N iters)."""
        b = self.buffers
        self._dispatch(self.k_clear_tracking,
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

    def _run_saturation_check(self) -> None:
        b = self.buffers
        self._dispatch(self.k_saturation_clear,
            thread_count=[1, 1, 1],
            saturation_flag=b.saturation_flag.storage,
        )
        for name in GRAD_SCALAR_COMPONENTS:
            self._dispatch(self.k_saturation_check,
                thread_count=[self.max_splats, 1, 1],
                grad_buffer=b.grad_buffers[name].storage,
                saturation_flag=b.saturation_flag.storage,
                max_splats=self.max_splats,
            )

    def _run_adam(self, lrs: AdamLRs) -> None:
        self.iteration += 1
        step = float(self.iteration)
        bc1 = 1.0 - 0.9 ** step
        bc2 = 1.0 - 0.999 ** step

        b = self.buffers
        gb = b.grad_buffers
        self._dispatch(self.k_adam,
            thread_count=[self.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            adam_state=b.adam_state.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
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
            grad_opacity=gb["opacity"].storage,
            grad_color_r=gb["color_r"].storage,
            grad_color_g=gb["color_g"].storage,
            grad_color_b=gb["color_b"].storage,
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

    def forward_and_loss(self, camera: Camera) -> None:
        """Run forward + loss only. Used by tests that read ``dL_dC``
        before the backward pass runs.

        No ``wait_for_idle`` here — callers that read back GPU data
        should call ``device.wait_for_idle()`` themselves, or use a
        helper like :meth:`render` / :meth:`read_output_image` which
        handle the sync. Tests that explicitly sample intermediates
        insert the wait at their own callsite.
        """
        self._run_forward(camera)
        self._run_loss()

    def backward(self) -> None:
        """Run the backward rasterizer. Assumes forward + loss have run."""
        self._run_backward()

    def check_saturation(self) -> bool:
        """Return True if any int32 gradient buffer saturated. The
        readback forces a GPU flush; callers should only use this
        diagnostically, not every iter."""
        self._run_saturation_check()
        self.device.wait_for_idle()
        flag = self.buffers.saturation_flag.to_numpy().view(np.uint32)[0]
        return flag != 0

    def train_step(self, camera: Camera, lrs: AdamLRs | None = None) -> None:
        """Run one training step: forward → loss → backward → regularization → adam.

        The target image must have been uploaded via
        :meth:`set_target_image` before calling.

        NO CPU/GPU sync here — training submits kernels and returns
        immediately. This keeps the GPU saturated; the queue drains
        when the training loop next reads back data (loss sample,
        preview render, densify).
        """
        if lrs is None:
            lrs = AdamLRs()
        enc = self.device.create_command_encoder()
        self._encoder = enc
        enc.push_debug_group("clear_grads", spy.math.float3(0.3, 0.3, 0.3))
        self._clear_grads()
        enc.pop_debug_group()
        enc.push_debug_group("forward", spy.math.float3(0.2, 0.8, 0.3))
        self._run_forward(camera)
        enc.pop_debug_group()
        enc.push_debug_group("loss", spy.math.float3(1.0, 0.8, 0.0))
        self._run_loss()
        enc.pop_debug_group()
        enc.push_debug_group("backward", spy.math.float3(0.9, 0.2, 0.2))
        self._run_backward()
        enc.pop_debug_group()
        enc.push_debug_group("regularization", spy.math.float3(0.6, 0.4, 0.9))
        self._run_regularization()
        enc.pop_debug_group()
        enc.push_debug_group("adam", spy.math.float3(0.2, 0.6, 1.0))
        self._run_adam(lrs)
        enc.pop_debug_group()
        cmd = enc.finish()
        with self.gpu_lock:
            self.last_submit_id = self.device.submit_command_buffer(cmd)
        self._encoder = None

    # ------------------------------------------------------------------
    # Gradient readback (for the finite-difference test)
    # ------------------------------------------------------------------

    def read_gradients(self) -> dict[str, np.ndarray]:
        """Read back the 14 SoA int64 fixed-point grad buffers."""
        from trainer.buffers import GRAD_SCALAR_COMPONENTS as NAMES

        n = int(self.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
        inv = 1.0 / 65536.0
        out: dict[str, np.ndarray] = {}
        for name in NAMES:
            raw = self.buffers.grad_buffers[name].to_numpy()
            ints = np.frombuffer(raw.tobytes(), dtype=np.int64)[:n]
            out[name] = ints.astype(np.float64) * inv
        return out

    def read_loss(self) -> float:
        """Read the L1 loss as a scalar. Computed on CPU from the
        output_image and target_image — the shader doesn't accumulate
        the scalar itself (we only needed dL/dC for the backward).

        Matches the L1 photometric objective in l2_linear.slang so the
        finite-difference gradient tests stay consistent with the shader."""
        C = self.buffers.output_image.to_numpy()[..., :3]
        T_rgba = self.buffers.target_image.to_numpy()[..., :3]
        T_linear = _srgb_to_linear_np(T_rgba)
        diff = C - T_linear
        return float(np.sum(np.abs(diff)))


def _srgb_to_linear_np(c: np.ndarray) -> np.ndarray:
    low = c / 12.92
    high = np.power(np.clip((c + 0.055) / 1.055, 1e-7, None), 2.4)
    return np.where(c <= 0.04045, low, high).astype(np.float32)

def _linear_to_srgb_np(c: np.ndarray) -> np.ndarray:
    """Inverse of _srgb_to_linear_np. Exact round-trip so that the loss
    shader's srgb_to_linear(target) recovers the linear value we intend."""
    c = np.clip(c, 0.0, None)
    low = c * 12.92
    high = 1.055 * np.power(np.clip(c, 1e-7, None), 1.0 / 2.4) - 0.055
    return np.where(c <= 0.0031308, low, high).astype(np.float32)


# -------- densification methods on Pipeline (added via monkey-patch-style
# direct method definitions; kept here for proximity to the scan + buffer
# structures they orchestrate) --------

def _scan(self: Pipeline, n: int,
          in_buf: spy.NDBuffer | None = None,
          out_buf: spy.NDBuffer | None = None) -> None:
    """Run a multi-level Blelloch scan on ``in_buf`` into ``out_buf``,
    writing the total into ``scan_total[0]``.

    Supports up to SCAN_BLOCK_SIZE^3 (~1 billion) elements via 3 levels.
    """
    b = self.buffers
    if in_buf is None:
        in_buf = b.valid_mask_buffer
    if out_buf is None:
        out_buf = b.scan_buffer

    num_blocks = max(1, (n + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE)

    # Level 1: per-block scan, producing block_sums[num_blocks].
    self._dispatch(self.k_scan_block,
        thread_count=[num_blocks * SCAN_BLOCK_SIZE, 1, 1],
        in_buf=in_buf.storage,
        out_buf=out_buf.storage,
        block_sums=b.scan_block_sums.storage,
        n=n,
    )

    if num_blocks <= SCAN_BLOCK_SIZE:
        # 2-level: single-workgroup scan of block sums.
        self._dispatch(self.k_scan_block_sums,
            thread_count=[SCAN_BLOCK_SIZE, 1, 1],
            block_sums=b.scan_block_sums.storage,
            total_out=b.scan_total.storage,
            num_blocks=num_blocks,
        )
    else:
        # 3-level: scan block_sums using the same 2-level pattern.
        num_l2_blocks = max(1, (num_blocks + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE)
        # Scan block_sums → block_sums (in-place prefix), with l2 block sums.
        self._dispatch(self.k_scan_block,
            thread_count=[num_l2_blocks * SCAN_BLOCK_SIZE, 1, 1],
            in_buf=b.scan_block_sums.storage,
            out_buf=b.scan_block_sums.storage,
            block_sums=b.scan_block_sums_l2.storage,
            n=num_blocks,
        )
        self._dispatch(self.k_scan_block_sums,
            thread_count=[SCAN_BLOCK_SIZE, 1, 1],
            block_sums=b.scan_block_sums_l2.storage,
            total_out=b.scan_total.storage,
            num_blocks=num_l2_blocks,
        )
        self._dispatch(self.k_add_block_offsets,
            thread_count=[num_l2_blocks * SCAN_BLOCK_SIZE, 1, 1],
            out_buf=b.scan_block_sums.storage,
            block_sums=b.scan_block_sums_l2.storage,
            n=num_blocks,
        )

    # Final: add block offsets to per-element results.
    self._dispatch(self.k_add_block_offsets,
        thread_count=[num_blocks * SCAN_BLOCK_SIZE, 1, 1],
        out_buf=out_buf.storage,
        block_sums=b.scan_block_sums.storage,
        n=n,
    )


Pipeline._scan = _scan  # type: ignore[attr-defined]


def compute_importance_scores(
    self: Pipeline,
    cameras: list,
    target_textures: list,
    cfg: DensifyConfig,
) -> None:
    """Run N scoring views to accumulate per-gaussian importance scores.

    For each view: render → compute binary error mask → walk tiles and
    count how many high-error pixels each gaussian touches. Results
    accumulate into buffers.importance_score_accum (uint per gaussian).
    """
    b = self.buffers

    self._dispatch(self.k_clear_importance,
        thread_count=[self.max_splats, 1, 1],
        importance_score_accum=b.importance_score_accum.storage,
        max_splats=self.max_splats,
    )

    for cam, target_tex in zip(cameras, target_textures):
        self._run_forward(cam)

        # Initialize minmax: min=+inf (0x7f7fffff), max=0 (0x00000000).
        b.l1_minmax.copy_from_numpy(np.array([0x7f7fffff, 0x00000000], dtype=np.uint32))

        # Pass 1: compute L1 per pixel, find min/max.
        self._dispatch(self.k_compute_l1_minmax,
            thread_count=[self.width, self.height, 1],
            output_image=b.output_image,
            target_image=target_tex,
            l1_map=b.l1_map,
            minmax_buf=b.l1_minmax.storage,
            frame_dim=spy.uint2(self.width, self.height),
        )

        # Pass 2: normalize and threshold.
        self._dispatch(self.k_compute_error_mask,
            thread_count=[self.width, self.height, 1],
            l1_map=b.l1_map,
            minmax_buf=b.l1_minmax.storage,
            error_mask=b.error_mask,
            frame_dim=spy.uint2(self.width, self.height),
            loss_thresh=cfg.loss_thresh,
        )

        sorted_values = getattr(self, "_sorted_values", b.sort_values_a)
        self._dispatch(self.k_score_forward,
            thread_count=[b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
            gaussians=b.gaussians.storage,
            projected_2d=b.projected_2d.storage,
            tile_start=b.tile_start.storage,
            tile_end=b.tile_end.storage,
            sorted_gaussian_ids=sorted_values.storage,
            camera=self._camera_buffer,
            error_mask=b.error_mask,
            importance_score_accum=b.importance_score_accum.storage,
        )


Pipeline.compute_importance_scores = compute_importance_scores  # type: ignore[attr-defined]


def densify(self: Pipeline, cfg: DensifyConfig,
            densify_step: int, n_densify_steps: int,
            init_count: int) -> dict:
    """Run one densification pass: score → rank → select top-B → split/clone.

    Growth follows a sqrt schedule: target(x) = S + (B - S) * sqrt(x/N).
    Front-loads growth (large steps early when sparse, tapering toward
    the target). The per-step budget is max(0, target - current_count),
    catching up if pruning dropped the count below the curve.

    Phases:
      A  decide_actions     - compute scores, apply pruning rules
      B  (CPU) rank+select  - top-B by score, classify split/clone by scale
      C  split_and_clone    - append children, overwrite parent slot for splits
      D  build_prune_mask   - mark valid[i] across post-split range
      E  scan + compact     - prefix-sum + scatter

    Returns a stats dict for logging.
    """
    ACTION_KEEP = 0
    ACTION_SPLIT = 1
    ACTION_CLONE = 2
    ACTION_PRUNE = 3

    b = self.buffers

    self.device.wait_for_idle()
    pre_count = int(b.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    if pre_count == 0:
        return {"pre_count": 0, "post_split_count": 0, "final_count": 0,
                "n_split": 0, "n_clone": 0, "n_prune": 0, "n_keep": 0}

    # Growth schedule: sqrt curve front-loads growth (big steps early when
    # sparse, tapering as we approach target). Catch-up: if pruning dropped
    # count below the curve, we grow more this step.
    B = self.max_splats
    S = init_count
    N = max(n_densify_steps, 1)
    x = min(densify_step, N)
    target_at_x = S + (B - S) * (x / N) ** 0.5
    growth = max(0, int(target_at_x) - pre_count)

    # Prune budget: at most 50% of current count.
    prune = max(1, pre_count // 2)
    b.prune_budget.copy_from_numpy(np.array([prune], dtype=np.int32))

    # Phase A: GPU computes scores and applies pruning.
    self._dispatch(self.k_decide_actions,
        thread_count=[pre_count, 1, 1],
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        importance_score_accum=b.importance_score_accum.storage,
        grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
        visibility_count_accum=b.visibility_count_accum.storage,
        action=b.action_buffer.storage,
        densify_score=b.densify_score.storage,
        prune_budget=b.prune_budget.storage,
        t_opacity=cfg.t_opacity,
        t_big_scale=cfg.t_big_scale,
        n_scoring_views=cfg.n_scoring_views,
    )
    self.device.wait_for_idle()

    # Phase B (CPU): read scores + actions, rank, select top-B.
    actions_raw = b.action_buffer.to_numpy()
    actions = np.frombuffer(actions_raw.tobytes(), dtype=np.uint32)[:pre_count].copy()

    scores_raw = b.densify_score.to_numpy()
    scores = np.frombuffer(scores_raw.tobytes(), dtype=np.float32)[:pre_count]

    if growth > 0:
        # Read gaussian scales for split/clone classification.
        gauss_raw = b.gaussians.to_numpy()
        gauss_floats = np.frombuffer(gauss_raw.tobytes(), dtype=np.float32).reshape(-1, 14)
        scales = np.exp(gauss_floats[:pre_count, 3:6])
        # Surface area proxy: sum of pairwise axis products (∝ ellipsoid SA).
        sa = (scales[:, 0] * scales[:, 1]
            + scales[:, 0] * scales[:, 2]
            + scales[:, 1] * scales[:, 2])

        # Only consider non-pruned gaussians with positive score.
        candidates = (actions == ACTION_KEEP) & (scores > 0)
        candidate_indices = np.where(candidates)[0]

        if len(candidate_indices) > 0:
            candidate_scores = scores[candidate_indices]
            # Select top-B by score (no threshold).
            n_select = min(growth, len(candidate_indices))
            top_local = np.argpartition(candidate_scores, -n_select)[-n_select:]
            top_indices = candidate_indices[top_local]

            # Classify by surface area: above-median → split (too big,
            # needs subdivision), at/below → clone (right size, region
            # needs more coverage). Relative to the selected set.
            selected_sa = sa[top_indices]
            sa_median = float(np.median(selected_sa))
            for idx in top_indices:
                if sa[idx] > sa_median:
                    actions[idx] = ACTION_SPLIT
                else:
                    actions[idx] = ACTION_CLONE

    # Write the final action buffer back to GPU.
    action_upload = np.zeros(self.max_splats, dtype=np.uint32)
    action_upload[:pre_count] = actions
    b.action_buffer.copy_from_numpy(action_upload)

    # Phase C: split and clone (may increment counter).
    # The shader now reads pre_count from a GPU buffer (unified path with
    # densify_simple's wait-free dispatch); upload our Python-known value.
    b.densify_pre_count.copy_from_numpy(np.array([pre_count], dtype=np.uint32))
    self._dispatch(self.k_split_and_clone,
        thread_count=[pre_count, 1, 1],
        gaussians=b.gaussians.storage,
        adam_state=b.adam_state.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        action=b.action_buffer.storage,
        pre_count_buf=b.densify_pre_count.storage,
        max_splats=self.max_splats,
    )
    self.device.wait_for_idle()
    post_split_count = int(b.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    post_split_count = min(post_split_count, self.max_splats)

    # Phases D/E: compact. Same upload-then-pass-buffer pattern.
    b.densify_post_count.copy_from_numpy(np.array([post_split_count], dtype=np.uint32))
    self._dispatch(self.k_build_prune_mask,
        thread_count=[self.max_splats, 1, 1],
        action=b.action_buffer.storage,
        valid_mask=b.valid_mask_buffer.storage,
        pre_count_buf=b.densify_pre_count.storage,
        post_count_buf=b.densify_post_count.storage,
        max_splats=self.max_splats,
    )
    self._scan(self.max_splats)
    self._dispatch(self.k_scatter_compact,
        thread_count=[self.max_splats, 1, 1],
        gaussians_in=b.gaussians.storage,
        adam_state_in=b.adam_state.storage,
        valid_mask=b.valid_mask_buffer.storage,
        scan=b.scan_buffer.storage,
        gaussians_out=b.gaussians_scratch.storage,
        adam_state_out=b.adam_state_scratch.storage,
        max_splats=self.max_splats,
    )

    b.gaussians, b.gaussians_scratch = b.gaussians_scratch, b.gaussians
    b.adam_state, b.adam_state_scratch = b.adam_state_scratch, b.adam_state

    self.device.wait_for_idle()
    final_count = int(b.scan_total.to_numpy().view(np.uint32)[0])
    b.num_gaussians_counter.copy_from_numpy(np.array([final_count], dtype=np.uint32))

    n_keep  = int((actions == ACTION_KEEP).sum())
    n_split = int((actions == ACTION_SPLIT).sum())
    n_clone = int((actions == ACTION_CLONE).sum())
    n_prune = int((actions == ACTION_PRUNE).sum())

    return {
        "pre_count": pre_count,
        "post_split_count": post_split_count,
        "final_count": final_count,
        "n_keep": n_keep,
        "n_split": n_split,
        "n_clone": n_clone,
        "n_prune": n_prune,
    }


Pipeline.densify = densify  # type: ignore[attr-defined]


def densify_simple(self: Pipeline, cfg: DensifyConfig,
                   grad_thresh: float = 0.0002,
                   scale_thresh: float | None = None,
                   prune_big_scale: float | None = None,
                   allow_scale_prune: bool = True) -> dict:
    """Reference 3DGS densification: threshold on avg_grad, split/clone by scale.

    No rank-selection, no importance scoring, no growth budget. Every gaussian
    that exceeds grad_thresh gets densified. Simple and proven.

    ``prune_big_scale`` is the world-space max-scale prune threshold
    (gsplat prune_scale3d * scene_scale). ``allow_scale_prune`` gates it so the
    caller can disable big-scale pruning until after the first opacity reset,
    matching gsplat's ``step > reset_every`` guard. Low-opacity pruning is
    always active.
    """

    b = self.buffers

    if scale_thresh is None:
        scale_thresh = cfg.t_big_scale * 0.1

    # Record all densify dispatches into ONE encoder + ONE submit so the
    # whole pass is a single command buffer with no mid-flight submit
    # overhead. Uses the same pattern as train_step.
    enc = self.device.create_command_encoder()
    self._encoder = enc
    enc.push_debug_group("densify_simple", spy.math.float3(0.8, 0.4, 0.9))

    # ``grad_thresh`` is expressed in gsplat's convention: the NDC-space
    # densification gradient of the MEAN L1 loss (F.l1_loss averages over
    # H*W*channels). Our accumulator already carries the NDC conversion
    # (accumulate_iter multiplies by frame_dim/2) but NOT the loss-mean
    # factor — the loss shader emits the raw per-pixel gradient sum, and
    # the 1/(H*W*3) factor is too small to survive the 2^16 fixed-point
    # accumulator. So we scale the threshold UP by H*W*channels here to
    # compare against our sum-convention signal. With this, grad_thresh
    # stays resolution- and convention-compatible with gsplat's 0.0002.
    channels = 3.0
    grad_thresh_internal = grad_thresh * (self.width * self.height * channels)
    big_scale = prune_big_scale if prune_big_scale is not None else cfg.t_big_scale

    # ---- Wait-free densify pipeline (all GPU, no CPU readbacks) ----
    #
    # Counts flow like this:
    #   num_gaussians_counter[0] ─(snapshot)─▶ densify_pre_count[0]
    #                                             │
    #     ┌───────────────────────────────────────┘
    #     ▼
    #   splitAndClone atomically bumps num_gaussians_counter[0]
    #     │
    #     └─(snapshot)─▶ densify_post_count[0]
    #                       │
    #     ┌─────────────────┴──────────────────────┐
    #     ▼                                        ▼
    #   buildPruneMask over [max_splats]         scatterCompact over [max_splats]
    #     (zeros valid_mask tail beyond           (skips i where valid_mask=0)
    #      post_count)
    #     │
    #     ▼
    #   _scan over max_splats → scan_total[0] = final_count
    #     │
    #     └─(snapshot)─▶ num_gaussians_counter[0]  (final_count)
    #
    # Everything except splitAndClone runs at [max_splats] and self-bounds
    # off the snapshot buffers, so we only need one indirect-dispatch path.

    # Phase A: decide actions. Dispatches [max_splats]; shader early-outs
    # per-thread on i >= num_gaussians_counter[0] (GPU-resident).
    self._dispatch(self.k_decide_actions_simple,
        thread_count=[self.max_splats, 1, 1],
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
        visibility_count_accum=b.visibility_count_accum.storage,
        action=b.action_buffer.storage,
        grad_thresh=grad_thresh_internal,
        scale_thresh=scale_thresh,
        t_opacity=cfg.t_opacity,
        t_big_scale=big_scale,
        allow_scale_prune=1 if allow_scale_prune else 0,
        max_splats=self.max_splats,
    )

    # Snapshot pre_count = num_gaussians_counter[0] (before splitAndClone
    # atomically bumps it). Then compute indirect-dispatch args from that
    # snapshot for splitAndClone.
    self._dispatch(self.k_densify_snapshot,
        thread_count=[1, 1, 1],
        src=b.num_gaussians_counter.storage,
        dst=b.densify_pre_count.storage,
    )
    self._dispatch(self.k_densify_prepare_args,
        thread_count=[1, 1, 1],
        count_buf=b.densify_pre_count.storage,
        dispatch_args=b.densify_dispatch_args.storage,
        max_splats=self.max_splats,
    )

    # Phase B: split and clone. Indirect so group count scales with
    # pre_count, not max_splats.
    self._dispatch_indirect(self.k_split_and_clone,
        b.densify_dispatch_args,
        gaussians=b.gaussians.storage,
        adam_state=b.adam_state.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        action=b.action_buffer.storage,
        pre_count_buf=b.densify_pre_count.storage,
        max_splats=self.max_splats,
    )

    # Snapshot post_count = num_gaussians_counter[0] (after splitAndClone).
    self._dispatch(self.k_densify_snapshot,
        thread_count=[1, 1, 1],
        src=b.num_gaussians_counter.storage,
        dst=b.densify_post_count.storage,
    )

    # Phase C: mark valid. Dispatched over [max_splats]; shader zeros
    # valid_mask for i >= post_count so the subsequent max_splats-wide
    # scan sees a padded 0-tail.
    self._dispatch(self.k_build_prune_mask,
        thread_count=[self.max_splats, 1, 1],
        action=b.action_buffer.storage,
        valid_mask=b.valid_mask_buffer.storage,
        pre_count_buf=b.densify_pre_count.storage,
        post_count_buf=b.densify_post_count.storage,
        max_splats=self.max_splats,
    )

    # Phase D: prefix-scan valid_mask into scan_buffer. Runs at max_splats
    # (the tail 0s contribute nothing to scan_total[0]).
    self._scan(self.max_splats)

    # Phase E: scatter live gaussians into scratch. Dispatched over
    # [max_splats]; skips slots where valid_mask=0 (includes the tail).
    self._dispatch(self.k_scatter_compact,
        thread_count=[self.max_splats, 1, 1],
        gaussians_in=b.gaussians.storage,
        adam_state_in=b.adam_state.storage,
        valid_mask=b.valid_mask_buffer.storage,
        scan=b.scan_buffer.storage,
        gaussians_out=b.gaussians_scratch.storage,
        adam_state_out=b.adam_state_scratch.storage,
        max_splats=self.max_splats,
    )

    b.gaussians, b.gaussians_scratch = b.gaussians_scratch, b.gaussians
    b.adam_state, b.adam_state_scratch = b.adam_state_scratch, b.adam_state

    # Final: num_gaussians_counter[0] = scan_total[0]. GPU copy — no
    # readback, no upload.
    self._dispatch(self.k_densify_snapshot,
        thread_count=[1, 1, 1],
        src=b.scan_total.storage,
        dst=b.num_gaussians_counter.storage,
    )

    enc.pop_debug_group()
    cmd = enc.finish()
    with self.gpu_lock:
        self.last_submit_id = self.device.submit_command_buffer(cmd)
    self._encoder = None

    # No wait_for_idle. Return an empty stats dict; the caller reads
    # num_gaussians_counter on its own cadence (e.g. log-every).
    return {
        "pre_count": None,
        "post_split_count": None,
        "final_count": None,
    }


Pipeline.densify_simple = densify_simple  # type: ignore[attr-defined]


def cap_opacity(self: Pipeline, cap_value: float = 0.8) -> None:
    """Cap per-gaussian opacity at cap_value. Softer alternative to full
    opacity reset — lets gaussians keep earned opacity up to the cap."""
    b = self.buffers
    import math
    cap_logit = math.log(cap_value / (1.0 - cap_value))
    self._dispatch(self.k_opacity_reset,
        thread_count=[self.max_splats, 1, 1],
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        reset_value=cap_value,
    )


Pipeline.cap_opacity = cap_opacity  # type: ignore[attr-defined]


def reset_opacity(self: Pipeline, cfg: DensifyConfig) -> None:
    """Per-gaussian opacity <- low value. Forces optimizer to re-earn
    opacity on real geometry."""
    b = self.buffers
    self._dispatch(self.k_opacity_reset,
        thread_count=[self.max_splats, 1, 1],
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        reset_value=cfg.t_opacity_reset_value,
    )


Pipeline.reset_opacity = reset_opacity  # type: ignore[attr-defined]
