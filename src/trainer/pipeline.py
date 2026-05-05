"""Dispatch orchestration for the v1 trainer.

A ``Pipeline`` owns a device, the compute kernels, and the GPU resources.
For inference, call :meth:`Pipeline.render`. For training, set a target
image with :meth:`Pipeline.set_target_image` and call
:meth:`Pipeline.train_step`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import slangpy as spy

from trainer import buffers
from trainer.buffers import GRAD_SCALAR_COMPONENTS
from trainer.dataset import Camera
from trainer.device import SHADERS_DIR


SCAN_BLOCK_SIZE = 1024  # must match shaders/common/scan.slang


@dataclass
class DensifyConfig:
    """FastGS-style multi-view importance score densification.

    Replaces gradient-threshold heuristics with a principled per-gaussian
    importance score: count of high-error pixels touched across N views.
    See arxiv 2511.04283 (FastGS).
    """
    n_scoring_views: int = 10       # cameras sampled for importance scoring
    loss_thresh: float = 0.1        # L1 threshold for binary error mask
    importance_thresh: float = 5.0  # min importance to allow split/clone
    t_split_scale: float = 0.01     # max_scale > this => split (else clone)
    t_opacity: float = 0.005        # prune if sigmoid(opacity) < this
    t_big_scale: float = 0.1        # prune if max_scale > this (world-space floater removal)
    growth_rate: float = 0.05       # max growth per densify step (fraction of current count)
    opacity_cap: float = 0.8        # post-densify opacity cap (softer than reset)


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
    color: float = 2.5e-3


def _make_kernel(device: spy.Device, path: Path, entry: str) -> spy.ComputeKernel:
    program = device.load_program(str(path), [entry])
    return device.create_compute_kernel(program)


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
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.max_splats = max_splats
        if max_sort_pairs is None:
            max_sort_pairs = buffers.DEFAULT_MAX_SORT_PAIRS
        self.max_sort_pairs = max_sort_pairs

        self.module = _module_for_reflection(device)

        # --- kernels ---
        tb = SHADERS_DIR / "rasterize" / "tile_binned"
        opt = SHADERS_DIR / "optimizers"
        loss = SHADERS_DIR / "losses"

        self.k_clear_textures = _make_kernel(device, tb / "clear.slang", "clearTextures")
        self.k_bin_count = _make_kernel(device, tb / "bin_count.slang", "binCount")
        self.k_bin_emit = _make_kernel(device, tb / "bin_emit.slang", "binEmit")
        self.k_clear_tile_ranges = _make_kernel(device, tb / "tile_ranges.slang", "clearTileRanges")
        self.k_identify_tile_ranges = _make_kernel(device, tb / "tile_ranges.slang", "identifyTileRanges")
        self.k_forward = _make_kernel(device, tb / "forward.slang", "forwardRasterize")
        self.k_loss = _make_kernel(device, loss / "l2_linear.slang", "computeLossGrad")
        self.k_loss_ssim = _make_kernel(device, loss / "ssim_loss.slang", "computeSSIMLossGrad")
        self.k_regularization = _make_kernel(device, loss / "regularization.slang", "addRegularization")
        self.k_backward = _make_kernel(device, tb / "backward.slang", "backwardRasterize")

        sort = SHADERS_DIR / "sort"
        self.k_radix_block_sort = _make_kernel(device, sort / "radix_sort.slang", "blockSort")
        self.k_radix_global_scan = _make_kernel(device, sort / "radix_sort.slang", "globalScan")
        self.k_radix_scan_digit_totals = _make_kernel(device, sort / "radix_sort.slang", "scanDigitTotals")
        self.k_radix_scatter_final = _make_kernel(device, sort / "radix_sort.slang", "scatterFinal")
        self.k_adam = _make_kernel(device, opt / "adam.slang", "adamStep")
        self.k_grad_clear = _make_kernel(device, opt / "grad_utils.slang", "clearGradBuffer")
        self.k_saturation_check = _make_kernel(device, opt / "grad_utils.slang", "checkSaturation")
        self.k_saturation_clear = _make_kernel(device, opt / "grad_utils.slang", "clearSaturationFlag")

        densify = SHADERS_DIR / "densify"
        self.k_clear_tracking = _make_kernel(device, densify / "clear_tracking.slang", "clearTracking")
        self.k_accumulate_iter = _make_kernel(device, densify / "accumulate_iter.slang", "accumulateIter")
        self.k_decide_actions = _make_kernel(device, densify / "decide_actions.slang", "decideActions")
        self.k_split_and_clone = _make_kernel(device, densify / "split_and_clone.slang", "splitAndClone")
        self.k_build_prune_mask = _make_kernel(device, densify / "build_prune_mask.slang", "buildPruneMask")
        self.k_scatter_compact = _make_kernel(device, densify / "scatter_compact.slang", "scatterCompact")
        self.k_opacity_reset = _make_kernel(device, densify / "opacity_reset.slang", "resetOpacity")
        self.k_clear_importance = _make_kernel(device, densify / "clear_importance.slang", "clearImportance")
        self.k_compute_error_mask = _make_kernel(device, densify / "compute_error_mask.slang", "computeErrorMask")
        self.k_score_forward = _make_kernel(device, densify / "score_forward.slang", "scoreForward")

        scan = SHADERS_DIR / "common" / "scan.slang"
        self.k_scan_block = _make_kernel(device, scan, "scanBlock")
        self.k_scan_block_sums = _make_kernel(device, scan, "scanBlockSums")
        self.k_add_block_offsets = _make_kernel(device, scan, "addBlockOffsets")

        # --- buffers ---
        self.buffers = buffers.allocate(
            device=device, module=self.module,
            max_splats=max_splats, width=width, height=height,
            max_sort_pairs=max_sort_pairs,
        )

        camera_type = self.module.layout.find_type_by_name("CameraData")
        if camera_type is None:
            raise RuntimeError("shader module does not expose type 'CameraData'")
        self._camera_buffer = spy.NDBuffer(device=device, dtype=camera_type, shape=(1,))

        # Training step counter (Python-side, for Adam bias correction).
        self.iteration = 0

        # Near-plane cull distance used by the bin kernel.
        self.z_near_cull = 0.01

        # Regularization weights (gsplat defaults: both 0; MCMC uses 0.001).
        self.opacity_reg = 0.01
        self.scale_reg = 0.01

        # Alpha loss weight. Set > 0 when target images have meaningful alpha.
        self.w_alpha = 0.0

        # SSIM loss mixing. 0 = pure L2, >0 = L1+SSIM blend (3DGS default 0.2).
        self.lambda_ssim = 0.2

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
        alpha=1. For RGBA inputs, RGB is premultiplied by alpha so that
        transparent regions have RGB=0 (matching our renderer's black
        background where nothing is rendered).
        """
        if img.ndim == 3 and img.shape[2] == 4:
            assert img.shape[:2] == (self.height, self.width)
            rgba = img.astype(np.float32).copy()
            rgba[..., :3] *= rgba[..., 3:4]
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
        cursor = self._camera_buffer.cursor()
        cursor[0].view_matrix = spy.float4x4(camera.view_matrix)
        cursor[0].proj_matrix = spy.float4x4(camera.proj_matrix)
        cursor[0].image_size = spy.float2(float(self.width), float(self.height))
        cursor.apply()

    def _radix_sort_pairs(self) -> tuple[spy.NDBuffer, spy.NDBuffer, spy.NDBuffer]:
        """Block-local 8-pass LSB radix sort. Returns the (keys_lo,
        keys_hi, values) triplet holding the sorted result.

        Per-pass structure (see shaders/sort/radix_sort.slang):
          blockSort       — in-place block sort + per-block digit counts
          globalScan      — per-(block, digit) running prefix + digit totals
          scanDigitTotals — exclusive scan of the 256 digit totals
          scatterFinal    — O(1)/thread scatter to the destination buffers

        ``n`` (the actual pair count) lives in ``scan_total[0]`` on the
        GPU; Python never reads it back in the hot path.
        """
        b = self.buffers
        src_lo, src_hi, src_v = b.sort_keys_lo_a, b.sort_keys_hi_a, b.sort_values_a
        dst_lo, dst_hi, dst_v = b.sort_keys_lo_b, b.sort_keys_hi_b, b.sort_values_b
        num_sort_blocks = (self.max_sort_pairs + 255) // 256

        for pass_idx in range(8):
            self.k_radix_block_sort.dispatch(
                thread_count=[num_sort_blocks * 256, 1, 1],
                keys_lo=src_lo.storage, keys_hi=src_hi.storage,
                values=src_v.storage,
                block_hist=b.radix_block_hist.storage,
                block_start=b.radix_block_start.storage,
                n_buf=b.scan_total.storage,
                pass_idx=pass_idx,
            )
            self.k_radix_global_scan.dispatch(
                thread_count=[256, 1, 1],
                block_hist=b.radix_block_hist.storage,
                block_prefix=b.radix_block_prefix.storage,
                digit_total=b.radix_digit_total.storage,
                n_buf=b.scan_total.storage,
            )
            self.k_radix_scan_digit_totals.dispatch(
                thread_count=[256, 1, 1],
                digit_total=b.radix_digit_total.storage,
                digit_global_base=b.radix_digit_global_base.storage,
            )
            self.k_radix_scatter_final.dispatch(
                thread_count=[self.max_sort_pairs, 1, 1],
                keys_lo_in=src_lo.storage, keys_hi_in=src_hi.storage, values_in=src_v.storage,
                block_start=b.radix_block_start.storage,
                block_prefix=b.radix_block_prefix.storage,
                digit_global_base=b.radix_digit_global_base.storage,
                keys_lo_out=dst_lo.storage, keys_hi_out=dst_hi.storage, values_out=dst_v.storage,
                n_buf=b.scan_total.storage,
                pass_idx=pass_idx,
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
        self.k_clear_textures.dispatch(
            thread_count=[self.width, self.height, 1],
            output_image=b.output_image,
            output_transmittance=b.output_transmittance,
            gaussian_count_per_pixel=b.gaussian_count_per_pixel,
            frame_dim=spy.uint2(self.width, self.height),
        )
        self.k_clear_tile_ranges.dispatch(
            thread_count=[num_tiles, 1, 1],
            tile_start=b.tile_start.storage,
            tile_end=b.tile_end.storage,
            num_tiles=num_tiles,
        )

        # Phase 1: per-gaussian tile count.
        self.k_bin_count.dispatch(
            thread_count=[self.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            camera=self._camera_buffer.storage,
            projected_2d=b.projected_2d.storage,
            tile_count_per_gaussian=b.tile_count_per_gaussian.storage,
            z_near=self.z_near_cull,
            max_splats=self.max_splats,
        )

        # Phase 2: exclusive scan → tile_offset_per_gaussian + total count.
        # scan_total[0] holds the total pair count on the GPU; no readback.
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
        self.k_bin_emit.dispatch(
            thread_count=[self.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            camera=self._camera_buffer.storage,
            projected_2d=b.projected_2d.storage,
            tile_offset_per_gaussian=b.tile_offset_per_gaussian.storage,
            keys_lo=b.sort_keys_lo_a.storage,
            keys_hi=b.sort_keys_hi_a.storage,
            values=b.sort_values_a.storage,
            z_near=self.z_near_cull,
        )

        # Phase 4: global radix sort. Reads n from scan_total[0].
        sorted_lo, sorted_hi, sorted_v = self._radix_sort_pairs()

        # Phase 5: identify per-tile start/end in the sorted array.
        self.k_identify_tile_ranges.dispatch(
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
        self.k_forward.dispatch(
            thread_count=[b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
            gaussians=b.gaussians.storage,
            projected_2d=b.projected_2d.storage,
            tile_start=b.tile_start.storage,
            tile_end=b.tile_end.storage,
            sorted_gaussian_ids=sorted_v.storage,
            camera=self._camera_buffer.storage,
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
            self.k_grad_clear.dispatch(
                thread_count=[self.max_splats, 1, 1],
                grad_buffer=b.grad_buffers[name].storage,
                max_splats=self.max_splats,
            )

    def _run_loss(self) -> None:
        b = self.buffers
        target = getattr(self, "_active_target", None) or b.target_image
        if self.lambda_ssim > 0:
            self.k_loss_ssim.dispatch(
                thread_count=[self.width, self.height, 1],
                output_image=b.output_image,
                target_image=target,
                dL_dC=b.dL_dC_texture,
                frame_dim=spy.uint2(self.width, self.height),
                lambda_ssim=self.lambda_ssim,
                w_alpha=self.w_alpha,
            )
        else:
            self.k_loss.dispatch(
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
        self.k_backward.dispatch(
            thread_count=[b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
            gaussians=b.gaussians.storage,
            projected_2d=b.projected_2d.storage,
            tile_start=b.tile_start.storage,
            tile_end=b.tile_end.storage,
            sorted_gaussian_ids=sorted_values.storage,
            camera=self._camera_buffer.storage,
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
        self.k_accumulate_iter.dispatch(
            thread_count=[self.max_splats, 1, 1],
            grad_2d_x_iter=b.grad_2d_x_iter.storage,
            grad_2d_y_iter=b.grad_2d_y_iter.storage,
            grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
            visibility_count_accum=b.visibility_count_accum.storage,
            max_splats=self.max_splats,
        )

    def _run_regularization(self) -> None:
        if self.opacity_reg <= 0 and self.scale_reg <= 0:
            return
        b = self.buffers
        gb = b.grad_buffers
        n = int(b.num_gaussians_counter.to_numpy().view(np.uint32)[0]) if False else self.max_splats
        self.k_regularization.dispatch(
            thread_count=[self.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            grad_opacity=gb["opacity"].storage,
            grad_log_scale_x=gb["log_scale_x"].storage,
            grad_log_scale_y=gb["log_scale_y"].storage,
            grad_log_scale_z=gb["log_scale_z"].storage,
            w_opacity=self.opacity_reg,
            w_scale=self.scale_reg,
            inv_n=1.0 / float(self.max_splats),
        )

    def clear_tracking(self) -> None:
        """Zero the v2 densification tracking accumulators. Called at the
        start of each densification window (every N iters)."""
        b = self.buffers
        self.k_clear_tracking.dispatch(
            thread_count=[self.max_splats, 1, 1],
            grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
            visibility_count_accum=b.visibility_count_accum.storage,
            grad_2d_x_iter=b.grad_2d_x_iter.storage,
            grad_2d_y_iter=b.grad_2d_y_iter.storage,
            max_splats=self.max_splats,
        )

    def read_tracking(self) -> dict[str, np.ndarray]:
        """Read window-level tracking buffers as float / uint arrays of
        length ``max_splats``. Used by tests and diagnostics."""
        b = self.buffers
        raw_mag = b.grad_mean_2d_mag_accum.to_numpy()
        raw_vis = b.visibility_count_accum.to_numpy()
        mag_int = np.frombuffer(raw_mag.tobytes(), dtype=np.int32)
        vis = np.frombuffer(raw_vis.tobytes(), dtype=np.uint32)
        return {
            "grad_mean_2d_mag": mag_int.astype(np.float64) / 65536.0,
            "visibility_count": vis.astype(np.int64),
        }

    def _run_saturation_check(self) -> None:
        b = self.buffers
        self.k_saturation_clear.dispatch(
            thread_count=[1, 1, 1],
            saturation_flag=b.saturation_flag.storage,
        )
        for name in GRAD_SCALAR_COMPONENTS:
            self.k_saturation_check.dispatch(
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
        self.k_adam.dispatch(
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
        self._clear_grads()
        self._run_forward(camera)
        self._run_loss()
        self._run_backward()
        self._run_regularization()
        self._run_adam(lrs)

    def profile_one_step(self, camera: Camera, lrs: AdamLRs | None = None,
                         warmup_steps: int = 3) -> list[tuple[str, float]]:
        """Run one full training step with per-phase GPU-timestamp queries
        and return ``[(label, ms), ...]``.

        A few untimed warmup steps run first so cold JIT / cache effects
        don't dominate the measurement. Prints the breakdown and returns
        it for programmatic use. Expects ``set_target_texture`` or
        ``set_target_image`` to have been called already.

        This method self-records timestamps on every dispatch; uses the
        per-kernel ``query_pool``/``query_index_before``/``query_index_after``
        knobs SlangPy exposes. The recorded values are wall times for
        that single GPU dispatch's execution, in milliseconds.
        """
        import time as _time

        if lrs is None:
            lrs = AdamLRs()

        for _ in range(warmup_steps):
            self._clear_grads()
            self._run_forward(camera)
            self._run_loss()
            self._run_backward()
            self._run_adam(lrs)
        self.device.wait_for_idle()

        # Allocate a fresh pool sized to the step's dispatch count.
        # Upper bound: ~30 for a non-densify step. 128 is plenty.
        pool = self.device.create_query_pool(spy.QueryType.timestamp, 128)
        b = self.buffers
        gb = b.grad_buffers
        num_tiles = b.num_tiles
        sorted_lo_after, sorted_hi_after, sorted_v_after = (
            b.sort_keys_lo_a, b.sort_keys_hi_a, b.sort_values_a,
        )

        # Each entry: (label, callable). We allocate query indices
        # incrementally, two per dispatch.
        events: list[tuple[str, "callable"]] = []
        idx = [0]

        def _time_kernel(label: str, kernel: spy.ComputeKernel,
                         thread_count: list[int], **kwargs) -> None:
            i0 = idx[0]; i1 = i0 + 1; idx[0] += 2
            kernel.dispatch(thread_count=thread_count,
                            query_pool=pool,
                            query_index_before=i0,
                            query_index_after=i1,
                            **kwargs)
            events.append((label, i0))

        # --- Upload camera (CPU side, not timed in GPU queries but
        # captured by bracket-wall-time for reference). ---
        t0_wall = _time.perf_counter()

        # Clear grads (14 dispatches; time only the first as
        # representative — the 14 are identical work so summing would
        # just be 14× a single dispatch's cost).
        _time_kernel("clear_grads (1 of 14)", self.k_grad_clear,
                     [self.max_splats, 1, 1],
                     grad_buffer=gb[GRAD_SCALAR_COMPONENTS[0]].storage,
                     max_splats=self.max_splats)
        for name in GRAD_SCALAR_COMPONENTS[1:]:
            self.k_grad_clear.dispatch(
                thread_count=[self.max_splats, 1, 1],
                grad_buffer=gb[name].storage, max_splats=self.max_splats,
            )

        # Forward.
        self._upload_camera(camera)
        _time_kernel("fwd: clear_textures", self.k_clear_textures,
                     [self.width, self.height, 1],
                     output_image=b.output_image,
                     output_transmittance=b.output_transmittance,
                     gaussian_count_per_pixel=b.gaussian_count_per_pixel,
                     frame_dim=spy.uint2(self.width, self.height))
        _time_kernel("fwd: clear_tile_ranges", self.k_clear_tile_ranges,
                     [num_tiles, 1, 1],
                     tile_start=b.tile_start.storage,
                     tile_end=b.tile_end.storage,
                     num_tiles=num_tiles)
        _time_kernel("fwd: bin_count", self.k_bin_count,
                     [self.max_splats, 1, 1],
                     gaussians=b.gaussians.storage,
                     num_gaussians_counter=b.num_gaussians_counter.storage,
                     camera=self._camera_buffer.storage,
                     projected_2d=b.projected_2d.storage,
                     tile_count_per_gaussian=b.tile_count_per_gaussian.storage,
                     z_near=self.z_near_cull,
                     max_splats=self.max_splats)

        # Scan is three dispatches; time them individually.
        in_buf, out_buf = b.tile_count_per_gaussian, b.tile_offset_per_gaussian
        num_blocks = max(1, (self.max_splats + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE)
        _time_kernel("fwd: scan_block", self.k_scan_block,
                     [num_blocks * SCAN_BLOCK_SIZE, 1, 1],
                     in_buf=in_buf.storage, out_buf=out_buf.storage,
                     block_sums=b.scan_block_sums.storage, n=self.max_splats)
        _time_kernel("fwd: scan_block_sums", self.k_scan_block_sums,
                     [SCAN_BLOCK_SIZE, 1, 1],
                     block_sums=b.scan_block_sums.storage,
                     total_out=b.scan_total.storage, num_blocks=num_blocks)
        _time_kernel("fwd: scan_add_offsets", self.k_add_block_offsets,
                     [num_blocks * SCAN_BLOCK_SIZE, 1, 1],
                     out_buf=out_buf.storage,
                     block_sums=b.scan_block_sums.storage, n=self.max_splats)

        _time_kernel("fwd: bin_emit", self.k_bin_emit,
                     [self.max_splats, 1, 1],
                     gaussians=b.gaussians.storage,
                     num_gaussians_counter=b.num_gaussians_counter.storage,
                     camera=self._camera_buffer.storage,
                     projected_2d=b.projected_2d.storage,
                     tile_offset_per_gaussian=b.tile_offset_per_gaussian.storage,
                     keys_lo=b.sort_keys_lo_a.storage,
                     keys_hi=b.sort_keys_hi_a.storage,
                     values=b.sort_values_a.storage,
                     z_near=self.z_near_cull)

        # Radix sort: 8 passes × 4 dispatches per pass.
        src_lo, src_hi, src_v = b.sort_keys_lo_a, b.sort_keys_hi_a, b.sort_values_a
        dst_lo, dst_hi, dst_v = b.sort_keys_lo_b, b.sort_keys_hi_b, b.sort_values_b
        num_sort_blocks = (self.max_sort_pairs + 255) // 256
        radix_events_per_pass: list[tuple[int, int, int, int]] = []
        for pass_idx in range(8):
            i_bs  = idx[0]; idx[0] += 2
            i_gs  = idx[0]; idx[0] += 2
            i_sdt = idx[0]; idx[0] += 2
            i_sf  = idx[0]; idx[0] += 2
            radix_events_per_pass.append((i_bs, i_gs, i_sdt, i_sf))
            self.k_radix_block_sort.dispatch(
                thread_count=[num_sort_blocks * 256, 1, 1],
                keys_lo=src_lo.storage, keys_hi=src_hi.storage,
                values=src_v.storage,
                block_hist=b.radix_block_hist.storage,
                block_start=b.radix_block_start.storage,
                n_buf=b.scan_total.storage, pass_idx=pass_idx,
                query_pool=pool, query_index_before=i_bs, query_index_after=i_bs + 1,
            )
            self.k_radix_global_scan.dispatch(
                thread_count=[256, 1, 1],
                block_hist=b.radix_block_hist.storage,
                block_prefix=b.radix_block_prefix.storage,
                digit_total=b.radix_digit_total.storage,
                n_buf=b.scan_total.storage,
                query_pool=pool, query_index_before=i_gs, query_index_after=i_gs + 1,
            )
            self.k_radix_scan_digit_totals.dispatch(
                thread_count=[256, 1, 1],
                digit_total=b.radix_digit_total.storage,
                digit_global_base=b.radix_digit_global_base.storage,
                query_pool=pool, query_index_before=i_sdt, query_index_after=i_sdt + 1,
            )
            self.k_radix_scatter_final.dispatch(
                thread_count=[self.max_sort_pairs, 1, 1],
                keys_lo_in=src_lo.storage, keys_hi_in=src_hi.storage, values_in=src_v.storage,
                block_start=b.radix_block_start.storage,
                block_prefix=b.radix_block_prefix.storage,
                digit_global_base=b.radix_digit_global_base.storage,
                keys_lo_out=dst_lo.storage, keys_hi_out=dst_hi.storage, values_out=dst_v.storage,
                n_buf=b.scan_total.storage, pass_idx=pass_idx,
                query_pool=pool, query_index_before=i_sf, query_index_after=i_sf + 1,
            )
            src_lo, dst_lo = dst_lo, src_lo
            src_hi, dst_hi = dst_hi, src_hi
            src_v, dst_v = dst_v, src_v
        sorted_v = src_v

        _time_kernel("fwd: identify_tile_ranges", self.k_identify_tile_ranges,
                     [self.max_sort_pairs, 1, 1],
                     keys_hi=src_hi.storage,
                     tile_start=b.tile_start.storage,
                     tile_end=b.tile_end.storage,
                     n_buf=b.scan_total.storage,
                     num_tiles=num_tiles)

        _time_kernel("fwd: forward_rasterize", self.k_forward,
                     [b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
                     gaussians=b.gaussians.storage,
                     projected_2d=b.projected_2d.storage,
                     tile_start=b.tile_start.storage,
                     tile_end=b.tile_end.storage,
                     sorted_gaussian_ids=sorted_v.storage,
                     camera=self._camera_buffer.storage,
                     output_image=b.output_image,
                     output_transmittance=b.output_transmittance,
                     gaussian_count_per_pixel=b.gaussian_count_per_pixel)

        # Loss.
        target = getattr(self, "_active_target", None) or b.target_image
        _time_kernel("loss", self.k_loss,
                     [self.width, self.height, 1],
                     output_image=b.output_image,
                     target_image=target,
                     dL_dC=b.dL_dC_texture,
                     frame_dim=spy.uint2(self.width, self.height))

        # Backward.
        _time_kernel("bwd: backwardRasterize", self.k_backward,
                     [b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
                     gaussians=b.gaussians.storage,
                     projected_2d=b.projected_2d.storage,
                     tile_start=b.tile_start.storage,
                     tile_end=b.tile_end.storage,
                     sorted_gaussian_ids=sorted_v.storage,
                     camera=self._camera_buffer.storage,
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
                     grad_2d_y_iter=b.grad_2d_y_iter.storage)

        _time_kernel("bwd: accumulate_iter", self.k_accumulate_iter,
                     [self.max_splats, 1, 1],
                     grad_2d_x_iter=b.grad_2d_x_iter.storage,
                     grad_2d_y_iter=b.grad_2d_y_iter.storage,
                     grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
                     visibility_count_accum=b.visibility_count_accum.storage,
                     max_splats=self.max_splats)

        # Adam.
        self.iteration += 1
        step = float(self.iteration)
        bc1 = 1.0 - 0.9 ** step
        bc2 = 1.0 - 0.999 ** step
        _time_kernel("adam", self.k_adam,
                     [self.max_splats, 1, 1],
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
                     cfg={"lr_mean": lrs.mean, "lr_scale": lrs.scale,
                          "lr_rot": lrs.rot, "lr_opacity": lrs.opacity,
                          "lr_color": lrs.color,
                          "bias_correction1": bc1, "bias_correction2": bc2},
                     pixels_per_step=float(self.width * self.height))

        self.device.wait_for_idle()
        wall_ms = (_time.perf_counter() - t0_wall) * 1000.0

        ts = pool.get_timestamp_results(0, idx[0])
        # Results are in seconds per SlangPy docs; convert to ms.
        results: list[tuple[str, float]] = []
        for label, i0_q in events:
            dt_ms = (ts[i0_q + 1] - ts[i0_q]) * 1000.0
            results.append((label, dt_ms))

        # Radix sort: 4 kernels × 8 passes → 5 summary rows.
        radix_bs  = sum((ts[p[0] + 1] - ts[p[0]]) * 1000.0 for p in radix_events_per_pass)
        radix_gs  = sum((ts[p[1] + 1] - ts[p[1]]) * 1000.0 for p in radix_events_per_pass)
        radix_sdt = sum((ts[p[2] + 1] - ts[p[2]]) * 1000.0 for p in radix_events_per_pass)
        radix_sf  = sum((ts[p[3] + 1] - ts[p[3]]) * 1000.0 for p in radix_events_per_pass)
        radix_total = radix_bs + radix_gs + radix_sdt + radix_sf
        insert_at = next(i for i, (lbl, _) in enumerate(results) if lbl == "fwd: bin_emit") + 1
        radix_rows = [
            ("fwd: radix blockSort        (×8)", radix_bs),
            ("fwd: radix globalScan       (×8)", radix_gs),
            ("fwd: radix scanDigitTotals  (×8)", radix_sdt),
            ("fwd: radix scatterFinal     (×8)", radix_sf),
            ("fwd: radix sort TOTAL       (×8)", radix_total),
        ]
        results = results[:insert_at] + radix_rows + results[insert_at:]

        # For the percent calculation we sum everything EXCEPT the
        # radix TOTAL (that row is derived; counting it double-counts).
        def _countable(label: str) -> bool:
            return "radix sort TOTAL" not in label
        total_gpu = sum(ms for lbl, ms in results if _countable(lbl))

        print(f"\n--- profile_one_step (n_g={int(b.num_gaussians_counter.to_numpy().view(np.uint32)[0])}, "
              f"max_sort_pairs={self.max_sort_pairs}) ---")
        print(f"{'phase':<34s}{'ms':>10s}{'% of GPU':>12s}")
        print("-" * 56)
        for label, ms in results:
            if _countable(label):
                pct = (ms / total_gpu * 100.0) if total_gpu > 0 else 0.0
                print(f"{label:<34s}{ms:>10.3f}{pct:>11.1f}%")
            else:
                print(f"{label:<34s}{ms:>10.3f}{'':>12s}")
        print("-" * 56)
        print(f"{'total GPU (sum of phases)':<34s}{total_gpu:>10.3f}")
        print(f"{'wall-clock (submit + idle)':<34s}{wall_ms:>10.3f}")
        return results

    # ------------------------------------------------------------------
    # Gradient readback (for the finite-difference test)
    # ------------------------------------------------------------------

    def read_gradients(self) -> dict[str, np.ndarray]:
        """Read back the 14 SoA grad buffers as float arrays of length
        ``num_gaussians``. Fixed-point division happens on CPU."""
        from trainer.buffers import GRAD_SCALAR_COMPONENTS as NAMES

        n = int(self.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
        inv = 1.0 / 65536.0
        out: dict[str, np.ndarray] = {}
        for name in NAMES:
            raw = self.buffers.grad_buffers[name].to_numpy()
            ints = np.frombuffer(raw.tobytes(), dtype=np.int32)[:n]
            out[name] = ints.astype(np.float64) * inv
        return out

    def read_loss(self) -> float:
        """Read the L2 loss as a scalar. Computed on CPU from the
        output_image and target_image — the shader doesn't accumulate
        the scalar itself (we only needed dL/dC for the backward)."""
        C = self.buffers.output_image.to_numpy()[..., :3]
        T_rgba = self.buffers.target_image.to_numpy()[..., :3]
        T_linear = _srgb_to_linear_np(T_rgba)
        diff = C - T_linear
        return float(np.sum(diff * diff))


def _srgb_to_linear_np(c: np.ndarray) -> np.ndarray:
    low = c / 12.92
    high = np.power(np.clip((c + 0.055) / 1.055, 1e-7, None), 2.4)
    return np.where(c <= 0.04045, low, high).astype(np.float32)


# -------- densification methods on Pipeline (added via monkey-patch-style
# direct method definitions; kept here for proximity to the scan + buffer
# structures they orchestrate) --------

def _scan(self: Pipeline, n: int,
          in_buf: spy.NDBuffer | None = None,
          out_buf: spy.NDBuffer | None = None) -> None:
    """Run the three-kernel Blelloch scan on ``in_buf`` into ``out_buf``,
    writing the total into ``scan_total[0]``.

    Defaults preserve the v2 densification path's behavior (scan the
    valid-mask buffer into the scan output buffer).
    """
    b = self.buffers
    if in_buf is None:
        in_buf = b.valid_mask_buffer
    if out_buf is None:
        out_buf = b.scan_buffer

    num_blocks = max(1, (n + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE)

    self.k_scan_block.dispatch(
        thread_count=[num_blocks * SCAN_BLOCK_SIZE, 1, 1],
        in_buf=in_buf.storage,
        out_buf=out_buf.storage,
        block_sums=b.scan_block_sums.storage,
        n=n,
    )
    self.k_scan_block_sums.dispatch(
        thread_count=[SCAN_BLOCK_SIZE, 1, 1],
        block_sums=b.scan_block_sums.storage,
        total_out=b.scan_total.storage,
        num_blocks=num_blocks,
    )
    self.k_add_block_offsets.dispatch(
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

    self.k_clear_importance.dispatch(
        thread_count=[self.max_splats, 1, 1],
        importance_score_accum=b.importance_score_accum.storage,
        max_splats=self.max_splats,
    )

    for cam, target_tex in zip(cameras, target_textures):
        self._run_forward(cam)

        self.k_compute_error_mask.dispatch(
            thread_count=[self.width, self.height, 1],
            output_image=b.output_image,
            target_image=target_tex,
            error_mask=b.error_mask,
            frame_dim=spy.uint2(self.width, self.height),
            loss_thresh=cfg.loss_thresh,
        )

        sorted_values = getattr(self, "_sorted_values", b.sort_values_a)
        self.k_score_forward.dispatch(
            thread_count=[b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
            gaussians=b.gaussians.storage,
            projected_2d=b.projected_2d.storage,
            tile_start=b.tile_start.storage,
            tile_end=b.tile_end.storage,
            sorted_gaussian_ids=sorted_values.storage,
            camera=self._camera_buffer.storage,
            error_mask=b.error_mask,
            importance_score_accum=b.importance_score_accum.storage,
        )


Pipeline.compute_importance_scores = compute_importance_scores  # type: ignore[attr-defined]


def densify(self: Pipeline, cfg: DensifyConfig) -> dict:
    """Run one densification pass using FastGS importance scores.

    Assumes compute_importance_scores() has been called beforehand to
    populate the importance_score_accum buffer.

    Phases:
      A  decide_actions     - write action[i] per gaussian (importance-based)
      B  split_and_clone    - append children, overwrite parent slot for splits
      C  build_prune_mask   - mark valid[i] across post-split range
      D  scan               - exclusive prefix-sum of valid[]
      E  scatter_compact    - ping-pong into scratch; Python swaps refs

    Returns a stats dict for logging.
    """
    b = self.buffers

    self.device.wait_for_idle()
    pre_count = int(b.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    if pre_count == 0:
        return {"pre_count": 0, "post_split_count": 0, "final_count": 0,
                "n_split": 0, "n_clone": 0, "n_prune": 0, "n_keep": 0}

    # Growth budget: max new gaussians this step.
    growth = max(1, int(cfg.growth_rate * pre_count))
    b.growth_budget.copy_from_numpy(np.array([growth], dtype=np.int32))
    # Prune budget: at most 50% of candidates actually pruned (FastGS).
    # We don't know the candidate count ahead of time, so set budget to
    # 50% of current count — an upper bound on candidates.
    prune = max(1, pre_count // 2)
    b.prune_budget.copy_from_numpy(np.array([prune], dtype=np.int32))

    self.k_decide_actions.dispatch(
        thread_count=[pre_count, 1, 1],
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        importance_score_accum=b.importance_score_accum.storage,
        action=b.action_buffer.storage,
        growth_budget=b.growth_budget.storage,
        prune_budget=b.prune_budget.storage,
        importance_thresh=cfg.importance_thresh,
        t_split_scale=cfg.t_split_scale,
        t_opacity=cfg.t_opacity,
        t_big_scale=cfg.t_big_scale,
        n_scoring_views=cfg.n_scoring_views,
    )

    # Phase B: split and clone (may increment counter).
    self.k_split_and_clone.dispatch(
        thread_count=[pre_count, 1, 1],
        gaussians=b.gaussians.storage,
        adam_state=b.adam_state.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        action=b.action_buffer.storage,
        pre_count=pre_count,
        max_splats=self.max_splats,
    )
    self.device.wait_for_idle()
    post_split_count = int(b.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    post_split_count = min(post_split_count, self.max_splats)

    # Phases C/D/E: compact.
    self.k_build_prune_mask.dispatch(
        thread_count=[post_split_count, 1, 1],
        action=b.action_buffer.storage,
        valid_mask=b.valid_mask_buffer.storage,
        pre_count=pre_count,
        post_count=post_split_count,
    )
    self._scan(post_split_count)
    self.k_scatter_compact.dispatch(
        thread_count=[post_split_count, 1, 1],
        gaussians_in=b.gaussians.storage,
        adam_state_in=b.adam_state.storage,
        valid_mask=b.valid_mask_buffer.storage,
        scan=b.scan_buffer.storage,
        gaussians_out=b.gaussians_scratch.storage,
        adam_state_out=b.adam_state_scratch.storage,
        post_count=post_split_count,
    )

    b.gaussians, b.gaussians_scratch = b.gaussians_scratch, b.gaussians
    b.adam_state, b.adam_state_scratch = b.adam_state_scratch, b.adam_state

    self.device.wait_for_idle()
    final_count = int(b.scan_total.to_numpy().view(np.uint32)[0])
    b.num_gaussians_counter.copy_from_numpy(np.array([final_count], dtype=np.uint32))

    actions_raw = b.action_buffer.to_numpy()
    actions = np.frombuffer(actions_raw.tobytes(), dtype=np.uint32)[:pre_count]
    n_keep  = int((actions == 0).sum())
    n_split = int((actions == 1).sum())
    n_clone = int((actions == 2).sum())
    n_prune = int((actions == 3).sum())

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


def cap_opacity(self: Pipeline, cap_value: float = 0.8) -> None:
    """Cap per-gaussian opacity at cap_value. Softer alternative to full
    opacity reset — lets gaussians keep earned opacity up to the cap."""
    b = self.buffers
    import math
    cap_logit = math.log(cap_value / (1.0 - cap_value))
    self.k_opacity_reset.dispatch(
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
    self.k_opacity_reset.dispatch(
        thread_count=[self.max_splats, 1, 1],
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        reset_value=0.01,
    )


Pipeline.reset_opacity = reset_opacity  # type: ignore[attr-defined]
