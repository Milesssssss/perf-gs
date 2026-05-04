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
    """Thresholds for the decide-actions kernel.

    Defaults match gsplat's ``DefaultStrategyConfig``:
      grow_grad2d  = 2e-4
      grow_scale3d = 0.01
      prune_opa    = 0.005
      prune_scale3d= 0.1
    which are in turn drawn from Kerbl et al. (3DGS).
    """
    t_grad: float = 2e-4           # gsplat grow_grad2d
    t_split_scale: float = 0.01    # gsplat grow_scale3d (split vs clone threshold)
    t_opacity: float = 0.005       # gsplat prune_opa
    t_big_scale: float = 0.1       # gsplat prune_scale3d — world-space scale cap
    opacity_reset_value: float = 0.01


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
        max_splats: int = 5_000_000,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.max_splats = max_splats

        self.module = _module_for_reflection(device)

        # --- kernels ---
        tb = SHADERS_DIR / "rasterize" / "tile_binned"
        opt = SHADERS_DIR / "optimizers"
        loss = SHADERS_DIR / "losses"

        self.k_clear_tiles = _make_kernel(device, tb / "clear.slang", "clearTilesCount")
        self.k_clear_textures = _make_kernel(device, tb / "clear.slang", "clearTextures")
        self.k_bin = _make_kernel(device, tb / "bin.slang", "binGaussians")
        self.k_sort = _make_kernel(device, tb / "sort.slang", "sortTile")
        self.k_forward = _make_kernel(device, tb / "forward.slang", "forwardRasterize")
        self.k_loss = _make_kernel(device, loss / "l2_linear.slang", "computeLossGrad")
        self.k_backward = _make_kernel(device, tb / "backward.slang", "backwardRasterize")
        self.k_adam = _make_kernel(device, opt / "adam.slang", "adamStep")
        self.k_grad_clear = _make_kernel(device, opt / "grad_utils.slang", "clearGradBuffer")
        self.k_saturation_check = _make_kernel(device, opt / "grad_utils.slang", "checkSaturation")
        self.k_saturation_clear = _make_kernel(device, opt / "grad_utils.slang", "clearSaturationFlag")

        densify = SHADERS_DIR / "densify"
        self.k_clear_tracking = _make_kernel(device, densify / "clear_tracking.slang", "clearTracking")
        self.k_decide_actions = _make_kernel(device, densify / "decide_actions.slang", "decideActions")
        self.k_split_and_clone = _make_kernel(device, densify / "split_and_clone.slang", "splitAndClone")
        self.k_build_prune_mask = _make_kernel(device, densify / "build_prune_mask.slang", "buildPruneMask")
        self.k_scatter_compact = _make_kernel(device, densify / "scatter_compact.slang", "scatterCompact")
        self.k_opacity_reset = _make_kernel(device, densify / "opacity_reset.slang", "resetOpacity")

        scan = SHADERS_DIR / "common" / "scan.slang"
        self.k_scan_block = _make_kernel(device, scan, "scanBlock")
        self.k_scan_block_sums = _make_kernel(device, scan, "scanBlockSums")
        self.k_add_block_offsets = _make_kernel(device, scan, "addBlockOffsets")

        # --- buffers ---
        self.buffers = buffers.allocate(
            device=device, module=self.module,
            max_splats=max_splats, width=width, height=height,
        )

        camera_type = self.module.layout.find_type_by_name("CameraData")
        if camera_type is None:
            raise RuntimeError("shader module does not expose type 'CameraData'")
        self._camera_buffer = spy.NDBuffer(device=device, dtype=camera_type, shape=(1,))

        # Training step counter (Python-side, for Adam bias correction).
        self.iteration = 0

        # Near-plane cull distance used by the bin kernel. Gaussians
        # closer than this in view space are skipped for the current
        # frame (they stay in the buffer). Default matches the projection
        # matrix z_near used by our camera helpers. Callers can adjust
        # per-scene (larger scenes with points potentially near a camera
        # pose benefit from raising this, e.g. 0.2).
        self.z_near_cull = 0.01

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
        """Upload a target image as ``(H, W, 3) float32`` sRGB.

        The loss kernel performs ``srgb_to_linear`` on read. We pad the
        image to RGBA because our texture format is rgba32_float.
        """
        assert rgb.shape == (self.height, self.width, 3), (
            f"target shape {rgb.shape} != ({self.height}, {self.width}, 3)"
        )
        rgba = np.concatenate(
            [rgb.astype(np.float32), np.ones((self.height, self.width, 1), np.float32)],
            axis=-1,
        )
        self.buffers.target_image.copy_from_numpy(np.ascontiguousarray(rgba))

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

    def _run_forward(self, camera: Camera) -> None:
        self._upload_camera(camera)
        b = self.buffers
        num_tiles = b.num_tiles

        self.k_clear_tiles.dispatch(
            thread_count=[num_tiles, 1, 1],
            screen_tiles_count=b.screen_tiles_count.storage,
            num_tiles=num_tiles,
        )
        self.k_clear_textures.dispatch(
            thread_count=[self.width, self.height, 1],
            output_image=b.output_image,
            output_transmittance=b.output_transmittance,
            gaussian_count_per_pixel=b.gaussian_count_per_pixel,
            frame_dim=spy.uint2(self.width, self.height),
        )
        self.k_bin.dispatch(
            thread_count=[self.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            camera=self._camera_buffer.storage,
            projected_2d=b.projected_2d.storage,
            screen_tiles_count=b.screen_tiles_count.storage,
            screen_tiles_indices=b.screen_tiles_indices.storage,
            z_near=self.z_near_cull,
        )
        self.k_sort.dispatch(
            thread_count=[num_tiles * 256, 1, 1],
            projected_2d=b.projected_2d.storage,
            screen_tiles_count=b.screen_tiles_count.storage,
            screen_tiles_indices=b.screen_tiles_indices.storage,
        )
        self.k_forward.dispatch(
            thread_count=[b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
            gaussians=b.gaussians.storage,
            projected_2d=b.projected_2d.storage,
            screen_tiles_count=b.screen_tiles_count.storage,
            screen_tiles_indices=b.screen_tiles_indices.storage,
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
        self.k_loss.dispatch(
            thread_count=[self.width, self.height, 1],
            output_image=b.output_image,
            target_image=b.target_image,
            dL_dC=b.dL_dC_texture,
            frame_dim=spy.uint2(self.width, self.height),
        )

    def _run_backward(self) -> None:
        b = self.buffers
        gb = b.grad_buffers
        self.k_backward.dispatch(
            thread_count=[b.num_tiles_x * 16, b.num_tiles_y * 16, 1],
            gaussians=b.gaussians.storage,
            projected_2d=b.projected_2d.storage,
            screen_tiles_count=b.screen_tiles_count.storage,
            screen_tiles_indices=b.screen_tiles_indices.storage,
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
            grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
            pixel_coverage_accum=b.pixel_coverage_accum.storage,
        )

    def clear_tracking(self) -> None:
        """Zero the v2 densification tracking accumulators. Called at the
        start of each densification window (every N iters)."""
        b = self.buffers
        self.k_clear_tracking.dispatch(
            thread_count=[self.max_splats, 1, 1],
            grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
            pixel_coverage_accum=b.pixel_coverage_accum.storage,
            max_splats=self.max_splats,
        )

    def read_tracking(self) -> dict[str, np.ndarray]:
        """Read both tracking buffers as float / uint arrays of length
        ``max_splats``. Used by tests and (temporarily) by Python-side
        densification prototyping."""
        b = self.buffers
        raw_mag = b.grad_mean_2d_mag_accum.to_numpy()
        raw_cov = b.pixel_coverage_accum.to_numpy()
        mag_int = np.frombuffer(raw_mag.tobytes(), dtype=np.int32)
        cov = np.frombuffer(raw_cov.tobytes(), dtype=np.uint32)
        return {
            "grad_mean_2d_mag": mag_int.astype(np.float64) / 65536.0,
            "pixel_coverage": cov.astype(np.int64),
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
        before the backward pass runs."""
        self._run_forward(camera)
        self._run_loss()
        self.device.wait_for_idle()

    def backward(self) -> None:
        """Run the backward rasterizer. Assumes forward + loss have run."""
        self._run_backward()
        self.device.wait_for_idle()

    def check_saturation(self) -> bool:
        """Return True if any int32 gradient buffer saturated."""
        self._run_saturation_check()
        self.device.wait_for_idle()
        flag = self.buffers.saturation_flag.to_numpy().view(np.uint32)[0]
        return flag != 0

    def train_step(self, camera: Camera, lrs: AdamLRs | None = None) -> None:
        """Run one training step: forward → loss → backward → adam.

        The target image must have been uploaded via
        :meth:`set_target_image` before calling.
        """
        if lrs is None:
            lrs = AdamLRs()
        self._clear_grads()
        self._run_forward(camera)
        self._run_loss()
        self._run_backward()
        self._run_adam(lrs)
        self.device.wait_for_idle()

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

def _scan(self: Pipeline, n: int) -> None:
    """Run the three-kernel Blelloch scan on ``valid_mask_buffer`` into
    ``scan_buffer``. ``scan_total[0]`` ends up holding the total sum
    (i.e. the new live count after compaction)."""
    b = self.buffers
    num_blocks = max(1, (n + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE)

    self.k_scan_block.dispatch(
        thread_count=[num_blocks * SCAN_BLOCK_SIZE, 1, 1],
        in_buf=b.valid_mask_buffer.storage,
        out_buf=b.scan_buffer.storage,
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
        out_buf=b.scan_buffer.storage,
        block_sums=b.scan_block_sums.storage,
        n=n,
    )


Pipeline._scan = _scan  # type: ignore[attr-defined]


def densify(self: Pipeline, cfg: DensifyConfig, window_iters: int) -> dict:
    """Run one densification pass.

    Phases:
      A  decide_actions     - write action[i] per gaussian
      B  split_and_clone    - append children, overwrite parent slot for splits
      C  build_prune_mask   - mark valid[i] across post-split range
      D  scan               - exclusive prefix-sum of valid[]
      E  scatter_compact    - ping-pong into scratch; Python swaps refs

    Returns a stats dict for logging:
        {pre_count, post_split_count, final_count, n_split, n_clone, n_prune}

    Does NOT clear tracking buffers — caller decides cadence. Does NOT
    reset opacity — that's a separate scheduled operation.
    """
    b = self.buffers

    # --- Phase A: decide actions ---
    pre_count = int(b.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    if pre_count == 0:
        return {"pre_count": 0, "post_split_count": 0, "final_count": 0,
                "n_split": 0, "n_clone": 0, "n_prune": 0}

    self.k_decide_actions.dispatch(
        thread_count=[pre_count, 1, 1],
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        grad_mean_2d_mag_accum=b.grad_mean_2d_mag_accum.storage,
        pixel_coverage_accum=b.pixel_coverage_accum.storage,
        action=b.action_buffer.storage,
        window_iters=float(window_iters),
        grad_scale=1.0 / float(max(self.width, self.height)),
        t_grad=cfg.t_grad,
        t_split_scale=cfg.t_split_scale,
        t_opacity=cfg.t_opacity,
        t_big_scale=cfg.t_big_scale,
    )
    self.device.wait_for_idle()

    # --- Phase B: split and clone (may increment counter) ---
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
    # Clamp: if Phase B was racing against max_splats some children may have
    # been dropped (atomic counter went past max; the writes past max were
    # guarded by the shader, but the counter itself still advanced). Clamp
    # so the mask pass doesn't walk past the real tail.
    post_split_count = min(post_split_count, self.max_splats)

    # --- Phase C: build prune mask over [0, post_split_count) ---
    self.k_build_prune_mask.dispatch(
        thread_count=[post_split_count, 1, 1],
        action=b.action_buffer.storage,
        valid_mask=b.valid_mask_buffer.storage,
        pre_count=pre_count,
        post_count=post_split_count,
    )
    self.device.wait_for_idle()

    # --- Phase D: scan ---
    self._scan(post_split_count)
    self.device.wait_for_idle()

    # --- Phase E: scatter compact into scratch buffers ---
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
    self.device.wait_for_idle()

    # --- Swap primary <-> scratch references ---
    b.gaussians, b.gaussians_scratch = b.gaussians_scratch, b.gaussians
    b.adam_state, b.adam_state_scratch = b.adam_state_scratch, b.adam_state

    # Update counter to scan_total (= sum of valid_mask = final live count).
    final_count = int(b.scan_total.to_numpy().view(np.uint32)[0])
    b.num_gaussians_counter.copy_from_numpy(np.array([final_count], dtype=np.uint32))

    # Stats for logging. Reading action_buffer back is ~pre_count bytes,
    # cheap on pre_count <= ~100k.
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


def reset_opacity(self: Pipeline, cfg: DensifyConfig) -> None:
    """Per-gaussian opacity <- cfg.opacity_reset_value. Standard 3DGS
    trick: forces optimizer to re-earn opacity on real geometry."""
    b = self.buffers
    self.k_opacity_reset.dispatch(
        thread_count=[self.max_splats, 1, 1],
        gaussians=b.gaussians.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        reset_value=cfg.opacity_reset_value,
    )
    self.device.wait_for_idle()


Pipeline.reset_opacity = reset_opacity  # type: ignore[attr-defined]
