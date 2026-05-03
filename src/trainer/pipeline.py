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


@dataclass
class AdamLRs:
    """Adam learning rates. Values taken from the 3DGS paper's defaults;
    ``lr_mean`` is the only one with decay (PLAN.md Q15)."""
    mean: float = 1.6e-4
    scale: float = 5e-3
    rot: float = 1e-3
    opacity: float = 5e-2
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
        )

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
