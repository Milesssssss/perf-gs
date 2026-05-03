"""Dispatch orchestration for the tile-binned forward rasterizer (M3).

A ``Pipeline`` owns a device, the compute kernels, and the GPU resources.
Call :meth:`Pipeline.render` once per frame; the resulting image is
available via :meth:`Pipeline.read_output_image` or written to disk via
:func:`trainer.export.render_to_png`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import slangpy as spy

from trainer import buffers
from trainer.dataset import Camera
from trainer.device import SHADERS_DIR


def _module_for_reflection(device: spy.Device) -> spy.Module:
    """Load one of the shader files that declares the shared struct types.

    Any file that includes ``common/types.slang`` works — we just need a
    module to reflect against so :mod:`trainer.buffers` can look up
    ``Gaussian3D`` / ``Projected2DGaussian``.
    """
    return spy.Module.load_from_file(
        device, str(SHADERS_DIR / "rasterize" / "tile_binned" / "bin.slang")
    )


def _make_kernel(device: spy.Device, path: Path, entry: str) -> spy.ComputeKernel:
    program = device.load_program(str(path), [entry])
    return device.create_compute_kernel(program)


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

        tb = SHADERS_DIR / "rasterize" / "tile_binned"
        self.kernel_clear_tiles = _make_kernel(device, tb / "clear.slang", "clearTilesCount")
        self.kernel_clear_textures = _make_kernel(device, tb / "clear.slang", "clearTextures")
        self.kernel_bin = _make_kernel(device, tb / "bin.slang", "binGaussians")
        self.kernel_sort = _make_kernel(device, tb / "sort.slang", "sortTile")
        self.kernel_forward = _make_kernel(device, tb / "forward.slang", "forwardRasterize")

        self.buffers = buffers.allocate(
            device=device, module=self.module,
            max_splats=max_splats, width=width, height=height,
        )

        camera_type = self.module.layout.find_type_by_name("CameraData")
        if camera_type is None:
            raise RuntimeError("shader module does not expose type 'CameraData'")
        self._camera_buffer = spy.NDBuffer(device=device, dtype=camera_type, shape=(1,))

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

    def render(self, camera: Camera) -> None:
        self._upload_camera(camera)
        b = self.buffers

        num_tiles = b.num_tiles
        frame_dim = spy.uint2(self.width, self.height)

        # --- clear ---
        self.kernel_clear_tiles.dispatch(
            thread_count=[num_tiles, 1, 1],
            screen_tiles_count=b.screen_tiles_count.storage,
            num_tiles=num_tiles,
        )
        self.kernel_clear_textures.dispatch(
            thread_count=[self.width, self.height, 1],
            output_image=b.output_image,
            output_transmittance=b.output_transmittance,
            gaussian_count_per_pixel=b.gaussian_count_per_pixel,
            frame_dim=frame_dim,
        )

        # --- bin ---
        self.kernel_bin.dispatch(
            thread_count=[self.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            camera=self._camera_buffer.storage,
            projected_2d=b.projected_2d.storage,
            screen_tiles_count=b.screen_tiles_count.storage,
            screen_tiles_indices=b.screen_tiles_indices.storage,
        )

        # --- sort: one workgroup per tile (256 threads each) ---
        self.kernel_sort.dispatch(
            thread_count=[num_tiles * 256, 1, 1],
            projected_2d=b.projected_2d.storage,
            screen_tiles_count=b.screen_tiles_count.storage,
            screen_tiles_indices=b.screen_tiles_indices.storage,
        )

        # --- forward: one thread per pixel, 16x16 workgroups ---
        self.kernel_forward.dispatch(
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

        self.device.wait_for_idle()

    def read_output_image(self) -> np.ndarray:
        return self.buffers.output_image.to_numpy()
