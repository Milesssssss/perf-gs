"""GPU buffer/texture allocation for the trainer.

v0 resources: everything the naive forward (M1) *and* the tile-binned
forward (M3) need. Allocating the tile-binning resources unconditionally
costs nothing when the naive forward is used; the reverse would force a
re-allocation step when M3 activates. Easier to over-allocate once.

Design notes:
- Gaussian buffer is allocated at ``max_splats`` capacity; the active count
  lives in a GPU-resident counter buffer. This is the "oversized + counter"
  pattern from PLAN.md Q13 — costs nothing in v0 but means densification
  won't force a rewrite in v1.
- Shader struct types are looked up via :mod:`slangpy` reflection from the
  loaded module, so Python never duplicates the memory layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import slangpy as spy


TILE_SIZE = 16
MAX_GAUSSIANS_PER_TILE = 1024


@dataclass
class GaussianBuffers:
    # --- always-live gaussian state ---
    gaussians: spy.NDBuffer            # Gaussian3D[max_splats]
    num_gaussians_counter: spy.NDBuffer  # uint[1]

    # --- per-frame scratch / output ---
    projected_2d: spy.NDBuffer         # Projected2DGaussian[max_splats]
    screen_tiles_count: spy.NDBuffer   # uint[max_tiles]
    screen_tiles_indices: spy.NDBuffer  # uint[max_tiles * MAX_GAUSSIANS_PER_TILE]

    output_image: spy.Texture          # rgba32_float, UAV+SRV
    output_transmittance: spy.Texture  # r32_float, UAV+SRV
    gaussian_count_per_pixel: spy.Texture  # r32_uint, UAV+SRV

    # --- sizes ---
    max_splats: int
    width: int
    height: int
    num_tiles_x: int
    num_tiles_y: int

    @property
    def num_tiles(self) -> int:
        return self.num_tiles_x * self.num_tiles_y


def allocate(
    device: spy.Device,
    module: spy.Module,
    max_splats: int,
    width: int,
    height: int,
) -> GaussianBuffers:
    """Allocate all GPU resources needed by the forward pipeline."""
    gaussian_type = module.layout.find_type_by_name("Gaussian3D")
    if gaussian_type is None:
        raise RuntimeError("shader module does not expose type 'Gaussian3D'")

    projected_type = module.layout.find_type_by_name("Projected2DGaussian")
    if projected_type is None:
        raise RuntimeError("shader module does not expose type 'Projected2DGaussian'")

    num_tiles_x = (width + TILE_SIZE - 1) // TILE_SIZE
    num_tiles_y = (height + TILE_SIZE - 1) // TILE_SIZE
    num_tiles = num_tiles_x * num_tiles_y

    gaussians = spy.NDBuffer(device=device, dtype=gaussian_type, shape=(max_splats,))
    num_gaussians_counter = spy.NDBuffer(device=device, dtype="uint", shape=(1,))

    projected_2d = spy.NDBuffer(device=device, dtype=projected_type, shape=(max_splats,))
    screen_tiles_count = spy.NDBuffer(device=device, dtype="uint", shape=(num_tiles,))
    screen_tiles_indices = spy.NDBuffer(
        device=device, dtype="uint", shape=(num_tiles * MAX_GAUSSIANS_PER_TILE,),
    )

    def _tex(fmt: spy.Format, label: str) -> spy.Texture:
        return device.create_texture(
            type=spy.TextureType.texture_2d,
            format=fmt,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=label,
        )

    output_image = _tex(spy.Format.rgba32_float, "output_image")
    output_transmittance = _tex(spy.Format.r32_float, "output_transmittance")
    gaussian_count_per_pixel = _tex(spy.Format.r32_uint, "gaussian_count_per_pixel")

    num_gaussians_counter.copy_from_numpy(np.zeros(1, dtype=np.uint32))

    return GaussianBuffers(
        gaussians=gaussians,
        num_gaussians_counter=num_gaussians_counter,
        projected_2d=projected_2d,
        screen_tiles_count=screen_tiles_count,
        screen_tiles_indices=screen_tiles_indices,
        output_image=output_image,
        output_transmittance=output_transmittance,
        gaussian_count_per_pixel=gaussian_count_per_pixel,
        max_splats=max_splats,
        width=width,
        height=height,
        num_tiles_x=num_tiles_x,
        num_tiles_y=num_tiles_y,
    )


def set_num_gaussians(buffers: GaussianBuffers, n: int) -> None:
    assert 0 <= n <= buffers.max_splats, (
        f"num_gaussians {n} out of range [0, {buffers.max_splats}]"
    )
    buffers.num_gaussians_counter.copy_from_numpy(np.array([n], dtype=np.uint32))


def write_gaussians(
    buffers: GaussianBuffers,
    means: np.ndarray,
    log_scales: np.ndarray,
    rot_quats: np.ndarray,
    opacities: np.ndarray,
    color_logits: np.ndarray,
) -> None:
    """Populate the gaussian buffer from numpy arrays.

    All arrays must have a leading dimension ``N`` with ``N <= max_splats``.
    Quaternion convention is ``(w, x, y, z)``.
    """
    n = means.shape[0]
    assert means.shape == (n, 3), f"means shape {means.shape}"
    assert log_scales.shape == (n, 3), f"log_scales shape {log_scales.shape}"
    assert rot_quats.shape == (n, 4), f"rot_quats shape {rot_quats.shape}"
    assert opacities.shape == (n,), f"opacities shape {opacities.shape}"
    assert color_logits.shape == (n, 3), f"color_logits shape {color_logits.shape}"
    assert n <= buffers.max_splats, f"{n} gaussians exceeds max_splats {buffers.max_splats}"

    cursor = buffers.gaussians.cursor()
    for i in range(n):
        cursor[i].mean = spy.float3(*means[i])
        cursor[i].log_scale = spy.float3(*log_scales[i])
        cursor[i].rot_quat = spy.float4(*rot_quats[i])
        cursor[i].opacity = float(opacities[i])
        cursor[i].color_logit = spy.float3(*color_logits[i])
    cursor.apply()

    set_num_gaussians(buffers, n)
