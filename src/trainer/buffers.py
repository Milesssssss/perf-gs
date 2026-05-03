"""GPU buffer/texture allocation for the trainer.

v1 resources: everything v0 had (forward) plus the backward/loss/Adam
pieces. Allocating the v1 resources unconditionally costs a small fixed
amount of VRAM when only the forward path runs; the reverse would force
a re-allocation when v1 training activates. Easier to over-allocate once.

Design notes:
- Gaussian buffer is allocated at ``max_splats`` capacity; the active count
  lives in a GPU-resident counter buffer (PLAN.md Q13).
- Gradients are SoA int32 fixed-point (PLAN.md Q9 + Q10): one buffer per
  scalar component. ``InterlockedAdd`` on int32 is universally supported
  cross-IHV; ``VK_EXT_shader_atomic_float`` is not.
- Adam state is AoS: one :class:`AdamState` per gaussian.
- Shader struct types are looked up via :mod:`slangpy` reflection from the
  loaded module, so Python never duplicates the memory layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import slangpy as spy


TILE_SIZE = 16
MAX_GAUSSIANS_PER_TILE = 1024


# Names of the 14 scalar grad buffers (SoA). Order matches how the
# backward kernel reads/writes them and how Adam consumes them.
GRAD_SCALAR_COMPONENTS = (
    "mean_x", "mean_y", "mean_z",
    "log_scale_x", "log_scale_y", "log_scale_z",
    "rot_w", "rot_x", "rot_y", "rot_z",
    "opacity",
    "color_r", "color_g", "color_b",
)


@dataclass
class GaussianBuffers:
    # --- always-live gaussian state ---
    gaussians: spy.NDBuffer
    num_gaussians_counter: spy.NDBuffer

    # --- per-frame forward scratch / output ---
    projected_2d: spy.NDBuffer
    screen_tiles_count: spy.NDBuffer
    screen_tiles_indices: spy.NDBuffer

    output_image: spy.Texture          # rgba32_float, UAV+SRV
    output_transmittance: spy.Texture  # r32_float
    gaussian_count_per_pixel: spy.Texture  # r32_uint

    # --- v1 training resources ---
    target_image: spy.Texture          # rgba32_float, SRV (read-only in shaders)
    dL_dC_texture: spy.Texture         # rgba32_float, UAV+SRV
    grad_buffers: dict[str, spy.NDBuffer] = field(default_factory=dict)  # name -> int32[max_splats]
    adam_state: spy.NDBuffer | None = None  # AdamState[max_splats]
    saturation_flag: spy.NDBuffer | None = None  # uint[1]
    training_step: spy.NDBuffer | None = None    # uint[1] (Adam bias correction)

    # --- sizes ---
    max_splats: int = 0
    width: int = 0
    height: int = 0
    num_tiles_x: int = 0
    num_tiles_y: int = 0

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
    """Allocate all GPU resources needed by forward + backward + Adam."""
    gaussian_type = module.layout.find_type_by_name("Gaussian3D")
    if gaussian_type is None:
        raise RuntimeError("shader module does not expose type 'Gaussian3D'")
    projected_type = module.layout.find_type_by_name("Projected2DGaussian")
    if projected_type is None:
        raise RuntimeError("shader module does not expose type 'Projected2DGaussian'")
    adam_type = module.layout.find_type_by_name("AdamState")
    if adam_type is None:
        raise RuntimeError("shader module does not expose type 'AdamState'")

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

    def _tex(fmt: spy.Format, label: str, usage: spy.TextureUsage | None = None) -> spy.Texture:
        return device.create_texture(
            type=spy.TextureType.texture_2d,
            format=fmt,
            width=width,
            height=height,
            usage=usage if usage is not None else (spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access),
            label=label,
        )

    output_image = _tex(spy.Format.rgba32_float, "output_image")
    output_transmittance = _tex(spy.Format.r32_float, "output_transmittance")
    gaussian_count_per_pixel = _tex(spy.Format.r32_uint, "gaussian_count_per_pixel")

    target_image = _tex(
        spy.Format.rgba32_float, "target_image",
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    dL_dC_texture = _tex(spy.Format.rgba32_float, "dL_dC")

    grad_buffers: dict[str, spy.NDBuffer] = {
        name: spy.NDBuffer(device=device, dtype="int", shape=(max_splats,))
        for name in GRAD_SCALAR_COMPONENTS
    }

    adam_state = spy.NDBuffer(device=device, dtype=adam_type, shape=(max_splats,))

    saturation_flag = spy.NDBuffer(device=device, dtype="uint", shape=(1,))
    training_step = spy.NDBuffer(device=device, dtype="uint", shape=(1,))

    num_gaussians_counter.copy_from_numpy(np.zeros(1, dtype=np.uint32))
    saturation_flag.copy_from_numpy(np.zeros(1, dtype=np.uint32))
    training_step.copy_from_numpy(np.zeros(1, dtype=np.uint32))

    return GaussianBuffers(
        gaussians=gaussians,
        num_gaussians_counter=num_gaussians_counter,
        projected_2d=projected_2d,
        screen_tiles_count=screen_tiles_count,
        screen_tiles_indices=screen_tiles_indices,
        output_image=output_image,
        output_transmittance=output_transmittance,
        gaussian_count_per_pixel=gaussian_count_per_pixel,
        target_image=target_image,
        dL_dC_texture=dL_dC_texture,
        grad_buffers=grad_buffers,
        adam_state=adam_state,
        saturation_flag=saturation_flag,
        training_step=training_step,
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


def zero_adam_state(buffers: GaussianBuffers) -> None:
    """Reset all Adam first/second-moment state to zero."""
    assert buffers.adam_state is not None
    n = buffers.max_splats
    cursor = buffers.adam_state.cursor()
    z3 = spy.float3(0.0, 0.0, 0.0)
    z4 = spy.float4(0.0, 0.0, 0.0, 0.0)
    for i in range(n):
        cursor[i].mean_m = z3
        cursor[i].mean_v = z3
        cursor[i].log_scale_m = z3
        cursor[i].log_scale_v = z3
        cursor[i].rot_m = z4
        cursor[i].rot_v = z4
        cursor[i].opacity_m = 0.0
        cursor[i].opacity_v = 0.0
        cursor[i].color_m = z3
        cursor[i].color_v = z3
    cursor.apply()
