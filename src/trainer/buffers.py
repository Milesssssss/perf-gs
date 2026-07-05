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

# Default cap on the total number of (tile_id, depth, gaussian_idx) pairs
# emitted by bin_emit before the global radix sort. A single gaussian
# covering K tiles contributes K pairs. Computed as max_splats * 16
# (average tile coverage with headroom) in allocate(); this constant
# is only used when no explicit value is provided.
DEFAULT_MAX_SORT_PAIRS = 8_000_000


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

    # v2.x uncapped-binning pipeline (replaces the legacy per-tile array).
    # See shaders/rasterize/tile_binned/bin_count.slang and bin_emit.slang.
    tile_count_per_gaussian: spy.NDBuffer   # uint[max_splats]
    tile_offset_per_gaussian: spy.NDBuffer  # uint[max_splats] — scan output
    # Ping-pong pair buffers for radix sort.
    sort_keys_lo_a: spy.NDBuffer            # uint[max_sort_pairs]
    sort_keys_hi_a: spy.NDBuffer
    sort_values_a: spy.NDBuffer
    sort_keys_lo_b: spy.NDBuffer
    sort_keys_hi_b: spy.NDBuffer
    sort_values_b: spy.NDBuffer
    # DeviceRadixSort scratch (ported from GPUSorting).
    # radix_global_hist   : uint[256 * 8], per-pass device digit offsets;
    #                       one 256-region per pass, cleared once at start.
    # radix_pass_hist     : uint[256 * max_blocks], per-(digit, partition)
    #                       counts, transposed layout with fixed stride =
    #                       max_blocks so indexing is uniform across
    #                       upsweep / scan / downsweep.
    # radix_dispatch_args : uint[3], indirect (groups_x, 1, 1) written by
    #                       prepareDispatchArgs so upsweep/downsweep scale
    #                       with active_blocks instead of max_blocks.
    radix_global_hist: spy.NDBuffer
    radix_pass_hist: spy.NDBuffer
    radix_dispatch_args: spy.NDBuffer
    # Per-tile sorted entry ranges.
    tile_start: spy.NDBuffer                # uint[num_tiles]
    tile_end: spy.NDBuffer                  # uint[num_tiles]

    output_image: spy.Texture          # rgba32_float, UAV+SRV
    output_transmittance: spy.Texture  # r32_float
    gaussian_count_per_pixel: spy.Texture  # r32_uint

    # --- v1 training resources ---
    target_image: spy.Texture          # rgba32_float, SRV (read-only in shaders)
    dL_dC_texture: spy.Texture         # rgba32_float, UAV+SRV
    # SSIM two-pass back-projection maps (m1/m2/m3), only used when lambda_ssim>0.
    ssim_m1: spy.Texture | None = None  # rgba32_float, WxH
    ssim_m2: spy.Texture | None = None  # rgba32_float, WxH
    ssim_m3: spy.Texture | None = None  # rgba32_float, WxH
    grad_buffers: dict[str, spy.NDBuffer] = field(default_factory=dict)  # name -> int32[max_splats]
    adam_state: spy.NDBuffer | None = None  # AdamState[max_splats]
    saturation_flag: spy.NDBuffer | None = None  # uint[1]
    training_step: spy.NDBuffer | None = None    # uint[1] (Adam bias correction)

    # --- v2 densification tracking (accumulated over a window) ---
    # grad_mean_2d_mag_accum   : fixed-point sum of per-iter norm-of-sum
    #                            2D gradients.
    # visibility_count_accum   : iter count (not pixel count) — matches
    #                            gsplat's `state["count"]`.
    # grad_2d_{x,y}_iter       : per-iter vector accumulators; cleared by
    #                            accumulate_iter at each step end.
    grad_mean_2d_mag_accum: spy.NDBuffer | None = None  # int32[max_splats], fixed-point
    visibility_count_accum: spy.NDBuffer | None = None  # uint[max_splats]
    grad_2d_x_iter: spy.NDBuffer | None = None          # int32[max_splats], fixed-point
    grad_2d_y_iter: spy.NDBuffer | None = None          # int32[max_splats], fixed-point

    # --- FastGS importance scoring ---
    importance_score_accum: spy.NDBuffer | None = None  # uint[max_splats]
    growth_budget: spy.NDBuffer | None = None           # int[1]
    prune_budget: spy.NDBuffer | None = None            # int[1]
    error_mask: spy.Texture | None = None               # r32_uint, WxH
    l1_map: spy.Texture | None = None                   # r32_float, WxH
    l1_minmax: spy.NDBuffer | None = None               # uint[2]

    # --- v2 densification scratch + pipeline state ---
    densify_score: spy.NDBuffer | None = None       # float[max_splats]
    gaussians_scratch: spy.NDBuffer | None = None   # Gaussian3D[max_splats]
    adam_state_scratch: spy.NDBuffer | None = None  # AdamState[max_splats]
    action_buffer: spy.NDBuffer | None = None       # uint[max_splats]
    valid_mask_buffer: spy.NDBuffer | None = None   # uint[max_splats]
    scan_buffer: spy.NDBuffer | None = None         # uint[max_splats]
    scan_block_sums: spy.NDBuffer | None = None     # uint[max_blocks]
    scan_block_sums_l2: spy.NDBuffer | None = None  # uint[max_l2_blocks]
    scan_total: spy.NDBuffer | None = None          # uint[1]

    # --- sizes ---
    max_splats: int = 0
    max_sort_pairs: int = 0
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
    max_sort_pairs: int = DEFAULT_MAX_SORT_PAIRS,
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

    # v2.x uncapped-binning resources.
    tile_count_per_gaussian = spy.NDBuffer(device=device, dtype="uint", shape=(max_splats,))
    tile_offset_per_gaussian = spy.NDBuffer(device=device, dtype="uint", shape=(max_splats,))
    sort_keys_lo_a = spy.NDBuffer(device=device, dtype="uint", shape=(max_sort_pairs,))
    sort_keys_hi_a = spy.NDBuffer(device=device, dtype="uint", shape=(max_sort_pairs,))
    sort_values_a = spy.NDBuffer(device=device, dtype="uint", shape=(max_sort_pairs,))
    sort_keys_lo_b = spy.NDBuffer(device=device, dtype="uint", shape=(max_sort_pairs,))
    sort_keys_hi_b = spy.NDBuffer(device=device, dtype="uint", shape=(max_sort_pairs,))
    sort_values_b = spy.NDBuffer(device=device, dtype="uint", shape=(max_sort_pairs,))
    # DeviceRadixSort scratch. _RADIX_PART_SIZE = 2048 must match
    # device_radix_sort.slang's PART_SIZE.
    _RADIX_PART_SIZE = 2048
    _RADIX_GLOBAL_HIST_SIZE = 256 * 8  # one 256-region per pass, 8 passes
    max_blocks = (max_sort_pairs + _RADIX_PART_SIZE - 1) // _RADIX_PART_SIZE
    radix_global_hist = spy.NDBuffer(
        device=device, dtype="uint", shape=(_RADIX_GLOBAL_HIST_SIZE,),
    )
    radix_pass_hist = spy.NDBuffer(
        device=device, dtype="uint", shape=(256 * max_blocks,),
    )
    # Indirect args need BufferUsage.indirect_argument in addition to
    # shader_resource | unordered_access (default) so the kernel can write
    # to it AND the driver can read it as dispatch args.
    radix_dispatch_args = spy.NDBuffer(
        device=device, dtype="uint", shape=(3,),
        usage=(spy.BufferUsage.shader_resource
               | spy.BufferUsage.unordered_access
               | spy.BufferUsage.indirect_argument),
    )
    tile_start = spy.NDBuffer(device=device, dtype="uint", shape=(num_tiles,))
    tile_end = spy.NDBuffer(device=device, dtype="uint", shape=(num_tiles,))

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
    ssim_m1 = _tex(spy.Format.rgba32_float, "ssim_m1")
    ssim_m2 = _tex(spy.Format.rgba32_float, "ssim_m2")
    ssim_m3 = _tex(spy.Format.rgba32_float, "ssim_m3")

    grad_buffers: dict[str, spy.NDBuffer] = {
        name: spy.NDBuffer(device=device, dtype="int64_t", shape=(max_splats,))
        for name in GRAD_SCALAR_COMPONENTS
    }

    adam_state = spy.NDBuffer(device=device, dtype=adam_type, shape=(max_splats,))

    saturation_flag = spy.NDBuffer(device=device, dtype="uint", shape=(1,))
    training_step = spy.NDBuffer(device=device, dtype="uint", shape=(1,))

    grad_mean_2d_mag_accum = spy.NDBuffer(device=device, dtype="int64_t", shape=(max_splats,))
    visibility_count_accum = spy.NDBuffer(device=device, dtype="uint", shape=(max_splats,))
    grad_2d_x_iter = spy.NDBuffer(device=device, dtype="int64_t", shape=(max_splats,))
    grad_2d_y_iter = spy.NDBuffer(device=device, dtype="int64_t", shape=(max_splats,))

    # FastGS importance scoring
    importance_score_accum = spy.NDBuffer(device=device, dtype="uint", shape=(max_splats,))
    growth_budget = spy.NDBuffer(device=device, dtype="int", shape=(1,))
    prune_budget = spy.NDBuffer(device=device, dtype="int", shape=(1,))
    error_mask = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.r32_uint,
        width=width, height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="error_mask",
    )
    l1_map = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.r32_float,
        width=width, height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="l1_map",
    )
    l1_minmax = spy.NDBuffer(device=device, dtype="uint", shape=(2,))

    # v2 densification scratch
    densify_score = spy.NDBuffer(device=device, dtype="float", shape=(max_splats,))
    gaussians_scratch = spy.NDBuffer(device=device, dtype=gaussian_type, shape=(max_splats,))
    adam_state_scratch = spy.NDBuffer(device=device, dtype=adam_type, shape=(max_splats,))
    action_buffer = spy.NDBuffer(device=device, dtype="uint", shape=(max_splats,))
    valid_mask_buffer = spy.NDBuffer(device=device, dtype="uint", shape=(max_splats,))
    scan_buffer = spy.NDBuffer(device=device, dtype="uint", shape=(max_splats,))
    # Scan primitive uses 1024 threads/block; number of blocks is ceil(N/1024).
    # Allocate upper bound = ceil(max_splats / 1024).
    num_l1_blocks = max(1, (max_splats + 1023) // 1024)
    scan_block_sums = spy.NDBuffer(
        device=device, dtype="uint",
        shape=(num_l1_blocks,),
    )
    # Level-2 block sums for 3-level scan (needed when max_splats > 1M).
    num_l2_blocks = max(1, (num_l1_blocks + 1023) // 1024)
    scan_block_sums_l2 = spy.NDBuffer(
        device=device, dtype="uint",
        shape=(num_l2_blocks,),
    )
    scan_total = spy.NDBuffer(device=device, dtype="uint", shape=(1,))

    num_gaussians_counter.copy_from_numpy(np.zeros(1, dtype=np.uint32))
    saturation_flag.copy_from_numpy(np.zeros(1, dtype=np.uint32))
    training_step.copy_from_numpy(np.zeros(1, dtype=np.uint32))

    # this is nuts
    return GaussianBuffers(
        gaussians=gaussians,
        num_gaussians_counter=num_gaussians_counter,
        projected_2d=projected_2d,
        tile_count_per_gaussian=tile_count_per_gaussian,
        tile_offset_per_gaussian=tile_offset_per_gaussian,
        sort_keys_lo_a=sort_keys_lo_a,
        sort_keys_hi_a=sort_keys_hi_a,
        sort_values_a=sort_values_a,
        sort_keys_lo_b=sort_keys_lo_b,
        sort_keys_hi_b=sort_keys_hi_b,
        sort_values_b=sort_values_b,
        radix_global_hist=radix_global_hist,
        radix_pass_hist=radix_pass_hist,
        radix_dispatch_args=radix_dispatch_args,
        tile_start=tile_start,
        tile_end=tile_end,
        output_image=output_image,
        output_transmittance=output_transmittance,
        gaussian_count_per_pixel=gaussian_count_per_pixel,
        target_image=target_image,
        dL_dC_texture=dL_dC_texture,
        ssim_m1=ssim_m1,
        ssim_m2=ssim_m2,
        ssim_m3=ssim_m3,
        grad_buffers=grad_buffers,
        adam_state=adam_state,
        saturation_flag=saturation_flag,
        training_step=training_step,
        grad_mean_2d_mag_accum=grad_mean_2d_mag_accum,
        visibility_count_accum=visibility_count_accum,
        grad_2d_x_iter=grad_2d_x_iter,
        grad_2d_y_iter=grad_2d_y_iter,
        importance_score_accum=importance_score_accum,
        growth_budget=growth_budget,
        prune_budget=prune_budget,
        error_mask=error_mask,
        l1_map=l1_map,
        l1_minmax=l1_minmax,
        densify_score=densify_score,
        gaussians_scratch=gaussians_scratch,
        adam_state_scratch=adam_state_scratch,
        action_buffer=action_buffer,
        valid_mask_buffer=valid_mask_buffer,
        scan_buffer=scan_buffer,
        scan_block_sums=scan_block_sums,
        scan_block_sums_l2=scan_block_sums_l2,
        scan_total=scan_total,
        max_splats=max_splats,
        max_sort_pairs=max_sort_pairs,
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
    """``opacities`` is a probability in (0, 1); we convert to the
    pre-sigmoid logit the shader expects."""
    n = means.shape[0]
    assert means.shape == (n, 3), f"means shape {means.shape}"
    assert log_scales.shape == (n, 3), f"log_scales shape {log_scales.shape}"
    assert rot_quats.shape == (n, 4), f"rot_quats shape {rot_quats.shape}"
    assert opacities.shape == (n,), f"opacities shape {opacities.shape}"
    assert color_logits.shape == (n, 3), f"color_logits shape {color_logits.shape}"
    assert n <= buffers.max_splats, f"{n} gaussians exceeds max_splats {buffers.max_splats}"

    # Convert probability → logit once on CPU. 1e-5 floor keeps log finite
    # when a caller passes exactly 0 or 1.
    p = np.clip(opacities, 1e-5, 1.0 - 1e-5).astype(np.float32)
    opacity_logits = np.log(p / (1.0 - p))

    # Gaussian3D is 14 contiguous float32s with no padding:
    #   mean(3), log_scale(3), rot_quat(4), opacity(1), color_logit(3).
    # Build the whole block with vectorized numpy and upload in one
    # copy_from_numpy — the per-element cursor loop below built ~5*n
    # temporary spy.float3/float4 objects (≈90ms per 5k gaussians, seconds
    # after densification grows the count into the hundreds of thousands).
    block = np.empty((n, 14), dtype=np.float32)
    block[:, 0:3] = means
    block[:, 3:6] = log_scales
    block[:, 6:10] = rot_quats
    block[:, 10] = opacity_logits
    block[:, 11:14] = color_logits
    buffers.gaussians.copy_from_numpy(np.ascontiguousarray(block.reshape(-1)))

    set_num_gaussians(buffers, n)


def write_gaussians_raw(buffers: GaussianBuffers, floats: np.ndarray, n: int) -> None:
    """Upload a snapshot already in the packed 14-float Gaussian3D layout
    (mean3, log_scale3, rot_quat4, opacity1, sh0_3) straight into the GPU
    buffer. Used by the live viewer, which receives snapshots from the
    trainer via the shared-memory splat stream — no per-field unpacking.

    ``floats`` must hold at least ``n * 14`` float32 values."""
    assert n <= buffers.max_splats, f"{n} gaussians exceeds max_splats {buffers.max_splats}"
    n_floats = n * 14
    block = np.ascontiguousarray(floats[:n_floats].astype(np.float32, copy=False))
    if n > 0:
        buffers.gaussians.copy_from_numpy(block)
    set_num_gaussians(buffers, n)


def zero_adam_state(buffers: GaussianBuffers) -> None:
    """Reset all Adam first/second-moment state to zero.

    The whole AdamState buffer is contiguous POD, so a single zero-filled
    copy_from_numpy clears it regardless of the internal field layout. The
    old per-element cursor loop ran over all max_splats (~820ms per 50k,
    i.e. ~80s at a 5M cap) on every set_gaussians_from_numpy call.
    """
    assert buffers.adam_state is not None
    n_floats = buffers.adam_state.to_numpy().view(np.float32).size
    buffers.adam_state.copy_from_numpy(np.zeros(n_floats, dtype=np.float32))
