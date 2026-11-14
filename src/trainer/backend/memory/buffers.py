# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU buffer/texture allocation for the trainer.

Trainer allocation includes forward, backward, optimizer, and densification
capabilities. Viewer allocation intentionally omits training-only capabilities.

Design notes:
- Gaussian buffer is allocated at ``max_splats`` capacity; the active count
  lives in a GPU-resident counter buffer (PLAN.md Q13).
- Gradients are SoA int32 fixed-point (PLAN.md Q9 + Q10): one buffer per
  scalar component. ``InterlockedAdd`` on int32 is universally supported
  cross-IHV; ``VK_EXT_shader_atomic_float`` is not.
- Adam state is AoS: one :class:`AdamState` per gaussian.
- Primitive, Adam, scan, densification, and eta resources have explicit
    capability owners. Flat aliases remain temporarily for unmigrated paths.
- Shader struct types are looked up via :mod:`slangpy` reflection from the
  loaded module, so Python never duplicates the memory layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import slangpy as spy
from slangpy.core.native import NativeNDBuffer, NativeNDBufferDesc, Shape
from slangpy.reflection.lookup import resolve_element_type, resolve_program_layout

# Authoritative host<->shader layout + execution constants (single source of
# truth; the same schema generates the Slang side). See trainer/contracts.py.
from trainer.contracts.codegen import (
    CLUSTER_SIZE,
    ETA_ACCUM_COMPONENTS,
    GAUSSIAN_FLOAT_COUNT,
    GRAD_2D_SEED_COMPONENTS,
    GRAD_SCALAR_COMPONENTS,
    RADIX_GLOBAL_HIST_SIZE,
    RADIX_PART_SIZE,
    RASTER_BUCKET_SIZE,
    SCAN_BLOCK_SIZE,
    TILE_AREA,
    TILE_HEIGHT,
    TILE_WIDTH,
)


# --- Shader-layout + execution constants ---
# The AUTHORITATIVE source is trainer.contracts (imported at the top of this
# module): the executable host<->shader schema that derives these Python values
# AND generates the Slang side, so the layout lives in exactly one place and
# cannot silently drift. See trainer/contracts.py + tests/test_contracts.py.

# Default cap on the total number of (tile_id, depth, gaussian_idx) pairs
# emitted by bin_emit before the global radix sort. A single gaussian
# covering K tiles contributes K pairs. Computed as max_splats * 16
# (average tile coverage with headroom) in allocate(); this constant
# is only used when no explicit value is provided.
DEFAULT_MAX_SORT_PAIRS = 12_000_000


def _ceil_div(a: int, b: int) -> int:
    """Ceiling division ``ceil(a / b)`` for non-negative ints."""
    return (a + b - 1) // b


_DEFAULT_BUFFER_USAGE = (
    spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access)


def _make_labeled_ndbuffer(device, dtype, shape, label, usage):
    """Create an NDBuffer whose underlying VkBuffer carries a debug ``label``.

    ``spy.NDBuffer`` has no label parameter and its VkBuffer is created
    internally, so we reproduce its construction (resolve the element type,
    build the same NativeNDBufferDesc) but pass the storage buffer in
    pre-created WITH a label. Vulkan records a resource's ``VK_EXT_debug_utils``
    object name only AT CREATION — mutating ``storage.desc.label`` afterwards
    does not rename the live VkBuffer — so the label must be set here for
    Nsight's Memory / Object Browser views to show it (textures already do this
    via ``create_texture(label=)``). No extra allocation vs. a plain NDBuffer:
    exactly one VkBuffer of the same size is created.

    Best-effort: if any of slangpy's internal reflection/native API shifts
    across versions, fall back to an unlabeled ``spy.NDBuffer`` so allocation
    can never break just because naming failed.
    """
    try:
        program_layout = resolve_program_layout(device, dtype, None)
        element_type = resolve_element_type(program_layout, dtype)
        element_count = 1
        for dim in shape:
            element_count *= dim
        shp = Shape(tuple(shape))
        desc = NativeNDBufferDesc()
        desc.usage = usage
        desc.memory_type = spy.MemoryType.device_local
        desc.shape = shp
        desc.strides = shp.calc_contiguous_strides()
        desc.dtype = element_type
        desc.element_layout = element_type.buffer_layout.reflection
        size = element_count * element_type.buffer_layout.reflection.stride
        buffer = device.create_buffer(
            size=size, usage=usage,
            memory_type=spy.MemoryType.device_local, label=label)
        return NativeNDBuffer(device, desc, buffer)
    except Exception:
        return spy.NDBuffer(device=device, dtype=dtype, shape=shape, usage=usage)


@dataclass
class PrimitiveStorage:
    values: spy.NDBuffer
    active_count: spy.NDBuffer
    filter_3d: spy.NDBuffer


@dataclass
class AdamResources:
    state: spy.NDBuffer


@dataclass
class ScratchArena:
    """One raw scratch buffer shared across all compaction / Morton re-sort
    destinations (gaussians, adam_state, features_rest, adam_rest).

    Sized to the largest whole-resource scatter (gaussians, main Adam, or SH
    features). SH Adam's independent SoA planes are scattered in batches that
    fit this size instead of growing the arena to the whole moment buffer.
    Only ONE resource/batch occupies it at a time: scatter -> arena -> live
    buffer, then reuse. ``buffer``
    is a uint NDBuffer; ``storage`` re-types cleanly to any element type at bind
    (a raw VkBuffer carries no element type — the shader param supplies it),
    verified for Gaussian3D / AdamState / uint. ``capacity_bytes`` is the usable
    byte span (>= each routed resource or plane batch). Active-count commits use
    GPU indirect copies, including each live SoA component prefix; ``copy_back``
    retains the explicit byte-copy path for callers without a GPU count."""
    buffer: spy.NDBuffer
    capacity_bytes: int
    copy_dispatch_args: spy.NDBuffer

    @property
    def storage(self):
        """Bind target for a scatter kernel's ``*_out`` parameter."""
        return self.buffer.storage

    def copy_back(self, encoder, dst: spy.NDBuffer, nbytes: int, *,
                  dst_offset_bytes: int = 0) -> None:
        """Copy the just-scattered ``nbytes`` from the arena into the live
        buffer ``dst`` (byte-for-byte; commits the compaction/re-sort for that
        resource). sgl auto-inserts the scatter->copy dependency barrier, and
        the following scatter->arena reuse is likewise auto-synchronized."""
        assert nbytes <= self.capacity_bytes, (
            f"scratch arena too small: need {nbytes}, have {self.capacity_bytes}")
        encoder.copy_buffer(dst.storage, dst_offset_bytes, self.buffer.storage, 0, nbytes)


@dataclass
class ScanResources:
    work: spy.NDBuffer | None
    block_sums: spy.NDBuffer
    block_sums_l2: spy.NDBuffer
    total: spy.NDBuffer


@dataclass
class DensificationResources:
    grad_mean_2d_mag: spy.NDBuffer
    visibility_count: spy.NDBuffer
    blend_weight: spy.NDBuffer
    # Shared scratch arena for compaction / Morton re-sort. One raw buffer sized
    # for the largest whole-resource scatter (normally main Adam). SH-rest Adam
    # uses bounded plane batches, so it does not enlarge the arena. Re-typed per
    # resource at bind time; scatter -> copy back -> reuse, without pointer swaps.
    scratch_arena: "ScratchArena"
    actions: spy.NDBuffer
    valid_mask: spy.NDBuffer
    pre_count: spy.NDBuffer
    post_count: spy.NDBuffer
    dispatch_args: spy.NDBuffer


@dataclass
class EtaResources:
    components: dict[str, spy.NDBuffer]
    n_total: spy.NDBuffer
    scratch_components: dict[str, spy.NDBuffer]
    scratch_n_total: spy.NDBuffer

    def scatter_kwargs(self) -> dict[str, spy.NDBuffer]:
        primary = {"n_total": self.n_total, **self.components}
        scratch = {"n_total": self.scratch_n_total, **self.scratch_components}
        kwargs = {f"eta_{name}_in": buffer.storage for name, buffer in primary.items()}
        kwargs.update(
            {f"eta_{name}_out": buffer.storage for name, buffer in scratch.items()})
        return kwargs

    def swap(self) -> None:
        self.n_total, self.scratch_n_total = self.scratch_n_total, self.n_total
        self.components, self.scratch_components = \
            self.scratch_components, self.components


@dataclass
class SphericalHarmonicsResources:
    """Optional view-dependent SH 'rest' coefficients (bands 1..sh_degree).

    Stored SEPARATELY from Gaussian3D so the default SH0 path allocates nothing.
    Features are fp16 pairs in ``features_rest[gid * feat_uints + word]``.
    Adam moments are SoA planes of max_splats words, ``moment_planes`` in all:
    float32 m and v planes per coefficient, or one bf16 (m, v) plane under
    ``ADAM_MOMENTS_BF16``. Compaction and Morton re-sort carry all planes
    through bounded batches of the shared scratch arena.
    """

    rest_floats: int
    feat_uints: int
    features_rest: spy.NDBuffer
    features_rest_grad: spy.NDBuffer
    adam_rest: spy.NDBuffer
    moment_planes: int
    # features_rest / adam_rest are compacted + Morton-permuted via the shared
    # ScratchArena (single-output scatter -> arena -> copy back), so no dedicated
    # scratch buffers here anymore. features_rest_grad is per-iter gradient
    # accumulation, untouched by compaction.


@dataclass
class GaussianBuffers:
    """Compatibility facade plus explicit resource capability owners.

    Trainer allocation supplies every capability. Viewer allocation supplies
    primitive + scan capabilities and forward resources only; training-only
    access fails explicitly. Flat fields remain while forward/backward bundles
    are migrated to injection.
    """

    # --- sizes (the only constructor inputs) ---
    max_splats: int = 0
    max_sort_pairs: int = 0
    width: int = 0
    height: int = 0
    num_tiles_x: int = 0
    num_tiles_y: int = 0

    # Authoritative capability owners. Viewer allocation intentionally omits
    # training-only capabilities (adam/densification/eta).
    primitive: PrimitiveStorage = field(init=False)
    scan: ScanResources = field(init=False)
    adam: AdamResources | None = field(init=False, default=None)
    densification: DensificationResources | None = field(init=False, default=None)
    eta: EtaResources | None = field(init=False, default=None)
    # Optional view-dependent SH rest coefficients; None => SH0-only (default).
    sh: "SphericalHarmonicsResources | None" = field(init=False, default=None)

    # --- per-frame forward scratch / output ---
    projected_2d: spy.NDBuffer = field(init=False)
    # Compact blend-backward gather record; packed by bin_count.
    blend_data: spy.NDBuffer = field(init=False)

    # v2.x uncapped-binning pipeline (replaces the legacy per-tile array).
    # See shaders/rasterize/tile_binned/bin_count.slang and bin_emit.slang.
    tile_count_per_gaussian: spy.NDBuffer = field(init=False)   # uint[max_splats]
    tile_offset_per_gaussian: spy.NDBuffer = field(init=False)  # uint[max_splats] — scan output
    # Ping-pong pair buffers for radix sort.
    sort_keys_lo_a: spy.NDBuffer = field(init=False)            # uint[max_sort_pairs]
    sort_keys_hi_a: spy.NDBuffer = field(init=False)
    sort_values_a: spy.NDBuffer = field(init=False)
    sort_keys_lo_b: spy.NDBuffer = field(init=False)
    sort_keys_hi_b: spy.NDBuffer = field(init=False)
    sort_values_b: spy.NDBuffer = field(init=False)
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
    radix_global_hist: spy.NDBuffer = field(init=False)
    radix_pass_hist: spy.NDBuffer = field(init=False)
    radix_dispatch_args: spy.NDBuffer = field(init=False)
    sort_stats: spy.NDBuffer = field(init=False)  # uint[2]: peak requested pairs, sorts that dropped pairs
    # Per-tile sorted entry ranges.
    tile_start: spy.NDBuffer = field(init=False)                # uint[num_tiles]
    tile_end: spy.NDBuffer = field(init=False)                  # uint[num_tiles]

    # Ragged (CSR) forward-checkpoint state for the transposed backward.
    # bucket_count       : uint[num_tiles], ceil(tile_len / P) per tile
    # bucket_offsets     : uint[num_tiles], exclusive scan of bucket_count
    #                      (base bucket index of each tile in the flat buffer)
    # bucket_offset_total: uint[1], total buckets used this frame (scan total)
    # raster_checkpoints : uint2[max_buckets * TILE_AREA], (T,R,G,B) fp16 per pixel
    #                      per bucket. max_buckets = ceil(max_sort_pairs/P)
    #                      + num_tiles (tight upper bound on sum of per-tile
    #                      ceil(len/P), no readback needed).
    bucket_count: spy.NDBuffer = field(init=False)
    bucket_offsets: spy.NDBuffer = field(init=False)
    bucket_offset_total: spy.NDBuffer = field(init=False)
    raster_checkpoints: spy.NDBuffer = field(init=False)

    # Live-bucket compaction for the flat (warp-per-bucket) backward
    # (use_flat_bucket_backward). live_bucket_count[t] = ceil(max_processed_t/P)
    # (<= bucket_count[t]); its exclusive scan is live_bucket_offsets with total
    # live_bucket_total[0]. live_buckets[slot] = uint2(tile, local_bucket) for
    # every live bucket, so the flat kernel needs no tile search and skips
    # saturated buckets. bucket_dispatch_args = indirect (groups_x,1,1).
    live_bucket_count: spy.NDBuffer = field(init=False)
    live_bucket_offsets: spy.NDBuffer = field(init=False)
    live_bucket_total: spy.NDBuffer = field(init=False)
    live_buckets: spy.NDBuffer = field(init=False)
    bucket_dispatch_args: spy.NDBuffer = field(init=False)

    output_image: spy.Texture = field(init=False)          # rgba32_float, UAV+SRV
                                                            # (.a = 1-T; final transmittance
                                                            # recovered as T = 1 - .a in backward)
    gaussian_count_per_pixel: spy.Texture = field(init=False)  # r32_uint

    # --- v1 training resources ---
    # rgba8_unorm: the GT is an 8-bit sRGB PNG (loaded as uint8/255), so 32-bit
    # float storage was pure waste. Shaders read it as RWTexture2D<float4> via
    # [] and get the same normalized [0,1] sRGB values (unorm auto-divides by
    # 255); the loss compares in sRGB space directly, so UNORM (NOT _srgb) is
    # the lossless match. 4x smaller than rgba32_float.
    target_image: spy.Texture = field(init=False)          # rgba8_unorm, sRGB [0,1] in .rgb
    dL_dC_texture: spy.Texture = field(init=False)         # rgba32_float, UAV+SRV
    # SSIM two-pass back-projection maps (m1/m2/m3), only used when lambda_ssim>0.
    ssim_m1: spy.Texture = field(init=False)  # rgba32_float, WxH
    ssim_m2: spy.Texture = field(init=False)  # rgba32_float, WxH
    ssim_m3: spy.Texture = field(init=False)  # rgba32_float, WxH
    grad_buffers: dict[str, spy.NDBuffer] = field(init=False)  # name -> int64[max_splats]
    # Per-gaussian summed 2D grad seeds (mean2d/cov2d) bridging the split
    # backward: blend kernel accumulates, projection kernel consumes.
    grad_2d_seed_buffers: dict[str, spy.NDBuffer] = field(init=False)
    training_step: spy.NDBuffer = field(init=False)    # uint[1] (Adam bias correction)

    # --- v2 densification tracking (accumulated over a window) ---
    # grad_mean_2d_mag_accum   : fixed-point sum of per-iter norm-of-sum
    #                            2D gradients.
    # visibility_count_accum   : iter count (not pixel count) — matches
    #                            gsplat's `state["count"]`.
    # grad_2d_{x,y}_iter       : per-iter vector accumulators; cleared by
    #                            accumulate_iter at each step end.
    grad_2d_x_iter: spy.NDBuffer = field(init=False)          # int64[max_splats], fixed-point
    grad_2d_y_iter: spy.NDBuffer = field(init=False)          # int64[max_splats], fixed-point

    # --- densify_simple wait-free dispatch state ---
    # Snapshots of num_gaussians_counter[0] taken via snapshotCount so that
    # splitAndClone/buildPruneMask/scatterCompact can read the counts they
    # need as GPU-resident buffers instead of via CPU readback.
    # DispatchIndirect args (groups_x, 1, 1) for the pre-count-sized
    # splitAndClone dispatch. Also requires BufferUsage.indirect_argument.
    # DispatchIndirect args for the Adam step. Populated every iter from
    # num_gaussians_counter via prepareDensifyArgs (reused shader). Sized so
    # groups_x = ceil(count/256); dispatching indirect lets Adam launch only
    # the workgroups it needs (~1-3M gaussians vs max_splats=10M = 4x fewer
    # workgroups). Also requires BufferUsage.indirect_argument.
    adam_dispatch_args: spy.NDBuffer = field(init=False)     # uint[3]
    # Mip recomputation launches one workgroup per LIVE gaussian, tiled over
    # two dispatch dimensions when the live population exceeds 65,535.
    mip_filter_dispatch_args: spy.NDBuffer = field(init=False)  # uint[6]: compute + tail clear
    mip_filter_previous_count: spy.NDBuffer = field(init=False)  # uint[2]: live count + clear end

    # --- structure-aware densification (SAD-GS, arXiv 2604.28016) ---
    # Per-axis multiview frequency-violation accumulators, name -> uint[max_splats].
    # Keys are ETA_ACCUM_COMPONENTS. Only allocated/used on the structure path.
    # Scratch doubles for the eta accumulators, so the SAD-GS window statistics
    # can be carried through the slot permutations of a filler densify
    # (scatterCompactEta) and a Morton re-sort (mortonScatterEta) instead of
    # being zeroed. Same swap idiom as gaussians_scratch / adam_state_scratch.
    # Per-axis grid split factor n_k (uint[max_splats]) written by the GPU
    # decision (decide_actions_structure.slang) and consumed by
    # split_anisotropic.slang.
    split_factor_x: spy.NDBuffer = field(init=False)
    split_factor_y: spy.NDBuffer = field(init=False)
    split_factor_z: spy.NDBuffer = field(init=False)

    # The single-pass structure decide atomically sums the total requested tail
    # growth (sum of prod-1 over splits) into `densify_children_sum`. The host
    # reads it back after the tick to warn when a window's requested splits
    # exceed the buffer headroom (overflow children are dropped at the tail by
    # split_anisotropic — surfaced, not silently demoted).
    densify_children_sum: spy.NDBuffer = field(init=False)      # uint[1]

    # --- Cluster-Cull-Compact (Laine-2011 pattern, applied to 3DGS) ---
    # Gaussians are Morton-sorted every ~1000 iters (see morton_sort), so
    # contiguous [128k, 128k+128) slots are spatially adjacent. Every
    # train_step record_forward rebuilds a per-cluster world-space AABB
    # (mean +/- 3-sigma * |R| * exp(log_scale)) and tests it against the
    # camera frustum; the resulting cluster_visible[] flag is read by
    # bin_count to skip whole clusters at zero cost (they emit no tile
    # pairs, so radix/rasterize/backward all naturally ignore them —
    # slot indexing stays intact throughout, no compaction).
    # max_clusters = ceil(max_splats / CLUSTER_SIZE).
    cluster_aabb_min: spy.NDBuffer = field(init=False)  # float4[max_clusters] (.w unused)
    cluster_aabb_max: spy.NDBuffer = field(init=False)  # float4[max_clusters]
    cluster_visible: spy.NDBuffer = field(init=False)   # uint[max_clusters]

    @property
    def num_tiles(self) -> int:
        return self.num_tiles_x * self.num_tiles_y

    def _require_adam(self) -> AdamResources:
        if self.adam is None:
            raise RuntimeError("optimizer resources are unavailable in viewer-only mode")
        return self.adam

    def _require_densification(self) -> DensificationResources:
        if self.densification is None:
            raise RuntimeError("densification resources are unavailable in viewer-only mode")
        return self.densification

    def _require_eta(self) -> EtaResources:
        if self.eta is None:
            raise RuntimeError("eta resources are unavailable in viewer-only mode")
        return self.eta

    def _require_sh(self) -> "SphericalHarmonicsResources":
        if self.sh is None:
            raise RuntimeError("SH resources are unavailable (sh_degree == 0)")
        return self.sh

    @property
    def sh_enabled(self) -> bool:
        return self.sh is not None

    # Transitional compatibility aliases for paths not yet migrated to nested
    # capabilities. New components receive the capability objects directly.
    gaussians = property(lambda self: self.primitive.values)
    num_gaussians_counter = property(lambda self: self.primitive.active_count)
    filter_3D = property(lambda self: self.primitive.filter_3d)
    adam_state = property(lambda self: self._require_adam().state)
    scan_buffer = property(lambda self: self.scan.work)
    scan_block_sums = property(lambda self: self.scan.block_sums)
    scan_block_sums_l2 = property(lambda self: self.scan.block_sums_l2)
    scan_total = property(lambda self: self.scan.total)
    grad_mean_2d_mag_accum = property(
        lambda self: self._require_densification().grad_mean_2d_mag)
    visibility_count_accum = property(
        lambda self: self._require_densification().visibility_count)
    blend_weight_accum = property(
        lambda self: self._require_densification().blend_weight)
    scratch_arena = property(
        lambda self: self._require_densification().scratch_arena)
    action_buffer = property(lambda self: self._require_densification().actions)
    valid_mask_buffer = property(lambda self: self._require_densification().valid_mask)
    densify_pre_count = property(lambda self: self._require_densification().pre_count)
    densify_post_count = property(lambda self: self._require_densification().post_count)
    densify_dispatch_args = property(
        lambda self: self._require_densification().dispatch_args)
    eta_buffers = property(lambda self: self._require_eta().components)
    eta_n_total = property(lambda self: self._require_eta().n_total)
    eta_buffers_scratch = property(lambda self: self._require_eta().scratch_components)
    eta_n_total_scratch = property(lambda self: self._require_eta().scratch_n_total)
    features_rest = property(lambda self: self._require_sh().features_rest)
    features_rest_grad = property(lambda self: self._require_sh().features_rest_grad)
    adam_rest_state = property(lambda self: self._require_sh().adam_rest)

    def grad_kwargs(self) -> dict[str, spy.NDBuffer]:
        """Dispatch bindings for the 14 SoA scalar gradients:
        ``{"grad_<component>": buf.storage}``. The kwarg name is exactly
        ``"grad_" + component``, matching the entry-point parameters in
        adam.slang / backward.slang / projection_backward.slang
        (grad_mean_x ... grad_color_b). Splat into a ``_dispatch`` call with
        ``**b.grad_kwargs()`` — one dict built per dispatch, identical work
        to the explicit per-component kwargs it replaces."""
        return {f"grad_{name}": buf.storage for name, buf in self.grad_buffers.items()}

    def grad_seed_kwargs(self) -> dict[str, spy.NDBuffer]:
        """Dispatch bindings for the 5 summed 2D gradient seeds:
        ``grad_mean2d_x``, ``grad_cov2d_a`` ... — consumed by
        backward_persplat.slang and projection_backward.slang. Same
        ``"grad_" + component`` naming rule as :meth:`grad_kwargs`."""
        return {f"grad_{name}": buf.storage for name, buf in self.grad_2d_seed_buffers.items()}

    def eta_kwargs(self) -> dict[str, spy.NDBuffer]:
        """Dispatch bindings for the 9 per-axis eta accumulators:
        ``{"eta_<component>": buf.storage}`` (eta_n_high_x ... eta_max_z),
        matching score_structure_aware.slang's entry-point parameters."""
        return {f"eta_{name}": buf.storage for name, buf in self.eta_buffers.items()}

    def eta_scatter_kwargs(self) -> dict[str, spy.NDBuffer]:
        """Dispatch bindings for the eta-carrying scatter kernels
        (scatterCompactEta / mortonScatterEta): the 6 eta accumulators as
        ``eta_<name>_in`` (primary, read) + ``eta_<name>_out`` (scratch, write),
        where ``<name>`` in {n_total, n_high, n_low, max_x, max_y, max_z}.
        eta_n_total is bundled with the 5 eta_buffers components. The caller
        swaps primary<->scratch after the dispatch via :meth:`swap_eta`."""
        return self._require_eta().scatter_kwargs()

    def swap_eta(self) -> None:
        """Swap the eta primary<->scratch references after an eta-carrying
        scatter. Eta keeps its own dedicated scratch (it is small, 0.22 GB at
        10M) and uses the pointer-swap idiom; gaussians / adam / SH instead
        route through the shared ScratchArena and commit via arena->live copy,
        so they no longer participate in a swap."""
        self._require_eta().swap()

#TODO_refactor: same thing here, the _ShaderTypes has a hard contract with the Slang module.
@dataclass
class _ShaderTypes:
    """The Slang struct types :func:`allocate` sizes NDBuffers from,
    resolved once via reflection."""
    gaussian: object
    projected: object
    blend: object
    adam: object
    sh_moments: object

                  
#TODO_refactor: the reflection still has a hard dependency on the Slang module here, Ideally this type information should be passed from the caller
def _reflect_types(module: spy.Module) -> _ShaderTypes:
    """Look up the shader struct layouts by name (Python never duplicates the
    memory layout). Raises with the offending name if the module doesn't
    expose one — e.g. the reflection module was loaded from a file that no
    longer transitively includes common/types.slang."""
    def need(name: str) -> object:
        slang_type = module.layout.find_type_by_name(name)
        if slang_type is None:
            raise RuntimeError(f"shader module does not expose type '{name}'")
        return slang_type

    return _ShaderTypes(
        gaussian=need("Gaussian3D"),
        projected=need("Projected2DGaussian"),
        blend=need("BlendData"),
        adam=need("AdamState"),
        sh_moments=need("ShRestMoments"),
    )


@dataclass
class _Alloc:
    """Construction-time buffer/texture factory. Captures the invariant
    allocation args (device + frame dims) so each of the ~40 resources below
    reads as one typed_buffer line instead of repeating ``device=``/``width=``/
    ``height=``/``usage=``. Pure construction helper — not stored on
    :class:`GaussianBuffers`, no per-frame cost."""
    device: spy.Device
    width: int
    height: int

    def typed_buffer(self, label: str, dtype, *shape) -> spy.NDBuffer:
        # ``label`` names the VkBuffer for Nsight's Memory view; see
        # _make_labeled_ndbuffer. Pass the destination field's name.
        return _make_labeled_ndbuffer(
            self.device, dtype, shape, label, _DEFAULT_BUFFER_USAGE)

    def uint_buffer(self, label: str, *shape) -> spy.NDBuffer:
        return self.typed_buffer(label, "uint", *shape)

    def i64_buffer(self, label: str, *shape) -> spy.NDBuffer:
        return self.typed_buffer(label, "int64_t", *shape)

    def indirect_buffer(self, label: str, records: int = 1) -> spy.NDBuffer:
        # Indirect-arg buffers need BufferUsage.indirect_argument in addition
        # to shader_resource | unordered_access (the NDBuffer default) so the
        # kernel can WRITE the args AND the driver can READ them as dispatch
        # args (uint[3] = groups_x, 1, 1).
        return _make_labeled_ndbuffer(
            self.device, "uint", (3 * records,), label,
            _DEFAULT_BUFFER_USAGE | spy.BufferUsage.indirect_argument)

    def tex(self, fmt: spy.Format, label: str,
            usage: spy.TextureUsage | None = None) -> spy.Texture:
        return self.device.create_texture(
            type=spy.TextureType.texture_2d,
            format=fmt,
            width=self.width,
            height=self.height,
            usage=usage if usage is not None else (
                spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access),
            label=label,
        )


def scratch_resource_bytes(types: "_ShaderTypes", max_splats: int,
                           sh_rest_floats: int) -> int:
    """Largest whole-resource scatter: gaussians, main Adam, and SH features.

    Element strides come from reflection so this can never drift from the shader
    struct layouts. SH moments are independent SoA planes, copied in batches
    that fit this arena rather than allocating a second full moment buffer.
    """
    g_stride = int(types.gaussian.buffer_layout.reflection.stride)
    a_stride = int(types.adam.buffer_layout.reflection.stride)
    candidates = [max_splats * g_stride, max_splats * a_stride]
    if sh_rest_floats > 0:
        feat_uints = (sh_rest_floats + 1) // 2
        candidates.append(max_splats * feat_uints * 4)
    return max(candidates)


def _make_scratch_arena(allocator: "_Alloc", types: "_ShaderTypes",
                        max_splats: int, sh_rest_floats: int) -> ScratchArena:
    """Allocate the shared scratch arena as a uint buffer big enough (bytes) for
    the largest routed resource. Rounded up to a whole uint."""
    nbytes = scratch_resource_bytes(types, max_splats, sh_rest_floats)
    n_uints = (nbytes + 3) // 4
    buf = allocator.uint_buffer("densification.scratch_arena", n_uints)
    return ScratchArena(
        buffer=buf, capacity_bytes=n_uints * 4,
        copy_dispatch_args=allocator.indirect_buffer("densification.arena_copy_args"))


def allocate(
    device: spy.Device,
    shared_data_module: spy.Module,
    max_splats: int,
    width: int,
    height: int,
    max_sort_pairs: int = DEFAULT_MAX_SORT_PAIRS,
    sh_rest_floats: int = 0,
    structure_aware: bool = False,
    _allocator=None,
) -> GaussianBuffers:
    """Allocate all GPU resources needed by forward + backward + Adam.

    Constructs the :class:`GaussianBuffers` container from the size fields,
    then assigns each resource via the :class:`_Alloc` factory. Every buffer
    is declared exactly twice — once as a typed_buffer field on the dataclass, once
    here — which is the minimum that preserves typed_buffer attribute access.

    ``sh_rest_floats`` > 0 additionally allocates the optional view-dependent SH
    'rest' coefficient storage (bands 1..sh_degree); 0 (default) allocates none.
    """
    types = _reflect_types(shared_data_module)
    allocator = _allocator if _allocator is not None else _Alloc(device, width, height)

    num_tiles_x = _ceil_div(width, TILE_WIDTH)
    num_tiles_y = _ceil_div(height, TILE_HEIGHT)
    num_tiles = num_tiles_x * num_tiles_y

    # DeviceRadixSort scratch sizing (see RADIX_PART_SIZE / RADIX_GLOBAL_HIST_SIZE).
    max_blocks = _ceil_div(max_sort_pairs, RADIX_PART_SIZE)
    # Ragged forward-checkpoint sizing (see RASTER_BUCKET_SIZE). Tight upper
    # bound on total buckets: sum_t ceil(len_t / P) < (sum_t len_t)/P +
    # num_tiles = max_sort_pairs/P + num_tiles.
    max_buckets = _ceil_div(max_sort_pairs, RASTER_BUCKET_SIZE) + num_tiles
    # Scan primitive uses SCAN_BLOCK_SIZE threads/block; number of blocks is
    # ceil(N / SCAN_BLOCK_SIZE). Level-2 block sums enable the 3-level scan
    # needed when max_splats > 1M.
    num_l1_blocks = max(1, _ceil_div(max_splats, SCAN_BLOCK_SIZE))
    num_l2_blocks = max(1, _ceil_div(num_l1_blocks, SCAN_BLOCK_SIZE))

    buffers = GaussianBuffers(
        max_splats=max_splats,
        max_sort_pairs=max_sort_pairs,
        width=width,
        height=height,
        num_tiles_x=num_tiles_x,
        num_tiles_y=num_tiles_y,
    )

    # --- always-live gaussian state ---
    primitive_values = allocator.typed_buffer("primitive.values", types.gaussian, max_splats)
    active_count = allocator.uint_buffer("primitive.active_count", 1)

    # --- per-frame forward scratch / output ---
    buffers.projected_2d = allocator.typed_buffer("projected_2d", types.projected, max_splats)
    # Compact per-gaussian blend gather record (mean/cov2d/opacity/sh0), packed
    # by bin_count and consumed by the blend backward. GPU-internal scratch.
    buffers.blend_data = allocator.typed_buffer("blend_data", types.blend, max_splats)

    # v2.x uncapped-binning resources.
    buffers.tile_count_per_gaussian = allocator.uint_buffer("tile_count_per_gaussian", max_splats)
    buffers.tile_offset_per_gaussian = allocator.uint_buffer("tile_offset_per_gaussian", max_splats)
    buffers.sort_keys_lo_a = allocator.uint_buffer("sort_keys_lo_a", max_sort_pairs)
    buffers.sort_keys_hi_a = allocator.uint_buffer("sort_keys_hi_a", max_sort_pairs)
    buffers.sort_values_a = allocator.uint_buffer("sort_values_a", max_sort_pairs)
    buffers.sort_keys_lo_b = allocator.uint_buffer("sort_keys_lo_b", max_sort_pairs)
    buffers.sort_keys_hi_b = allocator.uint_buffer("sort_keys_hi_b", max_sort_pairs)
    buffers.sort_values_b = allocator.uint_buffer("sort_values_b", max_sort_pairs)
    buffers.radix_global_hist = allocator.uint_buffer("radix_global_hist", RADIX_GLOBAL_HIST_SIZE)
    buffers.radix_pass_hist = allocator.uint_buffer("radix_pass_hist", 256 * max_blocks)
    buffers.radix_dispatch_args = allocator.indirect_buffer("radix_dispatch_args")
    buffers.sort_stats = allocator.uint_buffer("sort_stats", 2)
    buffers.tile_start = allocator.uint_buffer("tile_start", num_tiles)
    buffers.tile_end = allocator.uint_buffer("tile_end", num_tiles)

    buffers.bucket_count = allocator.uint_buffer("bucket_count", num_tiles)
    buffers.bucket_offsets = allocator.uint_buffer("bucket_offsets", num_tiles)
    buffers.bucket_offset_total = allocator.uint_buffer("bucket_offset_total", 1)
    buffers.raster_checkpoints = allocator.typed_buffer(
        "raster_checkpoints", "uint2", max_buckets * TILE_AREA)

    # Live-bucket compaction (flat warp-per-bucket backward).
    buffers.live_bucket_count = allocator.uint_buffer("live_bucket_count", num_tiles)
    buffers.live_bucket_offsets = allocator.uint_buffer("live_bucket_offsets", num_tiles)
    buffers.live_bucket_total = allocator.uint_buffer("live_bucket_total", 1)
    buffers.live_buckets = allocator.typed_buffer("live_buckets", "uint2", max_buckets)
    buffers.bucket_dispatch_args = allocator.indirect_buffer("bucket_dispatch_args")

    buffers.output_image = allocator.tex(spy.Format.rgba32_float, "output_image")
    buffers.gaussian_count_per_pixel = allocator.tex(spy.Format.r32_uint, "gaussian_count_per_pixel")

    # --- v1 training resources ---
    buffers.target_image = allocator.tex(spy.Format.rgba8_unorm, "target_image")
    buffers.dL_dC_texture = allocator.tex(spy.Format.rgba32_float, "dL_dC")
    buffers.ssim_m1 = allocator.tex(spy.Format.rgba32_float, "ssim_m1")
    buffers.ssim_m2 = allocator.tex(spy.Format.rgba32_float, "ssim_m2")
    buffers.ssim_m3 = allocator.tex(spy.Format.rgba32_float, "ssim_m3")

    buffers.grad_buffers = {name: allocator.i64_buffer(f"grad_buffers[{name}]", max_splats) for name in GRAD_SCALAR_COMPONENTS}
    buffers.grad_2d_seed_buffers = {name: allocator.i64_buffer(f"grad_2d_seed_buffers[{name}]", max_splats) for name in GRAD_2D_SEED_COMPONENTS}

    adam_state = allocator.typed_buffer("adam.state", types.adam, max_splats)
    buffers.training_step = allocator.uint_buffer("training_step", 1)

    # --- v2 densification tracking ---
    grad_mean_2d_mag = allocator.i64_buffer("densification.grad_mean_2d_mag", max_splats)
    visibility_count = allocator.uint_buffer("densification.visibility_count", max_splats)
    blend_weight = allocator.i64_buffer("densification.blend_weight", max_splats)
    buffers.grad_2d_x_iter = allocator.i64_buffer("grad_2d_x_iter", max_splats)
    buffers.grad_2d_y_iter = allocator.i64_buffer("grad_2d_y_iter", max_splats)

    # --- Mip-Splatting 3D smoothing filter (per-gaussian scalar) ---
    filter_3d = allocator.typed_buffer("primitive.filter_3d", "float", max_splats)
    buffers.primitive = PrimitiveStorage(primitive_values, active_count, filter_3d)
    buffers.adam = AdamResources(adam_state)

    # --- v2 densification scratch + pipeline state ---
    # Shared scratch arena: sized to the largest single resource that routes
    # through it as a whole: gaussians, main Adam, and SH features. SH moments
    # use plane batches, so SH3 needs 112 B/slot here rather than 360 B/slot.
    scratch_arena = _make_scratch_arena(
        allocator, types, max_splats, sh_rest_floats)
    actions = allocator.uint_buffer("densification.actions", max_splats)
    valid_mask = allocator.uint_buffer("densification.valid_mask", max_splats)
    scan_work = allocator.uint_buffer("scan.work", max_splats)
    scan_block_sums = allocator.uint_buffer("scan.block_sums", num_l1_blocks)
    scan_block_sums_l2 = allocator.uint_buffer("scan.block_sums_l2", num_l2_blocks)
    scan_total = allocator.uint_buffer("scan.total", 1)
    buffers.scan = ScanResources(
        scan_work, scan_block_sums, scan_block_sums_l2, scan_total)

    # --- densify_simple wait-free dispatch state ---
    pre_count = allocator.uint_buffer("densification.pre_count", 1)
    post_count = allocator.uint_buffer("densification.post_count", 1)
    densify_dispatch_args = allocator.indirect_buffer("densification.dispatch_args")
    buffers.densification = DensificationResources(
        grad_mean_2d_mag=grad_mean_2d_mag,
        visibility_count=visibility_count,
        blend_weight=blend_weight,
        scratch_arena=scratch_arena,
        actions=actions,
        valid_mask=valid_mask,
        pre_count=pre_count,
        post_count=post_count,
        dispatch_args=densify_dispatch_args,
    )
    # Adam indirect-dispatch args (populated every iter from num_gaussians_counter).
    buffers.adam_dispatch_args = allocator.indirect_buffer("adam_dispatch_args")
    buffers.mip_filter_dispatch_args = allocator.indirect_buffer("mip_filter_dispatch_args", records=2)
    buffers.mip_filter_previous_count = allocator.uint_buffer("mip_filter_previous_count", 2)
    # First recomputation initializes the entire dead tail; later calls only
    # clear slots that ceased to be live. Counts never leave the GPU thereafter.
    buffers.mip_filter_previous_count.copy_from_numpy(
        np.array([max_splats, max_splats], dtype=np.uint32))

    # --- structure-aware densification (SAD-GS) ---
    if structure_aware:
        buffers.eta = EtaResources(
            components={name: allocator.uint_buffer(f"eta.components[{name}]", max_splats) for name in ETA_ACCUM_COMPONENTS},
            n_total=allocator.uint_buffer("eta.n_total", max_splats),
            scratch_components={
                name: allocator.uint_buffer(f"eta.scratch_components[{name}]", max_splats) for name in ETA_ACCUM_COMPONENTS},
            scratch_n_total=allocator.uint_buffer("eta.scratch_n_total", max_splats),
        )
        buffers.split_factor_x = allocator.uint_buffer("split_factor_x", max_splats)
        buffers.split_factor_y = allocator.uint_buffer("split_factor_y", max_splats)
        buffers.split_factor_z = allocator.uint_buffer("split_factor_z", max_splats)
        buffers.densify_children_sum = allocator.uint_buffer("densify_children_sum", 1)

    # --- optional view-dependent SH rest coefficients ---
    # Allocated only when sh_degree > 0. Features + gradients use fp16 pairs;
    # Adam m/v are SoA planes in the compiled moment precision. Initial features
    # are uploaded by buffer_io; adam_rest is zeroed by the clear kernel.
    if sh_rest_floats > 0:
        feat_uints = (sh_rest_floats + 1) // 2
        moment_planes = sh_rest_floats * int(types.sh_moments.buffer_layout.reflection.stride) // 4
        buffers.sh = SphericalHarmonicsResources(
            rest_floats=sh_rest_floats,
            feat_uints=feat_uints,
            features_rest=allocator.uint_buffer("sh.features_rest", max_splats * feat_uints),
            features_rest_grad=allocator.uint_buffer("sh.features_rest_grad", max_splats * feat_uints),
            # Never fp16: it underflowed the tiny SH-rest v=g^2(~1e-12) to 0,
            # crushing the step ~1000x. Compacted/re-sorted in bounded plane batches.
            adam_rest=allocator.uint_buffer("sh.adam_rest", max_splats * moment_planes),
            moment_planes=moment_planes,
        )

    # --- Cluster-Cull-Compact scratch ---
    max_clusters = _ceil_div(max_splats, CLUSTER_SIZE)
    buffers.cluster_aabb_min = allocator.typed_buffer("cluster_aabb_min", "float4", max_clusters)
    buffers.cluster_aabb_max = allocator.typed_buffer("cluster_aabb_max", "float4", max_clusters)
    buffers.cluster_visible = allocator.uint_buffer("cluster_visible", max_clusters)
    # Seed cluster_visible = all-ones so bin_count's early-out is a no-op
    # until (and unless) the cluster-cull kernel writes real flags. Lets
    # runs without --cluster-cull skip the two cull dispatches entirely
    # and keeps the pre-cluster-cull first forward correct.
    buffers.cluster_visible.copy_from_numpy(
        np.ones(max_clusters, dtype=np.uint32))

    buffers.num_gaussians_counter.copy_from_numpy(np.zeros(1, dtype=np.uint32))
    buffers.training_step.copy_from_numpy(np.zeros(1, dtype=np.uint32))
    buffers.sort_stats.copy_from_numpy(np.zeros(2, dtype=np.uint32))

    return buffers


def allocate_viewer(
    device: spy.Device,
    shared_data_module: spy.Module,
    max_splats: int,
    width: int,
    height: int,
    max_sort_pairs: int = DEFAULT_MAX_SORT_PAIRS,
    sh_rest_floats: int = 0,
    _allocator=None,
) -> GaussianBuffers:
    """Allocate ONLY the resources the viewer's ``record_forward`` touches.

    Skips every training-only buffer (grads/Adam/densify scratch/loss textures/
    SSIM maps/importance state/eta accumulators/split factors/fanout scratch).
    Roughly ~3.5 GB smaller than :func:`allocate` at max_splats=7M.

    The unfilled resource fields on the returned :class:`GaussianBuffers`
    dataclass are left unset — do NOT read them; ``ViewerPipeline`` only
    touches the subset enumerated below. Kept as one dataclass (rather than a
    separate viewer type) so the shared render path in ``pipeline.record_forward``
    can stay untouched.
    """
    types = _reflect_types(shared_data_module)
    allocator = _allocator if _allocator is not None else _Alloc(device, width, height)

    num_tiles_x = _ceil_div(width, TILE_WIDTH)
    num_tiles_y = _ceil_div(height, TILE_HEIGHT)
    num_tiles = num_tiles_x * num_tiles_y
    max_blocks = _ceil_div(max_sort_pairs, RADIX_PART_SIZE)
    max_buckets = _ceil_div(max_sort_pairs, RASTER_BUCKET_SIZE) + num_tiles
    num_l1_blocks = max(1, _ceil_div(max_splats, SCAN_BLOCK_SIZE))
    num_l2_blocks = max(1, _ceil_div(num_l1_blocks, SCAN_BLOCK_SIZE))

    buffers = GaussianBuffers(
        max_splats=max_splats,
        max_sort_pairs=max_sort_pairs,
        width=width,
        height=height,
        num_tiles_x=num_tiles_x,
        num_tiles_y=num_tiles_y,
    )

    # Gaussian state (blit target from trainer).
    primitive_values = allocator.typed_buffer("primitive.values", types.gaussian, max_splats)
    active_count = allocator.uint_buffer("primitive.active_count", 1)
    # bin_count binds filter_3D unconditionally (see pipeline.record_forward). The
    # viewer never computes the mip 3D filter and renders with mip_enabled=0, so
    # the shader ignores the contents — but the buffer must still exist for the
    # bind. Allocate it (zero cost vs. the sort/scan buffers) so the shared
    # forward path doesn't AttributeError on b.filter_3D.
    filter_3d = allocator.typed_buffer("primitive.filter_3d", "float", max_splats)
    buffers.primitive = PrimitiveStorage(primitive_values, active_count, filter_3d)

    # Forward scratch.
    buffers.projected_2d = allocator.typed_buffer("projected_2d", types.projected, max_splats)
    buffers.blend_data = allocator.typed_buffer("blend_data", types.blend, max_splats)

    # Binning.
    buffers.tile_count_per_gaussian = allocator.uint_buffer("tile_count_per_gaussian", max_splats)
    buffers.tile_offset_per_gaussian = allocator.uint_buffer("tile_offset_per_gaussian", max_splats)

    # Sort ping-pong.
    buffers.sort_keys_lo_a = allocator.uint_buffer("sort_keys_lo_a", max_sort_pairs)
    buffers.sort_keys_hi_a = allocator.uint_buffer("sort_keys_hi_a", max_sort_pairs)
    buffers.sort_values_a = allocator.uint_buffer("sort_values_a", max_sort_pairs)
    buffers.sort_keys_lo_b = allocator.uint_buffer("sort_keys_lo_b", max_sort_pairs)
    buffers.sort_keys_hi_b = allocator.uint_buffer("sort_keys_hi_b", max_sort_pairs)
    buffers.sort_values_b = allocator.uint_buffer("sort_values_b", max_sort_pairs)
    buffers.radix_global_hist = allocator.uint_buffer("radix_global_hist", RADIX_GLOBAL_HIST_SIZE)
    buffers.radix_pass_hist = allocator.uint_buffer("radix_pass_hist", 256 * max_blocks)
    buffers.radix_dispatch_args = allocator.indirect_buffer("radix_dispatch_args")
    buffers.sort_stats = allocator.uint_buffer("sort_stats", 2)

    # Tile ranges + bucket CSR (used by identify_tile_ranges + bucket_count + forward).
    buffers.tile_start = allocator.uint_buffer("tile_start", num_tiles)
    buffers.tile_end = allocator.uint_buffer("tile_end", num_tiles)
    buffers.bucket_count = allocator.uint_buffer("bucket_count", num_tiles)
    buffers.bucket_offsets = allocator.uint_buffer("bucket_offsets", num_tiles)
    buffers.bucket_offset_total = allocator.uint_buffer("bucket_offset_total", 1)
    buffers.raster_checkpoints = allocator.typed_buffer(
        "raster_checkpoints", "uint2", max_buckets * TILE_AREA)

    # Live-bucket compaction (flat warp-per-bucket backward).
    buffers.live_bucket_count = allocator.uint_buffer("live_bucket_count", num_tiles)
    buffers.live_bucket_offsets = allocator.uint_buffer("live_bucket_offsets", num_tiles)
    buffers.live_bucket_total = allocator.uint_buffer("live_bucket_total", 1)
    buffers.live_buckets = allocator.typed_buffer("live_buckets", "uint2", max_buckets)
    buffers.bucket_dispatch_args = allocator.indirect_buffer("bucket_dispatch_args")

    # Render outputs.
    buffers.output_image = allocator.tex(spy.Format.rgba32_float, "output_image")
    buffers.gaussian_count_per_pixel = allocator.tex(spy.Format.r32_uint, "gaussian_count_per_pixel")

    # Scan scratch (needed for tile-offset scan + bucket-offset scan).
    buffers.scan = ScanResources(
        work=None,
        block_sums=allocator.uint_buffer("scan.block_sums", num_l1_blocks),
        block_sums_l2=allocator.uint_buffer("scan.block_sums_l2", num_l2_blocks),
        total=allocator.uint_buffer("scan.total", 1),
    )

    # bin_count reads cluster_visible unconditionally. The viewer never
    # runs the cluster-cull kernels (they mutate this buffer per-frame from
    # the trainer's forward), so seed all-ones so the viewer's bin_count
    # skip check is a no-op.
    max_clusters_v = _ceil_div(max_splats, CLUSTER_SIZE)
    buffers.cluster_visible = allocator.uint_buffer("cluster_visible", max_clusters_v)
    buffers.cluster_visible.copy_from_numpy(
        np.ones(max_clusters_v, dtype=np.uint32))

    # Optional view-dependent SH: the viewer forward reads features_rest (blit
    # target from the trainer stream) + writes clamp_mask. No grad/Adam/scratch.
    if sh_rest_floats > 0:
        feat_uints = (sh_rest_floats + 1) // 2
        buffers.sh = SphericalHarmonicsResources(
            rest_floats=sh_rest_floats,
            feat_uints=feat_uints,
            features_rest=allocator.uint_buffer("sh.features_rest", max_splats * feat_uints),
            features_rest_grad=allocator.uint_buffer("sh.features_rest_grad", 1),
            adam_rest=allocator.uint_buffer("sh.adam_rest", 1),
            moment_planes=0,
        )

    buffers.num_gaussians_counter.copy_from_numpy(np.zeros(1, dtype=np.uint32))
    buffers.sort_stats.copy_from_numpy(np.zeros(2, dtype=np.uint32))

    return buffers


class _AllocationCounter:
    """Run the real allocation recipe without creating GPU resources."""

    def __init__(self, width: int, height: int) -> None:
        self.pixels = width * height
        self.total = 0

    def typed_buffer(self, label, dtype, *shape):
        from types import SimpleNamespace
        stride = ({"uint": 4, "float": 4, "int64_t": 8, "uint2": 8, "float4": 16}[dtype]
                  if isinstance(dtype, str)
                  else int(dtype.buffer_layout.reflection.stride))
        size = int(np.prod(shape)) * stride
        self.total += size
        return SimpleNamespace(storage=SimpleNamespace(size=size),
                               copy_from_numpy=lambda _data: None)

    def uint_buffer(self, label, *shape):
        return self.typed_buffer(label, "uint", *shape)

    def i64_buffer(self, label, *shape):
        return self.typed_buffer(label, "int64_t", *shape)

    def indirect_buffer(self, label, records=1):
        return self.uint_buffer(label, 3 * records)

    def tex(self, fmt, label, usage=None):
        self.total += self.pixels * {
            spy.Format.rgba32_float: 16,
            spy.Format.rgba8_unorm: 4,
            spy.Format.r32_uint: 4,
        }[fmt]


def estimate_allocation_bytes(shared_data_module, *, viewer_only=False, **sizes) -> int:
    """Base buffer/texture bytes; excludes sampler resources and driver overhead."""
    counter = _AllocationCounter(sizes["width"], sizes["height"])
    allocator = allocate_viewer if viewer_only else allocate
    allocator(device=None, shared_data_module=shared_data_module,
              _allocator=counter, **sizes)
    return counter.total
