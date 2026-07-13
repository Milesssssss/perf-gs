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


# --- Shader-layout constants ---
# Each MUST match the named shader. There is no codegen bridge between Python
# and Slang, so these comments are the contract that keeps the two in sync.
# Consumers (pipeline.py) import these rather than redefining them, so the
# value lives in exactly one place on the Python side.
TILE_SIZE = 16                    # tile_binned/*.slang, densify/score_forward.slang
CLUSTER_SIZE = 128                # rasterize/cluster_cull.slang CLUSTER_SIZE
                                  # (also referenced by bin_count.slang via gid >> 7)
RADIX_PART_SIZE = 2048            # sort/device_radix_sort.slang PART_SIZE (D_DIM * KEYS_PER_THREAD)
RADIX_GLOBAL_HIST_SIZE = 256 * 8  # device_radix_sort.slang: one 256-region per pass, 8 passes
SCAN_BLOCK_SIZE = 1024            # common/scan.slang SCAN_BLOCK_SIZE
RASTER_BUCKET_SIZE = 32           # rasterize/tile_binned/checkpoint.slang RASTER_BUCKET_SIZE
# Gaussian3D packed layout: 14 contiguous float32s, no padding —
# mean(3) + log_scale(3) + rot_quat(4) + opacity(1) + sh0(3). Used by the CPU
# marshalling in buffer_io.py and the densify readback in pipeline.py.
GAUSSIAN_FLOAT_COUNT = 14

# Default cap on the total number of (tile_id, depth, gaussian_idx) pairs
# emitted by bin_emit before the global radix sort. A single gaussian
# covering K tiles contributes K pairs. Computed as max_splats * 16
# (average tile coverage with headroom) in allocate(); this constant
# is only used when no explicit value is provided.
DEFAULT_MAX_SORT_PAIRS = 8_000_000


def _ceil_div(a: int, b: int) -> int:
    """Ceiling division ``ceil(a / b)`` for non-negative ints."""
    return (a + b - 1) // b


# Names of the 14 scalar grad buffers (SoA). Order matches how the
# backward kernel reads/writes them and how Adam consumes them.
GRAD_SCALAR_COMPONENTS = (
    "mean_x", "mean_y", "mean_z",
    "log_scale_x", "log_scale_y", "log_scale_z",
    "rot_w", "rot_x", "rot_y", "rot_z",
    "opacity",
    "color_r", "color_g", "color_b",
)

# Per-gaussian 2D gradient seeds, summed across all tiles a gaussian touches
# by the blend backward (backward_persplat.slang). projection_backward.slang
# then applies the 3D projection backward ONCE per gaussian on these sums,
# instead of once per (gaussian, tile). Atomic-accumulated, so cleared each
# iteration alongside GRAD_SCALAR_COMPONENTS.
GRAD_2D_SEED_COMPONENTS = (
    "mean2d_x", "mean2d_y",
    "cov2d_a", "cov2d_b", "cov2d_c",
)

# Structure-aware densification multiview accumulators (SoA uint), matching
# SADGS. n_high/n_low are SCALAR per-gaussian view counts (classified on the
# max-over-axes eta), so high_ratio = n_high/n_total drives the split GATE.
# max_x/y/z are per-axis max eta (fixed-point) driving the split FACTOR
# n_k = ceil(sqrt(eta_max_k)). Names match score_structure_aware.slang.
ETA_ACCUM_COMPONENTS = (
    "n_high", "n_low",
    "max_x", "max_y", "max_z",
)


@dataclass
class GaussianBuffers:
    """Container for every GPU resource the trainer uses.

    The 6 size fields are the only constructor inputs; every resource field is
    ``field(init=False)`` and assigned by :func:`allocate` immediately after
    construction. This keeps the resource fields honestly typed_buffer (no ``| None``)
    without needing dummy defaults or a place in ``__init__`` — every buffer
    below is ALWAYS allocated (see the module docstring: over-allocate once).
    The comment banners are purely organizational.
    """

    # --- sizes (the only constructor inputs) ---
    max_splats: int = 0
    max_sort_pairs: int = 0
    width: int = 0
    height: int = 0
    num_tiles_x: int = 0
    num_tiles_y: int = 0

    # --- always-live gaussian state ---
    gaussians: spy.NDBuffer = field(init=False)
    num_gaussians_counter: spy.NDBuffer = field(init=False)

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
    # Per-tile sorted entry ranges.
    tile_start: spy.NDBuffer = field(init=False)                # uint[num_tiles]
    tile_end: spy.NDBuffer = field(init=False)                  # uint[num_tiles]

    # Ragged (CSR) forward-checkpoint state for the transposed backward.
    # bucket_count       : uint[num_tiles], ceil(tile_len / P) per tile
    # bucket_offsets     : uint[num_tiles], exclusive scan of bucket_count
    #                      (base bucket index of each tile in the flat buffer)
    # bucket_offset_total: uint[1], total buckets used this frame (scan total)
    # raster_checkpoints : uint2[max_buckets * 256], (T,R,G,B) fp16 per pixel
    #                      per bucket. max_buckets = ceil(max_sort_pairs/P)
    #                      + num_tiles (tight upper bound on sum of per-tile
    #                      ceil(len/P), no readback needed).
    bucket_count: spy.NDBuffer = field(init=False)
    bucket_offsets: spy.NDBuffer = field(init=False)
    bucket_offset_total: spy.NDBuffer = field(init=False)
    raster_checkpoints: spy.NDBuffer = field(init=False)

    output_image: spy.Texture = field(init=False)          # rgba32_float, UAV+SRV
                                                            # (.a = 1-T; final transmittance
                                                            # recovered as T = 1 - .a in backward)
    gaussian_count_per_pixel: spy.Texture = field(init=False)  # r32_uint

    # --- v1 training resources ---
    target_image: spy.Texture = field(init=False)          # rgba32_float, SRV (read-only in shaders)
    dL_dC_texture: spy.Texture = field(init=False)         # rgba32_float, UAV+SRV
    # SSIM two-pass back-projection maps (m1/m2/m3), only used when lambda_ssim>0.
    ssim_m1: spy.Texture = field(init=False)  # rgba32_float, WxH
    ssim_m2: spy.Texture = field(init=False)  # rgba32_float, WxH
    ssim_m3: spy.Texture = field(init=False)  # rgba32_float, WxH
    grad_buffers: dict[str, spy.NDBuffer] = field(init=False)  # name -> int64[max_splats]
    # Per-gaussian summed 2D grad seeds (mean2d/cov2d) bridging the split
    # backward: blend kernel accumulates, projection kernel consumes.
    grad_2d_seed_buffers: dict[str, spy.NDBuffer] = field(init=False)
    adam_state: spy.NDBuffer = field(init=False)  # AdamState[max_splats]
    training_step: spy.NDBuffer = field(init=False)    # uint[1] (Adam bias correction)

    # --- v2 densification tracking (accumulated over a window) ---
    # grad_mean_2d_mag_accum   : fixed-point sum of per-iter norm-of-sum
    #                            2D gradients.
    # visibility_count_accum   : iter count (not pixel count) — matches
    #                            gsplat's `state["count"]`.
    # grad_2d_{x,y}_iter       : per-iter vector accumulators; cleared by
    #                            accumulate_iter at each step end.
    grad_mean_2d_mag_accum: spy.NDBuffer = field(init=False)  # int64[max_splats], fixed-point
    visibility_count_accum: spy.NDBuffer = field(init=False)  # uint[max_splats]
    grad_2d_x_iter: spy.NDBuffer = field(init=False)          # int64[max_splats], fixed-point
    grad_2d_y_iter: spy.NDBuffer = field(init=False)          # int64[max_splats], fixed-point

    # --- FastGS importance scoring ---
    importance_score_accum: spy.NDBuffer = field(init=False)  # uint[max_splats]
    growth_budget: spy.NDBuffer = field(init=False)           # int[1]
    prune_budget: spy.NDBuffer = field(init=False)            # int[1]
    error_mask: spy.Texture = field(init=False)               # r32_uint, WxH
    l1_map: spy.Texture = field(init=False)                   # r32_float, WxH
    l1_minmax: spy.NDBuffer = field(init=False)               # uint[2]

    # --- v2 densification scratch + pipeline state ---
    densify_score: spy.NDBuffer = field(init=False)       # float[max_splats]
    gaussians_scratch: spy.NDBuffer = field(init=False)   # Gaussian3D[max_splats]
    adam_state_scratch: spy.NDBuffer = field(init=False)  # AdamState[max_splats]
    action_buffer: spy.NDBuffer = field(init=False)       # uint[max_splats]
    valid_mask_buffer: spy.NDBuffer = field(init=False)   # uint[max_splats]
    scan_buffer: spy.NDBuffer = field(init=False)         # uint[max_splats]
    scan_block_sums: spy.NDBuffer = field(init=False)     # uint[max_blocks]
    scan_block_sums_l2: spy.NDBuffer = field(init=False)  # uint[max_l2_blocks]
    scan_total: spy.NDBuffer = field(init=False)          # uint[1]

    # --- densify_simple wait-free dispatch state ---
    # Snapshots of num_gaussians_counter[0] taken via snapshotCount so that
    # splitAndClone/buildPruneMask/scatterCompact can read the counts they
    # need as GPU-resident buffers instead of via CPU readback.
    densify_pre_count: spy.NDBuffer = field(init=False)      # uint[1]
    densify_post_count: spy.NDBuffer = field(init=False)     # uint[1]
    # DispatchIndirect args (groups_x, 1, 1) for the pre-count-sized
    # splitAndClone dispatch. Also requires BufferUsage.indirect_argument.
    densify_dispatch_args: spy.NDBuffer = field(init=False)  # uint[3]
    # DispatchIndirect args for the Adam step. Populated every iter from
    # num_gaussians_counter via prepareDensifyArgs (reused shader). Sized so
    # groups_x = ceil(count/256); dispatching indirect lets Adam launch only
    # the workgroups it needs (~1-3M gaussians vs max_splats=10M = 4x fewer
    # workgroups). Also requires BufferUsage.indirect_argument.
    adam_dispatch_args: spy.NDBuffer = field(init=False)     # uint[3]

    # --- structure-aware densification (SAD-GS, arXiv 2604.28016) ---
    # Per-axis multiview frequency-violation accumulators, name -> uint[max_splats].
    # Keys are ETA_ACCUM_COMPONENTS. Only allocated/used on the structure path.
    eta_buffers: dict[str, spy.NDBuffer] = field(init=False)
    eta_n_total: spy.NDBuffer = field(init=False)  # uint[max_splats], views observed
    # Scratch doubles for the eta accumulators, so the SAD-GS window statistics
    # can be carried through the slot permutations of a filler densify
    # (scatterCompactEta) and a Morton re-sort (mortonScatterEta) instead of
    # being zeroed. Same swap idiom as gaussians_scratch / adam_state_scratch.
    eta_buffers_scratch: dict[str, spy.NDBuffer] = field(init=False)
    eta_n_total_scratch: spy.NDBuffer = field(init=False)
    # Per-axis grid split factor n_k (uint[max_splats]) written by the GPU
    # decision (decide_actions_structure.slang) and consumed by
    # split_anisotropic.slang.
    split_factor_x: spy.NDBuffer = field(init=False)
    split_factor_y: spy.NDBuffer = field(init=False)
    split_factor_z: spy.NDBuffer = field(init=False)

    # GPU-side capacity-bound scratch for the wait-free structure densify.
    # Pass 1 accumulates the total requested tail growth (sum of prod-1 over
    # tentative splits) into `densify_children_sum` and a histogram of the
    # per-split fan-out (`prod-1` bins) into `densify_fanout_hist` (513 bins
    # is enough since prod<=max_split_factor^3=512). Pass 1.5 (a single-thread
    # dispatch) scans the histogram top-down and picks
    # `densify_fanout_threshold` — the smallest fan-out whose bin (plus all
    # higher bins) must be demoted to CLONE so the tail growth fits under
    # `max_splats - pre_count`. UINT_MAX means "no demotion needed". Pass 2
    # reads that threshold and demotes.
    densify_children_sum: spy.NDBuffer = field(init=False)      # uint[1]
    densify_fanout_hist: spy.NDBuffer = field(init=False)       # uint[513]
    densify_fanout_threshold: spy.NDBuffer = field(init=False)  # uint[1]

    # --- Cluster-Cull-Compact (Laine-2011 pattern, applied to 3DGS) ---
    # Gaussians are Morton-sorted every ~1000 iters (see morton_sort), so
    # contiguous [128k, 128k+128) slots are spatially adjacent. Every
    # train_step _run_forward rebuilds a per-cluster world-space AABB
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
        prim = {"n_total": self.eta_n_total, **self.eta_buffers}
        scratch = {"n_total": self.eta_n_total_scratch, **self.eta_buffers_scratch}
        kw = {f"eta_{name}_in": buf.storage for name, buf in prim.items()}
        kw.update({f"eta_{name}_out": buf.storage for name, buf in scratch.items()})
        return kw

    def swap_eta(self) -> None:
        """Swap the eta primary<->scratch references after an eta-carrying
        scatter, mirroring the gaussians/adam_state swap. The just-written
        scratch becomes the live buffer set."""
        self.eta_n_total, self.eta_n_total_scratch = \
            self.eta_n_total_scratch, self.eta_n_total
        self.eta_buffers, self.eta_buffers_scratch = \
            self.eta_buffers_scratch, self.eta_buffers


@dataclass
class _ShaderTypes:
    """The four Slang struct types :func:`allocate` sizes NDBuffers from,
    resolved once via reflection."""
    gaussian: object
    projected: object
    blend: object
    adam: object


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

    def typed_buffer(self, dtype, *shape) -> spy.NDBuffer:
        return spy.NDBuffer(device=self.device, dtype=dtype, shape=shape)

    def uint_buffer(self, *shape) -> spy.NDBuffer:
        return self.typed_buffer("uint", *shape)

    def i64_buffer(self, *shape) -> spy.NDBuffer:
        return self.typed_buffer("int64_t", *shape)

    def indirect_buffer(self) -> spy.NDBuffer:
        # Indirect-arg buffers need BufferUsage.indirect_argument in addition
        # to shader_resource | unordered_access (the NDBuffer default) so the
        # kernel can WRITE the args AND the driver can READ them as dispatch
        # args (uint[3] = groups_x, 1, 1).
        return spy.NDBuffer(
            device=self.device, dtype="uint", shape=(3,),
            usage=(spy.BufferUsage.shader_resource
                   | spy.BufferUsage.unordered_access
                   | spy.BufferUsage.indirect_argument),
        )

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


def allocate(
    device: spy.Device,
    shared_data_module: spy.Module,
    max_splats: int,
    width: int,
    height: int,
    max_sort_pairs: int = DEFAULT_MAX_SORT_PAIRS,
) -> GaussianBuffers:
    """Allocate all GPU resources needed by forward + backward + Adam.

    Constructs the :class:`GaussianBuffers` container from the size fields,
    then assigns each resource via the :class:`_Alloc` factory. Every buffer
    is declared exactly twice — once as a typed_buffer field on the dataclass, once
    here — which is the minimum that preserves typed_buffer attribute access.
    """
    types = _reflect_types(shared_data_module)
    allocator = _Alloc(device, width, height)

    num_tiles_x = _ceil_div(width, TILE_SIZE)
    num_tiles_y = _ceil_div(height, TILE_SIZE)
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
    buffers.gaussians = allocator.typed_buffer(types.gaussian, max_splats)
    buffers.num_gaussians_counter = allocator.uint_buffer(1)

    # --- per-frame forward scratch / output ---
    buffers.projected_2d = allocator.typed_buffer(types.projected, max_splats)
    # Compact per-gaussian blend gather record (mean/cov2d/opacity/sh0), packed
    # by bin_count and consumed by the blend backward. GPU-internal scratch.
    buffers.blend_data = allocator.typed_buffer(types.blend, max_splats)

    # v2.x uncapped-binning resources.
    buffers.tile_count_per_gaussian = allocator.uint_buffer(max_splats)
    buffers.tile_offset_per_gaussian = allocator.uint_buffer(max_splats)
    buffers.sort_keys_lo_a = allocator.uint_buffer(max_sort_pairs)
    buffers.sort_keys_hi_a = allocator.uint_buffer(max_sort_pairs)
    buffers.sort_values_a = allocator.uint_buffer(max_sort_pairs)
    buffers.sort_keys_lo_b = allocator.uint_buffer(max_sort_pairs)
    buffers.sort_keys_hi_b = allocator.uint_buffer(max_sort_pairs)
    buffers.sort_values_b = allocator.uint_buffer(max_sort_pairs)
    buffers.radix_global_hist = allocator.uint_buffer(RADIX_GLOBAL_HIST_SIZE)
    buffers.radix_pass_hist = allocator.uint_buffer(256 * max_blocks)
    buffers.radix_dispatch_args = allocator.indirect_buffer()
    buffers.tile_start = allocator.uint_buffer(num_tiles)
    buffers.tile_end = allocator.uint_buffer(num_tiles)

    buffers.bucket_count = allocator.uint_buffer(num_tiles)
    buffers.bucket_offsets = allocator.uint_buffer(num_tiles)
    buffers.bucket_offset_total = allocator.uint_buffer(1)
    buffers.raster_checkpoints = allocator.typed_buffer("uint2", max_buckets * 256)

    buffers.output_image = allocator.tex(spy.Format.rgba32_float, "output_image")
    buffers.gaussian_count_per_pixel = allocator.tex(spy.Format.r32_uint, "gaussian_count_per_pixel")

    # --- v1 training resources ---
    buffers.target_image = allocator.tex(spy.Format.rgba32_float, "target_image")
    buffers.dL_dC_texture = allocator.tex(spy.Format.rgba32_float, "dL_dC")
    buffers.ssim_m1 = allocator.tex(spy.Format.rgba32_float, "ssim_m1")
    buffers.ssim_m2 = allocator.tex(spy.Format.rgba32_float, "ssim_m2")
    buffers.ssim_m3 = allocator.tex(spy.Format.rgba32_float, "ssim_m3")

    buffers.grad_buffers = {name: allocator.i64_buffer(max_splats) for name in GRAD_SCALAR_COMPONENTS}
    buffers.grad_2d_seed_buffers = {name: allocator.i64_buffer(max_splats) for name in GRAD_2D_SEED_COMPONENTS}

    buffers.adam_state = allocator.typed_buffer(types.adam, max_splats)
    buffers.training_step = allocator.uint_buffer(1)

    # --- v2 densification tracking ---
    buffers.grad_mean_2d_mag_accum = allocator.i64_buffer(max_splats)
    buffers.visibility_count_accum = allocator.uint_buffer(max_splats)
    buffers.grad_2d_x_iter = allocator.i64_buffer(max_splats)
    buffers.grad_2d_y_iter = allocator.i64_buffer(max_splats)

    # --- FastGS importance scoring ---
    buffers.importance_score_accum = allocator.uint_buffer(max_splats)
    buffers.growth_budget = allocator.typed_buffer("int", 1)
    buffers.prune_budget = allocator.typed_buffer("int", 1)
    buffers.error_mask = allocator.tex(spy.Format.r32_uint, "error_mask")
    buffers.l1_map = allocator.tex(spy.Format.r32_float, "l1_map")
    buffers.l1_minmax = allocator.uint_buffer(2)

    # --- v2 densification scratch + pipeline state ---
    buffers.densify_score = allocator.typed_buffer("float", max_splats)
    buffers.gaussians_scratch = allocator.typed_buffer(types.gaussian, max_splats)
    buffers.adam_state_scratch = allocator.typed_buffer(types.adam, max_splats)
    buffers.action_buffer = allocator.uint_buffer(max_splats)
    buffers.valid_mask_buffer = allocator.uint_buffer(max_splats)
    buffers.scan_buffer = allocator.uint_buffer(max_splats)
    buffers.scan_block_sums = allocator.uint_buffer(num_l1_blocks)
    buffers.scan_block_sums_l2 = allocator.uint_buffer(num_l2_blocks)
    buffers.scan_total = allocator.uint_buffer(1)

    # --- densify_simple wait-free dispatch state ---
    buffers.densify_pre_count = allocator.uint_buffer(1)
    buffers.densify_post_count = allocator.uint_buffer(1)
    buffers.densify_dispatch_args = allocator.indirect_buffer()
    # Adam indirect-dispatch args (populated every iter from num_gaussians_counter).
    buffers.adam_dispatch_args = allocator.indirect_buffer()

    # --- structure-aware densification (SAD-GS) ---
    buffers.eta_buffers = {name: allocator.uint_buffer(max_splats) for name in ETA_ACCUM_COMPONENTS}
    buffers.eta_n_total = allocator.uint_buffer(max_splats)
    buffers.eta_buffers_scratch = {name: allocator.uint_buffer(max_splats) for name in ETA_ACCUM_COMPONENTS}
    buffers.eta_n_total_scratch = allocator.uint_buffer(max_splats)
    buffers.split_factor_x = allocator.uint_buffer(max_splats)
    buffers.split_factor_y = allocator.uint_buffer(max_splats)
    buffers.split_factor_z = allocator.uint_buffer(max_splats)
    # Wait-free structure densify capacity-bound scratch (tiny: ~2 KB total).
    buffers.densify_children_sum = allocator.uint_buffer(1)
    buffers.densify_fanout_hist = allocator.uint_buffer(513)
    buffers.densify_fanout_threshold = allocator.uint_buffer(1)

    # --- Cluster-Cull-Compact scratch ---
    max_clusters = _ceil_div(max_splats, CLUSTER_SIZE)
    buffers.cluster_aabb_min = allocator.typed_buffer("float4", max_clusters)
    buffers.cluster_aabb_max = allocator.typed_buffer("float4", max_clusters)
    buffers.cluster_visible = allocator.uint_buffer(max_clusters)
    # Seed cluster_visible = all-ones so bin_count's early-out is a no-op
    # until (and unless) the cluster-cull kernel writes real flags. Lets
    # runs without --cluster-cull skip the two cull dispatches entirely
    # and keeps the pre-cluster-cull first forward correct.
    buffers.cluster_visible.copy_from_numpy(
        np.ones(max_clusters, dtype=np.uint32))

    buffers.num_gaussians_counter.copy_from_numpy(np.zeros(1, dtype=np.uint32))
    buffers.training_step.copy_from_numpy(np.zeros(1, dtype=np.uint32))

    return buffers


def allocate_viewer(
    device: spy.Device,
    shared_data_module: spy.Module,
    max_splats: int,
    width: int,
    height: int,
    max_sort_pairs: int = DEFAULT_MAX_SORT_PAIRS,
) -> GaussianBuffers:
    """Allocate ONLY the resources the viewer's ``_run_forward`` touches.

    Skips every training-only buffer (grads/Adam/densify scratch/loss textures/
    SSIM maps/importance state/eta accumulators/split factors/fanout scratch).
    Roughly ~3.5 GB smaller than :func:`allocate` at max_splats=7M.

    The unfilled resource fields on the returned :class:`GaussianBuffers`
    dataclass are left unset — do NOT read them; ``ViewerPipeline`` only
    touches the subset enumerated below. Kept as one dataclass (rather than a
    separate viewer type) so the shared render path in ``pipeline._run_forward``
    can stay untouched.
    """
    types = _reflect_types(shared_data_module)
    allocator = _Alloc(device, width, height)

    num_tiles_x = _ceil_div(width, TILE_SIZE)
    num_tiles_y = _ceil_div(height, TILE_SIZE)
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
    buffers.gaussians = allocator.typed_buffer(types.gaussian, max_splats)
    buffers.num_gaussians_counter = allocator.uint_buffer(1)

    # Forward scratch.
    buffers.projected_2d = allocator.typed_buffer(types.projected, max_splats)
    buffers.blend_data = allocator.typed_buffer(types.blend, max_splats)

    # Binning.
    buffers.tile_count_per_gaussian = allocator.uint_buffer(max_splats)
    buffers.tile_offset_per_gaussian = allocator.uint_buffer(max_splats)

    # Sort ping-pong.
    buffers.sort_keys_lo_a = allocator.uint_buffer(max_sort_pairs)
    buffers.sort_keys_hi_a = allocator.uint_buffer(max_sort_pairs)
    buffers.sort_values_a = allocator.uint_buffer(max_sort_pairs)
    buffers.sort_keys_lo_b = allocator.uint_buffer(max_sort_pairs)
    buffers.sort_keys_hi_b = allocator.uint_buffer(max_sort_pairs)
    buffers.sort_values_b = allocator.uint_buffer(max_sort_pairs)
    buffers.radix_global_hist = allocator.uint_buffer(RADIX_GLOBAL_HIST_SIZE)
    buffers.radix_pass_hist = allocator.uint_buffer(256 * max_blocks)
    buffers.radix_dispatch_args = allocator.indirect_buffer()

    # Tile ranges + bucket CSR (used by identify_tile_ranges + bucket_count + forward).
    buffers.tile_start = allocator.uint_buffer(num_tiles)
    buffers.tile_end = allocator.uint_buffer(num_tiles)
    buffers.bucket_count = allocator.uint_buffer(num_tiles)
    buffers.bucket_offsets = allocator.uint_buffer(num_tiles)
    buffers.bucket_offset_total = allocator.uint_buffer(1)
    buffers.raster_checkpoints = allocator.typed_buffer("uint2", max_buckets * 256)

    # Render outputs.
    buffers.output_image = allocator.tex(spy.Format.rgba32_float, "output_image")
    buffers.gaussian_count_per_pixel = allocator.tex(spy.Format.r32_uint, "gaussian_count_per_pixel")

    # Scan scratch (needed for tile-offset scan + bucket-offset scan).
    buffers.scan_block_sums = allocator.uint_buffer(num_l1_blocks)
    buffers.scan_block_sums_l2 = allocator.uint_buffer(num_l2_blocks)
    buffers.scan_total = allocator.uint_buffer(1)

    # bin_count reads cluster_visible unconditionally. The viewer never
    # runs the cluster-cull kernels (they mutate this buffer per-frame from
    # the trainer's forward), so seed all-ones so the viewer's bin_count
    # skip check is a no-op.
    max_clusters_v = _ceil_div(max_splats, CLUSTER_SIZE)
    buffers.cluster_visible = allocator.uint_buffer(max_clusters_v)
    buffers.cluster_visible.copy_from_numpy(
        np.ones(max_clusters_v, dtype=np.uint32))

    buffers.num_gaussians_counter.copy_from_numpy(np.zeros(1, dtype=np.uint32))
    return buffers


# CPU-side data marshalling lives in buffer_io.py (a different axis of change
# from GPU resource allocation). Re-exported here so existing callers can keep
# using ``buffers.write_gaussians(...)`` etc. The import is at the bottom to
# avoid a circular import — buffer_io imports GaussianBuffers from this module.
from trainer.buffer_io import (  # noqa: E402
    set_num_gaussians,
    write_gaussians,
    write_gaussians_raw,
    zero_adam_state,
)

__all__ = [
    "TILE_SIZE",
    "CLUSTER_SIZE",
    "RADIX_PART_SIZE",
    "RADIX_GLOBAL_HIST_SIZE",
    "SCAN_BLOCK_SIZE",
    "RASTER_BUCKET_SIZE",
    "GAUSSIAN_FLOAT_COUNT",
    "DEFAULT_MAX_SORT_PAIRS",
    "GRAD_SCALAR_COMPONENTS",
    "GRAD_2D_SEED_COMPONENTS",
    "GaussianBuffers",
    "allocate",
    "allocate_viewer",
    "set_num_gaussians",
    "write_gaussians",
    "write_gaussians_raw",
    "zero_adam_state",
]
