# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU driver for importance-guided sparse backward sampling.

Owns the resources and the dispatch sequence for
``shaders/losses/adaptive_importance.slang``, and — the point of the design —
hands the Gaussian backward the SAME four ``stbn_*`` parameters it already takes.
The table it produces is screen-sized and single-slice instead of 128x128x64, and
its per-tile counts vary because the marginal is nonuniform, but both of those
were already true of a global STBN threshold (which also varies per-tile counts),
so the shader side needs no change at all.

Ordering inside a training step:

    forward
    loss          DENSE (mask off) -> dL_dC holds h_i for the whole image
    THIS PASS     h_i -> s -> s~ -> a -> lambda -> p -> M; rewrites dL_dC as
                  M h / p and writes the packed mask table
    backward      reads the packed table exactly as before

Everything runs on the device: the lambda solve is a bisection over a 257-entry
histogram inside one workgroup, not a host readback. A per-iteration readback
would stall the queue and would show up as a regression in the very timing this
feature exists to improve — so the diagnostics are accumulated into a
per-iteration slot on the device and drained once, at the end.

The math and the reasoning for each step live in
:mod:`trainer.backend.render.adaptive_stbn`, which is also the host reference
the tests pin this against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

from trainer.backend.render import adaptive_stbn as _adaptive
from trainer.backend.render.adaptive_stbn import AdaptiveSpec
from trainer.backend.render.stbn import StbnMask
from trainer.contracts.codegen import TILE_AREA, TILE_HEIGHT, TILE_WIDTH

# Mirrors of the shader-side layout constants. A mismatch would be silent, so
# ``test_adaptive_stbn.py::test_shader_constants_match_the_host`` greps the
# shader for each of these rather than trusting the comment.
SCALAR_MEAN = 0
SCALAR_A_MIN = 1
SCALAR_A_MAX = 2
SCALAR_LAMBDA = 3
SCALAR_BIN_LO = 4
SCALAR_BIN_WIDTH = 5
SCALAR_DEGENERATE = 6
SCALAR_COUNT = 8

BINS = 256
HIST_SCALE = 4096.0

STAT_SUM_P = 0
STAT_SELECTED = 1
STAT_SATURATED = 2
STAT_LAMBDA = 3
STAT_I64_STRIDE = 4
STAT_MIN_P = 0
STAT_MAX_P = 1
STAT_PIXELS = 2
STAT_U32_STRIDE = 4
P_SCALE = 1073741824.0                                    # 2^30

REDUCE_THREADS = 256

# How many per-iteration diagnostic slots the stats buffers hold. 4096 covers the
# 3000-iteration runs this study uses, so the drained history is complete rather
# than a wrapped tail; longer runs wrap and the drain reports how many slots were
# overwritten instead of pretending otherwise.
DEFAULT_STATS_SLOTS = 4096


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


@dataclass(frozen=True)
class AdaptiveStats:
    """One iteration's realized sampling, drained from the device."""
    iteration: int
    n_pixels: int
    p_mean: float
    p_min: float
    p_max: float
    saturated_fraction: float
    expected_samples: float
    selected_samples: int
    selected_fraction: float
    lam: float

    def as_row(self) -> dict:
        return {
            "iteration": self.iteration,
            "n_pixels": self.n_pixels,
            "p_mean": self.p_mean,
            "p_min": self.p_min,
            "p_max": self.p_max,
            "saturated_fraction": self.saturated_fraction,
            "expected_samples": self.expected_samples,
            "selected_samples": self.selected_samples,
            "selected_fraction": self.selected_fraction,
            "lambda": self.lam,
        }


class AdaptiveSampler:
    """Per-iteration adaptive mask construction on the device.

    One instance per pipeline; sized to the frame at construction because the
    coarse grid, the padded mask table and the ``a`` texture all depend on it.
    """

    def __init__(self, *, device: spy.Device, width: int, height: int,
                 spec: AdaptiveSpec, kernels: Any,
                 stats_slots: int = DEFAULT_STATS_SLOTS,
                 cache_dir: Path | str | None = None) -> None:
        spec.validate()
        self.device = device
        self.spec = spec
        self.width = int(width)
        self.height = int(height)
        self.k = kernels

        self.downsample = _adaptive.resolve_downsample(spec.sigma, spec.downsample)
        self.coarse_sigma = _adaptive.coarse_sigma(spec.sigma, self.downsample)
        self.blur_radius = (0 if self.coarse_sigma <= 0.0
                            else int(np.ceil(3.0 * self.coarse_sigma)))
        coarse_h, coarse_w = _adaptive.coarse_dims(self.width, self.height,
                                                   self.downsample)
        self.coarse_dim = (coarse_w, coarse_h)
        self.cell_count = coarse_w * coarse_h
        self.stat_groups = _ceil_div(self.cell_count, REDUCE_THREADS)

        # The mask table is allocated at the TILE-PADDED frame size so that
        # stbnBlockBase(tile origin, padded dims, slice 0) == tile_idx *
        # TILE_AREA — i.e. so the existing shader indexing lands on it unchanged.
        self.num_tiles = (_ceil_div(self.width, TILE_WIDTH),
                          _ceil_div(self.height, TILE_HEIGHT))
        self.table_dims = (self.num_tiles[0] * TILE_WIDTH,
                           self.num_tiles[1] * TILE_HEIGHT)
        self.pixel_count = self.width * self.height
        self.stats_slots = max(1, int(stats_slots))
        # Slots are assigned by RECORD ORDER, not by the caller's iteration
        # number. The gradient diagnostics call this many times at a frozen state
        # where the iteration counter never advances, and keying on it would have
        # every measurement overwrite slot 0.
        self._iterations: list[int] = []

        usage = (spy.BufferUsage.shader_resource
                 | spy.BufferUsage.unordered_access)

        def buf(dtype: str, count: int, label: str) -> spy.NDBuffer:
            return spy.NDBuffer(device=device, dtype=dtype,
                                shape=(max(int(count), 1),), usage=usage)

        # Ping-pong for the two separable blur passes. Two buffers rather than an
        # in-place blur because a separable pass reads a neighbourhood the same
        # dispatch is writing.
        self.coarse_a = buf("float", self.cell_count, "adaptive_coarse_a")
        self.coarse_b = buf("float", self.cell_count, "adaptive_coarse_b")
        self.partials = buf("float", self.stat_groups * 3, "adaptive_partials")
        self.scalars = buf("float", SCALAR_COUNT, "adaptive_scalars")
        self.hist_count = buf("uint", BINS + 1, "adaptive_hist_count")
        self.hist_sum = buf("int64_t", BINS, "adaptive_hist_sum")
        self.stats_i64 = buf("int64_t", self.stats_slots * STAT_I64_STRIDE,
                             "adaptive_stats_i64")
        self.stats_u32 = buf("uint", self.stats_slots * STAT_U32_STRIDE,
                             "adaptive_stats_u32")
        self.mask_table = buf("uint", self.table_dims[0] * self.table_dims[1],
                              "adaptive_mask_table")
        self.a_texture = device.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.r32_float,
            width=self.width, height=self.height,
            usage=(spy.TextureUsage.shader_resource
                   | spy.TextureUsage.unordered_access),
            label="adaptive_a")

        self.u_table, self.u_dims, self.u_slices = self._upload_uniform(cache_dir)
        # The backward reads the mask table through StbnMask's parameter block, so
        # the adaptive path presents itself as just another table. slices=1: the
        # temporal axis lives in the U volume, not in this per-iteration table.
        self._mask = StbnMask(table=self.mask_table.storage, dims=self.table_dims,
                              slices=1)

        print(f"[adaptive] {spec.uniform} alpha={spec.alpha:g} sigma={spec.sigma:g} "
              f"(coarse {coarse_w}x{coarse_h} /{self.downsample}, sigma_c="
              f"{self.coarse_sigma:.3f}, radius {self.blur_radius}) "
              f"rho={spec.budget_fraction():.6g}"
              f"{'' if spec.unbiased else ' BIASED CONTROL (no 1/p weight)'}")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _upload_uniform(self, cache_dir) -> tuple[spy.Buffer, tuple[int, int], int]:
        """The U volume, or a 1-element dummy for the hash mode.

        A declared shader parameter must be bound even when the branch that reads
        it is not taken, which is why the hash mode still gets a buffer.
        """
        if not self.spec.uses_table:
            dummy = self.device.create_buffer(
                size=4, usage=(spy.BufferUsage.shader_resource
                               | spy.BufferUsage.unordered_access),
                label="adaptive_u_disabled")
            dummy.copy_from_numpy(np.zeros(1, dtype=np.float32).view(np.uint8))
            return dummy, (TILE_WIDTH, TILE_HEIGHT), 1

        mask_spec = _adaptive.spec_to_mask_spec(self.spec)
        volume = _cached_uniform_volume(mask_spec, cache_dir)
        flat = np.ascontiguousarray(volume.astype(np.float32).reshape(-1))
        buffer = self.device.create_buffer(
            size=flat.nbytes, usage=(spy.BufferUsage.shader_resource
                                     | spy.BufferUsage.unordered_access),
            label="adaptive_u_table")
        buffer.copy_from_numpy(flat.view(np.uint8))
        slices, height, width = volume.shape
        return buffer, (width, height), slices

    # ------------------------------------------------------------------
    # Per-iteration
    # ------------------------------------------------------------------

    @property
    def slices(self) -> int:
        """Temporal period of the U volume — what the frontend wraps visits on."""
        return self.u_slices

    def shader_kwargs(self, budget_pixels_per_tile: int) -> dict:
        """The four ``stbn_*`` parameters for the backward, pointing at THIS
        iteration's table.

        ``stbn_tile_pixels`` is the nominal per-tile budget. Under the adaptive
        marginal it is not the count of any particular tile — the backward already
        reads that from the table — and its only remaining job is to make
        ``stbnActive`` true so the compacted path is taken. Passing the nominal
        budget rather than, say, 1 keeps that gate reading the same value the
        uniform path passes at the same rho.
        """
        return self._mask.shader_kwargs(0, int(budget_pixels_per_tile))

    def record(self, pipeline: Any, *, image_gradient: spy.Texture,
               slice_index: int, sample_seed: int, iteration: int) -> None:
        """Record the eight-dispatch sequence onto ``pipeline``'s active batch."""
        spec = self.spec
        rho = spec.budget_fraction()
        slot = len(self._iterations) % self.stats_slots
        self._iterations.append(int(iteration))

        frame = spy.uint2(self.width, self.height)
        coarse = spy.uint2(self.coarse_dim[0], self.coarse_dim[1])

        with pipeline.marker_group("adaptive_importance"):
            pipeline.dispatch(self.k.k_importance_reduce, "importance_reduce",
                thread_count=[self.coarse_dim[0], self.coarse_dim[1], 1],
                dL_dC=image_gradient,
                coarse_out=self.coarse_a.storage,
                frame_dim=frame,
                coarse_dim=coarse,
                downsample=int(self.downsample),
            )
            # Both passes always run, even at radius 0 (where the tap loop
            # degenerates to a copy). Skipping them would leave the smoothed field
            # in whichever buffer the last non-skipped pass wrote, and that
            # bookkeeping is a far better source of bugs than two trivial
            # dispatches on a grid of a few tens of thousands of cells.
            pipeline.dispatch(self.k.k_importance_blur_x, "importance_blur_x",
                thread_count=[self.coarse_dim[0], self.coarse_dim[1], 1],
                source=self.coarse_a.storage,
                destination=self.coarse_b.storage,
                coarse_dim=coarse,
                sigma=float(max(self.coarse_sigma, 1e-6)),
                radius=int(self.blur_radius),
            )
            pipeline.dispatch(self.k.k_importance_blur_y, "importance_blur_y",
                thread_count=[self.coarse_dim[0], self.coarse_dim[1], 1],
                source=self.coarse_b.storage,
                destination=self.coarse_a.storage,
                coarse_dim=coarse,
                sigma=float(max(self.coarse_sigma, 1e-6)),
                radius=int(self.blur_radius),
            )
            pipeline.dispatch(self.k.k_importance_stats, "importance_stats",
                thread_count=[self.stat_groups * REDUCE_THREADS, 1, 1],
                smoothed=self.coarse_a.storage,
                partials=self.partials.storage,
                cell_count=int(self.cell_count),
                group_count=int(self.stat_groups),
            )
            pipeline.dispatch(self.k.k_importance_resolve, "importance_resolve",
                thread_count=[REDUCE_THREADS, 1, 1],
                partials=self.partials.storage,
                scalars=self.scalars.storage,
                hist_count=self.hist_count.storage,
                hist_sum=self.hist_sum.storage,
                stats_i64=self.stats_i64.storage,
                stats_u32=self.stats_u32.storage,
                group_count=int(self.stat_groups),
                cell_count=int(self.cell_count),
                pixel_count=int(self.pixel_count),
                stats_slot=slot,
                alpha=float(spec.alpha),
                rho=float(rho),
            )
            pipeline.dispatch(self.k.k_importance_histogram, "importance_histogram",
                thread_count=[self.pixel_count, 1, 1],
                smoothed=self.coarse_a.storage,
                scalars=self.scalars.storage,
                a_texture=self.a_texture,
                hist_count=self.hist_count.storage,
                hist_sum=self.hist_sum.storage,
                frame_dim=frame,
                coarse_dim=coarse,
                downsample=int(self.downsample),
                alpha=float(spec.alpha),
            )
            pipeline.dispatch(self.k.k_solve_lambda, "solve_lambda",
                thread_count=[REDUCE_THREADS, 1, 1],
                hist_count=self.hist_count.storage,
                hist_sum=self.hist_sum.storage,
                scalars=self.scalars.storage,
                stats_i64=self.stats_i64.storage,
                pixel_count=int(self.pixel_count),
                stats_slot=slot,
                rho=float(rho),
            )
        with pipeline.marker_group("adaptive_apply"):
            pipeline.dispatch(self.k.k_apply_adaptive, "apply_adaptive",
                thread_count=[self.num_tiles[0] * self.num_tiles[1] * TILE_AREA,
                              1, 1],
                dL_dC=image_gradient,
                a_texture=self.a_texture,
                scalars=self.scalars.storage,
                u_table=self.u_table,
                mask_table=self.mask_table.storage,
                stats_i64=self.stats_i64.storage,
                stats_u32=self.stats_u32.storage,
                frame_dim=frame,
                num_tiles=spy.uint2(self.num_tiles[0], self.num_tiles[1]),
                u_dims=spy.uint2(self.u_dims[0], self.u_dims[1]),
                u_slice=int(slice_index) % max(self.u_slices, 1),
                u_mode=int(spec.u_mode),
                u_seed=int(sample_seed) & 0xFFFFFFFF,
                stats_slot=slot,
                rho=float(rho),
                unbiased=1 if spec.unbiased else 0,
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def read_scalars(self) -> dict:
        """The current iteration's lambda and importance range. Synchronizing —
        for tests and offline probes, never the training loop."""
        raw = np.frombuffer(memoryview(self.scalars.to_numpy()),
                            dtype=np.float32)[:SCALAR_COUNT]
        return {
            "mean_s": float(raw[SCALAR_MEAN]),
            "a_min": float(raw[SCALAR_A_MIN]),
            "a_max": float(raw[SCALAR_A_MAX]),
            "lambda": float(raw[SCALAR_LAMBDA]),
            "bin_lo": float(raw[SCALAR_BIN_LO]),
            "bin_width": float(raw[SCALAR_BIN_WIDTH]),
            "degenerate": bool(raw[SCALAR_DEGENERATE] > 0.5),
        }

    def read_probabilities(self) -> np.ndarray:
        """``p_i`` at full resolution. Synchronizing; visualization / tests."""
        lam = self.read_scalars()["lambda"]
        a = np.asarray(self.a_texture.to_numpy(), dtype=np.float64)
        return np.minimum(1.0, lam * a.reshape(self.height, self.width))

    def read_mask(self) -> np.ndarray:
        """``M_i`` at full resolution, decoded from the PACKED TABLE.

        Read from the table the backward will consume rather than recomputed from
        p and U, so a test that compares against the uniform path is checking the
        bytes the rasterizer sees — which is where a packing bug would otherwise
        hide as a quietly wrong gradient.
        """
        packed = np.frombuffer(memoryview(self.mask_table.to_numpy()),
                              dtype=np.uint32)
        tiles_x, tiles_y = self.num_tiles
        blocks = packed[:tiles_x * tiles_y * TILE_AREA].reshape(
            tiles_y, tiles_x, TILE_HEIGHT, TILE_WIDTH)
        full = blocks.transpose(0, 2, 1, 3).reshape(tiles_y * TILE_HEIGHT,
                                                    tiles_x * TILE_WIDTH)
        return ((full & 0xFF) != 0xFF)[:self.height, :self.width]

    def drain_stats(self) -> list[AdaptiveStats]:
        """Every recorded iteration's realized sampling. Synchronizing; call once.

        Reports the MEASURED mean(p) and the MEASURED selected count, not the
        histogram solver's estimate of either. The budget claim
        ``(1/N) sum p_i = rho`` is then something the run verifies rather than
        something the solver asserts.
        """
        i64 = np.frombuffer(memoryview(self.stats_i64.to_numpy()), dtype=np.int64)
        u32 = np.frombuffer(memoryview(self.stats_u32.to_numpy()), dtype=np.uint32)
        rows: list[AdaptiveStats] = []
        # A run longer than the slot count overwrites the oldest entries, so only
        # the surviving tail is reported. wrapped_slots says how many were lost
        # rather than letting a partial history look complete.
        recorded = len(self._iterations)
        surviving = self._iterations[max(0, recorded - self.stats_slots):]
        for offset, iteration in enumerate(surviving):
            slot = (recorded - len(surviving) + offset) % self.stats_slots
            n_pixels = int(u32[slot * STAT_U32_STRIDE + STAT_PIXELS])
            if n_pixels <= 0:
                continue
            sum_p = float(i64[slot * STAT_I64_STRIDE + STAT_SUM_P]) / P_SCALE
            selected = int(i64[slot * STAT_I64_STRIDE + STAT_SELECTED])
            saturated = int(i64[slot * STAT_I64_STRIDE + STAT_SATURATED])
            lam = float(i64[slot * STAT_I64_STRIDE + STAT_LAMBDA]) / P_SCALE
            min_bits = np.uint32(u32[slot * STAT_U32_STRIDE + STAT_MIN_P])
            max_bits = np.uint32(u32[slot * STAT_U32_STRIDE + STAT_MAX_P])
            rows.append(AdaptiveStats(
                iteration=iteration,
                n_pixels=n_pixels,
                p_mean=sum_p / n_pixels,
                p_min=float(min_bits.view(np.float32)) if min_bits != 0xFFFFFFFF else 0.0,
                p_max=float(max_bits.view(np.float32)),
                saturated_fraction=saturated / n_pixels,
                expected_samples=sum_p,
                selected_samples=selected,
                selected_fraction=selected / n_pixels,
                lam=lam,
            ))
        return rows

    def stats_summary(self) -> dict:
        """Run-level aggregate of :meth:`drain_stats`, for the run report."""
        rows = self.drain_stats()
        wrapped = max(0, len(self._iterations) - self.stats_slots)
        if not rows:
            return {"iterations_recorded": 0, "wrapped_slots": wrapped}
        def mean(key: str) -> float:
            return float(np.mean([getattr(r, key) for r in rows]))
        return {
            "iterations_recorded": len(rows),
            "iterations_dispatched": len(self._iterations),
            "wrapped_slots": wrapped,
            "rho_target": float(self.spec.budget_fraction()),
            "p_mean": mean("p_mean"),
            "p_min": float(np.min([r.p_min for r in rows])),
            "p_max": float(np.max([r.p_max for r in rows])),
            "saturated_fraction_mean": mean("saturated_fraction"),
            "selected_fraction_mean": mean("selected_fraction"),
            "lambda_mean": mean("lam"),
            "lambda_min": float(np.min([r.lam for r in rows])),
            "lambda_max": float(np.max([r.lam for r in rows])),
            "budget_error_max": float(np.max(
                [abs(r.p_mean - self.spec.budget_fraction()) for r in rows])),
        }

    def config_record(self) -> dict:
        """Everything that determines this sampler's behavior, for the manifest."""
        return {
            "uniform": self.spec.uniform,
            "alpha": float(self.spec.alpha),
            "sigma": float(self.spec.sigma),
            "downsample_requested": int(self.spec.downsample),
            "downsample_used": int(self.downsample),
            "coarse_sigma": float(self.coarse_sigma),
            "blur_radius": int(self.blur_radius),
            "coarse_dim": list(self.coarse_dim),
            "unbiased": bool(self.spec.unbiased),
            "importance": self.spec.importance,
            "source": self.spec.source,
            "stbn_seed": int(self.spec.seed),
            "rho_requested": float(self.spec.fraction),
            "rho_enforced": float(self.spec.budget_fraction()),
            "u_slices": int(self.u_slices),
            "bins": BINS,
        }


# ---------------------------------------------------------------------------
# Uniform-volume cache
# ---------------------------------------------------------------------------

_MEMO: dict[tuple, np.ndarray] = {}


def _cached_uniform_volume(mask_spec, cache_dir) -> np.ndarray:
    """The per-slice rank normalization of a scalar volume, cached on disk.

    Ranking 64 slices of 128x128 costs a couple of seconds — fine once, wasteful
    on every cell of a sweep — and is a pure function of the volume. A corrupt or
    stale entry is rebuilt rather than raised on: the cache is an optimization,
    never a correctness dependency (same policy as ``stbn.load_or_build``).
    """
    key = (mask_spec.source, str(mask_spec.resolved_path()), mask_spec.seed,
           mask_spec.slices, mask_spec.height, mask_spec.width)
    memo = _MEMO.get(key)
    if memo is not None:
        return memo
    path = None
    if cache_dir is not None:
        name = (f"adaptive_u_{mask_spec.source}_{mask_spec.width}x"
                f"{mask_spec.height}x{mask_spec.slices}_s{mask_spec.seed}")
        if mask_spec.source != "white":
            stat = mask_spec.resolved_path().stat()
            name += f"_{stat.st_size}_{int(stat.st_mtime)}"
        path = Path(cache_dir) / f"{name}.npy"
        if path.is_file():
            try:
                cached = np.load(path)
                if cached.shape == (mask_spec.slices, mask_spec.height,
                                    mask_spec.width):
                    _MEMO[key] = cached
                    return cached
            except (OSError, ValueError):
                pass
    volume = _adaptive.uniform_volume(mask_spec).astype(np.float32)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, volume)
        except OSError:
            pass
    _MEMO[key] = volume
    return volume
