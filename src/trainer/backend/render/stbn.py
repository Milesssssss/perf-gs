# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Tile-stratified spatiotemporal blue-noise (STBN) pixel-sampling masks.

Builds the sampling mask for the opt-in sparse-supervision mode: each backward
supervises a fraction rho of the pixels, reweighted by 1/rho so the gradient
stays an unbiased estimate of the full-image one. Which pixels matters —
independent draws clump spatially and leave pixels unsampled for many visits,
where blue noise spreads them in both.

Sources (``StbnMaskSpec.source``):
  nvidia  the published 128x128x64 scalar set from github.com/NVIDIA-RTX/STBN,
          fetched by scripts/data/download_stbn.py (NON-COMMERCIAL license, so
          the download is opt-in and the assets are never vendored)
  file    any (slices, height, width) scalar volume
  white   independent random per-tile permutations - the control

Which pixels a density selects is set by ``StbnMaskSpec.select_mode``, and the
choice matters more than it looks:

  global (default)  Threshold the WHOLE slice — keep its lowest rho fraction.
                    This is the point set NVIDIA's construction is about: their
                    spatial energy function couples samples across the entire
                    slice, and the published frequency properties are those of a
                    single global threshold. The 8x8 execution tiling is then
                    purely an implementation detail that looks up an
                    already-decided mask, and per-tile counts are free to vary
                    (measured 16 +/- 1.4 on the asset at rho=0.25).

  tile_rank         Rank within each 8x8 tile and keep exactly k per tile. This
                    is a DIFFERENT sampling process — the effective threshold
                    varies from tile to tile — and it measurably degrades both
                    ends of the comparison: on the asset it costs ~30% of the
                    low-frequency suppression, and it makes the white-noise
                    control ~24% bluer than true white by forcing 16/tile on a
                    process whose natural tile counts span 5..31. Kept as a
                    legitimate ablation, not as the faithful reading.

The 1/rho unbiasing weight is UNIFORM in both modes (see stbnWeight), not
per-tile. A global threshold selects each pixel with probability rho, so 1/rho is
the correct importance weight for the image sum; a per-tile TILE_AREA/count
weight would re-impose exactly the tile-grid structure the global mode removes.

The slice index is a camera's VISIT count plus a per-camera phase, not the
iteration - see TrainingLoop._stbn_variables.

GPU layout (:func:`pack_selection`), one uint per voxel, block-major:

    packed[block_base + i] =  slot(pixel i)            // bits 0..7, 0xFF = none
                            | compacted(i) << 8        // i-th selected pixel
                            | count        << 16       // selected in this tile
    block_base = ((slice * blocks_y + by) * blocks_x + bx) * TILE_AREA

The low field answers "is pixel i supervised?" for the loss kernels with no
notion of a threshold; the middle answers "which pixel is the i-th selected
one?" so the backward walks a compacted list; the high field is that list's
length. Because the table carries the SET rather than a ranking, the shader
cannot tell which selection mode produced it — which is the point.

Full design and the measured density/quality trade-off: docs/stbn_pixel_sampling.md
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy

from trainer.contracts.codegen import TILE_AREA, TILE_HEIGHT, TILE_WIDTH

SOURCE_NVIDIA = "nvidia"
SOURCE_WHITE = "white"
SOURCE_FILE = "file"
SOURCES = (SOURCE_NVIDIA, SOURCE_WHITE, SOURCE_FILE)

# NVIDIA's scalar STBN asset shape, and the default for every other source so
# masks stay comparable across sources. The volume is toroidal in x, y AND t
# (their generator wraps), which is what lets us tile it across the screen and
# cycle the slice index without seams.
NVIDIA_WIDTH = 128
NVIDIA_HEIGHT = 128
NVIDIA_SLICES = 64
NVIDIA_VOLUME_NAME = "stbn_scalar_2Dx1Dx1D_128x128x64.npy"

_FETCH_HINT = (
    "run `python scripts/data/download_stbn.py` to fetch it from "
    "https://github.com/NVIDIA-RTX/STBN (NON-COMMERCIAL license — see the "
    "script's docstring), or point --stbn-source file --stbn-file at your own "
    "scalar STBN volume")

SELECT_GLOBAL = "global"
SELECT_TILE_RANK = "tile_rank"
SELECT_MODES = (SELECT_GLOBAL, SELECT_TILE_RANK)

# GPU table layout, one uint32 per voxel, block-major (see the module docstring
# for the index formula). Three 8-bit fields, all that a 64-pixel tile needs:
#
#   bits  0..7   slot of this pixel in its tile's selected list, or SLOT_NONE
#   bits  8..15  pixel-in-tile index of the slot-i selected pixel (i < count)
#   bits 16..23  number of selected pixels in this tile, replicated per voxel
#
# The count is replicated rather than kept in a side buffer because the bits are
# free and it costs the reader one fetch from a cache line it already touches.
# Replicating it is what lets the SHADER be selection-mode agnostic: it reads a
# compacted list and a length, and never needs to know how the set was chosen.
_SLOT_MASK = 0xFF
_SLOT_NONE = 0xFF                 # 0..63 are valid slots, so 0xFF is free
_COMPACT_SHIFT = 8
_COMPACT_MASK = 0xFF
_COUNT_SHIFT = 16
_COUNT_MASK = 0xFF


def default_cache_dir() -> Path:
    """Keep downloads in the checkout or working directory, outside installed code."""
    root = Path(__file__).resolve().parents[4]
    return (root if (root / "pyproject.toml").is_file() else Path.cwd()) / ".stbn"


def nvidia_volume_path(cache_dir: Path | str | None = None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return root / "assets" / NVIDIA_VOLUME_NAME


@dataclass(frozen=True)
class StbnMaskSpec:
    """Everything that determines the mask table's CONTENT.

    Two specs that compare equal produce byte-identical tables, which is what
    makes :func:`load_or_build`'s disk cache sound: ``cache_key`` is derived
    from every field below (plus, for the volume-backed sources, the identity of
    the volume itself) and nothing else feeds the table.
    """

    width: int = NVIDIA_WIDTH
    height: int = NVIDIA_HEIGHT
    slices: int = NVIDIA_SLICES
    source: str = SOURCE_NVIDIA
    # Which pixels a density selects.
    #
    #   "global"     threshold the WHOLE slice: keep the lowest
    #                round(fraction * width * height) values. This is the point
    #                set NVIDIA's construction is about — the spatial energy
    #                function couples samples across the entire slice, and their
    #                published frequency properties are those of a single global
    #                threshold. The 8x8 execution tiling is then purely an
    #                implementation detail: it looks up an already-decided mask.
    #
    #   "tile_rank"  rank within each 8x8 tile and keep exactly k per tile.
    #                Convenient (uniform per-tile work, exact 1/rho per tile) but
    #                it is a DIFFERENT sampling process: the effective threshold
    #                varies per tile, which lets pixels on either side of a tile
    #                boundary both survive and cluster in a way the global
    #                ranking specifically avoids. Measured on the NVIDIA asset at
    #                rho=0.25 it costs ~30% of the low-frequency suppression
    #                (0.0033 -> 0.0047), and it makes the white-noise CONTROL
    #                ~24% bluer than true white (0.048 -> 0.037) by forcing
    #                16/tile on a process whose natural tile counts span 5..31 —
    #                so it compresses the blue-vs-white contrast from both ends.
    #                Kept because it is a legitimate ablation ("STBN-ranked
    #                per-tile stratified sampling"), not because it is the
    #                faithful reading of the asset.
    select_mode: str = SELECT_GLOBAL
    # The density the table is built FOR. Only "global" needs it — a global
    # threshold decides membership over the whole slice, so the selected set (and
    # hence the compacted list and the per-tile counts) depends on the density.
    # "tile_rank" ignores it and derives k from the runtime budget.
    fraction: float = 0.25
    # Only meaningful for source="white".
    seed: int = 0
    # source="file": the scalar volume to rank per tile. source="nvidia":
    # optional override of the cached asset location (blank = default cache).
    path: str = ""
    # Tile shape the stratification is built around. Defaults to the execution
    # contract's rasterizer tile; a spec whose tile disagreed with the shaders
    # would select the wrong pixel counts, so :meth:`validate` pins it.
    tile_width: int = TILE_WIDTH
    tile_height: int = TILE_HEIGHT

    @property
    def tile_area(self) -> int:
        return self.tile_width * self.tile_height

    @property
    def blocks_x(self) -> int:
        return self.width // self.tile_width

    @property
    def blocks_y(self) -> int:
        return self.height // self.tile_height

    @property
    def voxels(self) -> int:
        return self.slices * self.height * self.width

    @property
    def selected_per_slice(self) -> int:
        """Pixels a ``global`` threshold keeps per slice.

        Quantized to a whole number of per-tile budgets — ``pixels_per_tile(rho) *
        blocks`` — rather than to ``round(rho * width * height)``, for two reasons
        that are both correctness, not tidiness:

        1. The 1/rho unbiasing WEIGHT the shaders apply is derived from the
           per-tile budget (``stbnWeight``), which has 1/TILE_AREA resolution. If
           the selected count did not agree with it, the weight would be wrong by
           the quantization error — at rho=0.10 that is round(6.4)=6 versus 6.4,
           a +6.6% inflation of every gradient. Adam absorbs a uniform scale, but
           the densifier's ``--grad-thresh`` is an ABSOLUTE threshold, so an
           inflated gradient densifies more eagerly. Silent, and it would have
           shown up as "sparse supervision grows more Gaussians".
        2. It makes the two selection modes exactly density-matched at every
           density, so the ``global`` vs ``tile_rank`` ablation compares the
           ARRANGEMENT of a fixed number of pixels rather than confounding it with
           a slightly different pixel count.

        Never 0 and never the whole slice: an empty mask would give every splat
        zero gradient, and a full one is the mask-off path.
        """
        total = self.height * self.width
        if self.fraction >= 1.0:
            return total
        per_tile = pixels_per_tile(float(self.fraction), self.tile_area)
        return int(min(total, max(1, per_tile * self.blocks_x * self.blocks_y)))

    def validate(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(
                f"unknown STBN source {self.source!r}; expected one of {list(SOURCES)}")
        if self.select_mode not in SELECT_MODES:
            raise ValueError(
                f"unknown STBN select_mode {self.select_mode!r}; expected one of "
                f"{list(SELECT_MODES)}")
        if self.tile_width != TILE_WIDTH or self.tile_height != TILE_HEIGHT:
            raise ValueError(
                f"STBN tile {self.tile_width}x{self.tile_height} must match the "
                f"rasterizer tile {TILE_WIDTH}x{TILE_HEIGHT} (execution contract) "
                f"— the shaders index the table by tile.")
        if self.width % self.tile_width or self.height % self.tile_height:
            raise ValueError(
                f"STBN texture {self.width}x{self.height} must be a whole number "
                f"of {self.tile_width}x{self.tile_height} tiles so the mask tiles "
                f"the screen without straddling tile boundaries.")
        if self.slices < 1:
            raise ValueError(f"STBN needs >= 1 temporal slice, got {self.slices}.")
        if self.blocks_x < 1 or self.blocks_y < 1:
            raise ValueError(
                f"STBN texture {self.width}x{self.height} is smaller than one tile.")
        if self.source == SOURCE_FILE and not self.path:
            raise ValueError("--stbn-source file requires --stbn-file <path.npy>.")
        if self.source == SOURCE_NVIDIA and (
                self.width != NVIDIA_WIDTH or self.height != NVIDIA_HEIGHT
                or self.slices != NVIDIA_SLICES):
            # Cropping the asset would break its toroidal wrap in x/y/t, which
            # is exactly what lets it tile the screen and cycle seamlessly.
            raise ValueError(
                f"--stbn-source nvidia is fixed at {NVIDIA_WIDTH}x{NVIDIA_HEIGHT}"
                f"x{NVIDIA_SLICES} (the published asset); got "
                f"{self.width}x{self.height}x{self.slices}. Cropping it would "
                f"break the mask's toroidal wrap.")

    def resolved_path(self) -> Path:
        """Where this spec's source volume lives (nvidia / file sources)."""
        if self.path:
            return Path(self.path)
        return nvidia_volume_path()

    def cache_key(self) -> str:
        """Filename stem identifying this spec's table on disk.

        Includes the selection mode AND the density, for BOTH modes. The packed
        table carries the selected SET (not a ranking), so it is density-specific
        in every mode — unlike the old rank-based table, where density was a
        runtime comparison and one table served all densities.

        This bit us: omitting the density for ``tile_rank`` meant a table built at
        rho=0.09375 was served for a rho=0.25 run. The mask was 9.4% dense while
        the shaders applied the 25% weight, so every gradient was 2.67x too small,
        densification starved, and the run lost ~0.5 dB — with nothing anywhere
        reporting an error.
        """
        if self.source == SOURCE_WHITE:
            tag = f"white_s{self.seed}"
        else:
            # The volume's own content is the source of truth; include its
            # identity so two different assets can't share a cache entry.
            source_path = self.resolved_path()
            stat = source_path.stat()
            tag = (f"{self.source}_{source_path.stem}"
                   f"_{stat.st_size}_{int(stat.st_mtime)}")
        return (f"stbn_{self.width}x{self.height}x{self.slices}"
                f"_t{self.tile_width}x{self.tile_height}"
                f"_{self.select_mode}{self.selected_per_slice}_{tag}")


# ----------------------------------------------------------------------------
# Rank-volume construction
# ----------------------------------------------------------------------------

def _block_voxel_index(spec: StbnMaskSpec) -> np.ndarray:
    """``(slices, blocks_y, blocks_x, tile_area) -> flat voxel index``.

    The last axis is the pixel-in-tile index the shaders use
    (``p = row * TILE_WIDTH + col``), so a permutation along it IS the per-tile
    rank assignment. One array expresses both the GPU table's block-major layout
    and the "operate per tile" reshape every function below needs.
    """
    t, h, w = spec.slices, spec.height, spec.width
    flat = np.arange(spec.voxels, dtype=np.int64).reshape(t, h, w)
    tiled = flat.reshape(t, spec.blocks_y, spec.tile_height,
                         spec.blocks_x, spec.tile_width)
    # (t, by, th, bx, tw) -> (t, by, bx, th, tw) -> (t, by, bx, th*tw)
    return np.ascontiguousarray(tiled.transpose(0, 1, 3, 2, 4)).reshape(
        t, spec.blocks_y, spec.blocks_x, spec.tile_area)


def _ranks_from_order(order: np.ndarray, index: np.ndarray,
                      spec: StbnMaskSpec) -> np.ndarray:
    """Scatter a per-block ordering (``order[..., j]`` = the j-th smallest
    pixel) back into a voxel-indexed rank volume."""
    ranks = np.empty(spec.voxels, dtype=np.uint16)
    ranks[np.take_along_axis(index, order, axis=-1).reshape(-1)] = np.broadcast_to(
        np.arange(spec.tile_area, dtype=np.uint16), index.shape).reshape(-1)
    return ranks.reshape(spec.slices, spec.height, spec.width)


def rank_volume_from_scalar(volume: np.ndarray, spec: StbnMaskSpec) -> np.ndarray:
    """Rank a scalar STBN volume per tile.

    Within each tile of each slice the values are replaced by their order, which
    preserves the mask's arrangement while making the per-tile selected count
    exact at every density. Larger volumes are cropped; the spec's dims are
    authoritative (and :meth:`StbnMaskSpec.validate` forbids cropping the
    ``nvidia`` asset, whose wrap-around would not survive it).
    """
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(
            f"STBN scalar volume must be (slices, height, width), got {array.shape}")
    if (array.shape[0] < spec.slices or array.shape[1] < spec.height
            or array.shape[2] < spec.width):
        raise ValueError(
            f"STBN scalar volume {array.shape} is smaller than the requested "
            f"({spec.slices}, {spec.height}, {spec.width})")
    array = array[:spec.slices, :spec.height, :spec.width].astype(np.float64)

    index = _block_voxel_index(spec)
    values = array.reshape(-1)[index]                       # (t, by, bx, area)
    # Stable sort so equal 8-bit levels (NVIDIA's asset quantizes 64 pixels per
    # level per slice, so ties inside a tile do happen) break deterministically
    # by pixel index rather than by sort implementation.
    return _ranks_from_order(np.argsort(values, axis=-1, kind="stable"), index, spec)


def _white_ranks(spec: StbnMaskSpec) -> np.ndarray:
    """Independent random per-tile permutations: the control that isolates what
    the blue-noise ARRANGEMENT buys. Still exactly k pixels per tile, so it
    differs from the STBN mask ONLY in arrangement — not in count, not in
    normalization, not in any downstream code path."""
    rng = np.random.default_rng(spec.seed)
    index = _block_voxel_index(spec)
    return _ranks_from_order(np.argsort(rng.random(index.shape), axis=-1),
                             index, spec)


def load_scalar_volume(spec: StbnMaskSpec) -> np.ndarray:
    """The raw scalar volume behind a ``nvidia`` / ``file`` spec.

    Separate from ranking so a caller (or a test) can inspect the mask as
    published, before tile stratification reorders it.
    """
    path = spec.resolved_path()
    if not path.is_file():
        if spec.source == SOURCE_NVIDIA:
            raise FileNotFoundError(
                f"NVIDIA scalar STBN asset not found at {path}. {_FETCH_HINT}.")
        raise FileNotFoundError(f"--stbn-file {path} does not exist.")
    return np.load(path)


def build_rank_volume(spec: StbnMaskSpec) -> np.ndarray:
    """``(slices, height, width)`` uint16 per-tile ranks; each tile holds a
    permutation of ``0..tile_area-1``.

    This is the ``tile_rank`` construction. It remains the definition of that
    mode and is still what the density/quality metrics below characterize; the
    ``global`` mode goes through :func:`global_selected_mask` instead.
    """
    spec.validate()
    if spec.source == SOURCE_WHITE:
        return _white_ranks(spec)
    return rank_volume_from_scalar(load_scalar_volume(spec), spec)


def _scalar_volume(spec: StbnMaskSpec) -> np.ndarray:
    """The scalar field a spec ranks, whatever its source.

    White noise is generated at full slice resolution here (rather than as
    per-tile permutations) so that a global threshold sees a genuine independent
    field — which is the whole point of the control.
    """
    if spec.source == SOURCE_WHITE:
        rng = np.random.default_rng(spec.seed)
        return rng.random((spec.slices, spec.height, spec.width))
    array = np.asarray(load_scalar_volume(spec), dtype=np.float64)
    return array[:spec.slices, :spec.height, :spec.width]


def scalar_field(spec: StbnMaskSpec) -> np.ndarray:
    """Public view of the scalar field a spec ranks — see :func:`_scalar_volume`.

    Exists for the adaptive sampler, which needs the volume as a UNIFORM variable
    rather than as a selected set: it thresholds the field against a per-pixel
    probability instead of a global density, so it cannot go through
    :func:`global_selected_mask`. Same volume, same crop, same white-noise
    stream — sharing this function is what guarantees the two paths threshold the
    identical field.
    """
    return _scalar_volume(spec)


def global_selected_mask(spec: StbnMaskSpec) -> np.ndarray:
    """``(slices, height, width)`` bool: the lowest ``fraction`` of EACH SLICE.

    One threshold per slice, applied across the whole slice — the point set
    NVIDIA's spatial energy function was optimized for. Implemented as a rank
    (argsort) rather than a literal value comparison so the kept count is exact
    despite the asset's 8-bit quantization: at 128x128 there are only 256 levels
    for 16384 pixels, so a value threshold would land mid-level and overshoot or
    undershoot by tens of pixels. A stable sort breaks ties by flat pixel index,
    which keeps the table reproducible.

    Note what this does NOT do: it does not constrain how many selected pixels
    land in any given 8x8 tile. On the NVIDIA asset at rho=0.25 the per-tile
    count comes out 16 +/- 1.4 (range 11..21); on white noise, 16 +/- 3.4 (range
    5..31). That variation is the honest consequence of sampling a global field,
    and the shader handles it by reading a per-tile count.
    """
    spec.validate()
    volume = _scalar_volume(spec)
    keep = spec.selected_per_slice
    total = spec.height * spec.width
    mask = np.zeros((spec.slices, total), dtype=bool)
    for index in range(spec.slices):
        order = np.argsort(volume[index].reshape(-1), kind="stable")
        mask[index, order[:keep]] = True
    return mask.reshape(spec.slices, spec.height, spec.width)


def selected_volume(spec: StbnMaskSpec,
                    pixels_per_tile_budget: int | None = None) -> np.ndarray:
    """The boolean selection a spec realizes, for either mode.

    The single place that answers "which pixels does this configuration
    supervise?", so the GPU table, the tests and the sampling diagnostics cannot
    drift from each other.
    """
    spec.validate()
    if spec.select_mode == SELECT_GLOBAL:
        return global_selected_mask(spec)
    budget = (pixels_per_tile_budget if pixels_per_tile_budget is not None
              else pixels_per_tile(spec.fraction, spec.tile_area))
    return selected_mask(build_rank_volume(spec), budget)


# ----------------------------------------------------------------------------
# GPU table packing + cache
# ----------------------------------------------------------------------------

def pack_selection(selected: np.ndarray, spec: StbnMaskSpec) -> np.ndarray:
    """Pack a boolean selection into the block-major ``uint32`` table.

    ``packed[block_base + i] = slot(i) | compacted(i) << 8 | count << 16`` where

      * ``slot(i)`` is pixel i's position in its tile's selected list, or
        ``_SLOT_NONE`` when it is not selected — so "is this pixel supervised?"
        is one fetch and one compare, with no notion of a threshold;
      * ``compacted(i)`` is the pixel-in-tile index of the i-th selected pixel,
        the list the backward walks;
      * ``count`` is how many of the tile's 64 pixels are selected, replicated
        across the block so the backward's loop bound is one fetch.

    Deliberately takes a SELECTION rather than a rank volume: that is what makes
    the layout mode-agnostic. Whether the set was chosen by a global threshold or
    by a per-tile argsort is settled before this function, and the shader is
    unable to tell the difference. Slots are assigned in ascending pixel-in-tile
    order, which is arbitrary but fixed — the backward sums over the tile's
    selected pixels and the order does not affect the result.
    """
    index = _block_voxel_index(spec)
    flags = selected.reshape(-1)[index]                      # (t, by, bx, area)
    counts = flags.sum(axis=-1).astype(np.uint32)            # (t, by, bx)
    if int(counts.max(initial=0)) > _COUNT_MASK:
        raise ValueError(
            f"per-tile selected count {int(counts.max())} exceeds the "
            f"{_COUNT_MASK} the packed layout can hold")

    # Slot of each selected pixel = its rank among the tile's selected pixels in
    # ascending pixel-in-tile order; cumsum-1 over the flags gives exactly that.
    slots = np.cumsum(flags, axis=-1) - 1
    slots = np.where(flags, slots, _SLOT_NONE).astype(np.uint32)

    # Compacted list: the pixel indices of the selected pixels, in the same
    # order, left-packed. argsort on (not selected) is stable, so the selected
    # indices come first in ascending order.
    order = np.argsort(~flags, axis=-1, kind="stable").astype(np.uint32)

    packed = (slots
              | (order << _COMPACT_SHIFT)
              | (counts[..., None].astype(np.uint32) << _COUNT_SHIFT))
    return np.ascontiguousarray(packed.reshape(-1))


def unpack_table(packed: np.ndarray, spec: StbnMaskSpec
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_selection`, block-major ``(slots, compacted,
    counts)``.

    Test affordance: lets a test assert the GPU-VISIBLE bytes rather than the
    array they were derived from, which is where a packing bug would otherwise
    hide until it showed up as a quietly wrong gradient.
    """
    blocks = packed.reshape(spec.slices, spec.blocks_y, spec.blocks_x,
                           spec.tile_area)
    return ((blocks & _SLOT_MASK).astype(np.uint16),
            ((blocks >> _COMPACT_SHIFT) & _COMPACT_MASK).astype(np.uint16),
            ((blocks >> _COUNT_SHIFT) & _COUNT_MASK).astype(np.uint16)[..., 0])


def build_table(spec: StbnMaskSpec) -> np.ndarray:
    """The packed GPU table for ``spec``, for either selection mode."""
    return pack_selection(selected_volume(spec), spec)


def _table_matches(packed: np.ndarray, spec: StbnMaskSpec) -> bool:
    """Does a cached table actually realize ``spec``'s density?

    Shape and dtype are far too weak a check: every table of a given volume size
    has the same shape and dtype regardless of its density or selection mode, so a
    key collision serves a silently wrong mask. That happened — a 9.375%-dense
    table was served for a 25% run, and the only symptom was 0.5 dB of lost
    quality. Verifying the count the shaders will actually read makes such a
    collision a rebuild rather than a wrong result.
    """
    try:
        counts = ((packed >> _COUNT_SHIFT) & _COUNT_MASK).reshape(
            spec.slices, spec.blocks_y, spec.blocks_x, spec.tile_area)[..., 0]
        return int(counts.sum(axis=(1, 2)).max()) == spec.selected_per_slice
    except Exception:                                      # noqa: BLE001
        return False


def load_or_build(spec: StbnMaskSpec, cache_dir: Path | str | None) -> np.ndarray:
    """The packed table for ``spec``, from disk when it is already there.

    Ranking the 128x128x64 asset takes a couple of seconds — fine once, wasteful
    on every run of a sweep — and the result is a pure function of the spec, so
    it caches cleanly. A corrupt or stale entry is rebuilt rather than raising:
    the cache is an optimization, never a correctness dependency.
    """
    spec.validate()
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{spec.cache_key()}.npy"
        if path.is_file():
            try:
                cached = np.load(path)
                if (cached.shape == (spec.voxels,)
                        and cached.dtype == np.uint32
                        and _table_matches(cached, spec)):
                    return cached
                print(f"[stbn] rebuilding {path.name}: cached table does not "
                      f"match the requested spec", flush=True)
            except (OSError, ValueError):
                pass                      # fall through and rebuild

    packed = build_table(spec)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, packed)
        except OSError:
            pass                          # a read-only cache dir is not fatal
    return packed


# ----------------------------------------------------------------------------
# GPU resource
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class StbnMask:
    """The uploaded table plus the dims the shaders index it with.

    Built once per pipeline (the table is a few MB and never changes); only the
    per-iteration slice/budget vary, which is why :meth:`shader_kwargs` takes
    them as arguments rather than storing them.
    """

    table: Any                    # spy.Buffer holding the packed uint32 table
    dims: tuple[int, int]         # (width, height) in pixels
    slices: int

    def shader_kwargs(self, slice_index: int, pixels_per_tile: int) -> dict:
        """The four ``stbn_*`` parameters every masked kernel takes."""
        return {
            "stbn_table": self.table,
            "stbn_dims": spy.uint2(self.dims[0], self.dims[1]),
            "stbn_slice": int(slice_index) % max(self.slices, 1),
            "stbn_tile_pixels": int(pixels_per_tile),
        }


def upload_mask(device: Any, spec: StbnMaskSpec,
                cache_dir: Path | str | None = None) -> StbnMask:
    """Build (or load) the table for ``spec`` and upload it to the device."""
    packed = load_or_build(spec, cache_dir)
    buffer = device.create_buffer(
        size=packed.nbytes,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        label="stbn_table",
    )
    buffer.copy_from_numpy(np.ascontiguousarray(packed).view(np.uint8))
    return StbnMask(table=buffer, dims=(spec.width, spec.height),
                    slices=spec.slices)


def upload_packed_table(device: Any, packed: np.ndarray,
                        spec: StbnMaskSpec) -> StbnMask:
    """Upload an ALREADY-PACKED table, bypassing the spec-driven builders.

    For diagnostics that need a mask no ``StbnMaskSpec`` describes — e.g. the
    temporal-only control in the gradient study, whose per-pixel schedule is
    constructed directly rather than ranked from a scalar volume. Training never
    takes this path; it exists so an evaluation can define a mask without
    inventing a new production source.
    """
    if packed.shape != (spec.voxels,) or packed.dtype != np.uint32:
        raise ValueError(
            f"packed table must be uint32[{spec.voxels}], got "
            f"{packed.dtype}{packed.shape}")
    buffer = device.create_buffer(
        size=packed.nbytes,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        label="stbn_table_custom",
    )
    buffer.copy_from_numpy(np.ascontiguousarray(packed).view(np.uint8))
    return StbnMask(table=buffer, dims=(spec.width, spec.height),
                    slices=spec.slices)


def upload_disabled_mask(device: Any) -> StbnMask:
    """A 1-element placeholder table for runs with the mask off.

    The kernels take the ``stbnActive`` early-out before ever reading the table,
    but a declared shader parameter still has to be BOUND — so "off" needs a
    resource, just not a 4 MB one.
    """
    buffer = device.create_buffer(
        size=4,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        label="stbn_table_disabled",
    )
    buffer.copy_from_numpy(np.zeros(1, dtype=np.uint32).view(np.uint8))
    # Non-zero dims so the shaders' modulo stays defined even if a future reader
    # forgets the active check; slices=1 keeps the slice index harmless.
    return StbnMask(table=buffer, dims=(TILE_WIDTH, TILE_HEIGHT), slices=1)


def pixels_per_tile(fraction: float, tile_area: int = TILE_AREA) -> int:
    if fraction >= 1.0:
        return tile_area
    return int(min(tile_area, max(1, round(float(fraction) * tile_area))))


def selected_mask(ranks: np.ndarray, pixels_per_tile: int) -> np.ndarray:
    """Boolean selection at a per-tile budget — the shaders' ``rank < k``."""
    return ranks < pixels_per_tile


def tile_counts(ranks: np.ndarray, pixels_per_tile: int,
                spec: StbnMaskSpec) -> np.ndarray:
    """Per (slice, tile) selected-pixel count. Constant by construction; a test
    asserts that, because it is what makes the 1/rho reweighting exact."""
    index = _block_voxel_index(spec)
    return selected_mask(ranks.reshape(-1)[index], pixels_per_tile).sum(axis=-1)


def low_frequency_power(mask: np.ndarray, cutoff: float = 0.25) -> float:
    """Fraction of a 2D mask's spectral power (DC removed) below ``cutoff`` of
    Nyquist. Blue noise pushes energy AWAY from low frequencies, so lower is
    bluer; white noise sits near the band's area fraction."""
    centered = mask.astype(np.float64) - mask.mean()
    power = np.abs(np.fft.fft2(centered)) ** 2
    fy = np.fft.fftfreq(mask.shape[0])[:, None]
    fx = np.fft.fftfreq(mask.shape[1])[None, :]
    radius = np.sqrt((fy / 0.5) ** 2 + (fx / 0.5) ** 2)
    total = power.sum()
    if total <= 0.0:
        return 0.0
    return float(power[radius <= cutoff].sum() / total)


def temporal_adjacency(ranks: np.ndarray, pixels_per_tile: int) -> float:
    """Mean per-pixel count of consecutive slices that are BOTH selected.

    For an independent mask of density rho this is ``(slices-1) * rho**2``; a
    temporally well-distributed mask spreads a pixel's visits out and scores
    below that. This is the statistic that says whether the temporal axis is
    doing anything — i.e. whether a spatiotemporal mask is buying anything over
    one static blue-noise mask reused on every visit.
    """
    mask = selected_mask(ranks, pixels_per_tile)
    if mask.shape[0] < 2:
        return 0.0
    return float((mask[:-1] & mask[1:]).sum(axis=0).mean())
