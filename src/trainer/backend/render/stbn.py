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

The mask is re-ranked WITHIN each tile, so every tile of every slice holds a
permutation of 0..TILE_AREA-1 and "rank < k" selects exactly k pixels per tile.
That makes the 1/rho weight exact and the selected sets nested across densities.
The slice index is a camera's VISIT count plus a per-camera phase, not the
iteration - see TrainingLoop._stbn_variables.

GPU layout (:func:`pack_table`), one uint per voxel, block-major:

    packed[block_base + i] = rank(pixel i) | (pixel of rank i) << 16
    block_base = ((slice * blocks_y + by) * blocks_x + bx) * TILE_AREA

Low half answers "is pixel i selected?" for the loss kernels; high half answers
"which pixel is the j-th selected one?" so the backward walks a compacted list.

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

# Ranks are packed into the low 16 bits, the inverse permutation into the high
# 16. TILE_AREA is 64 today; the split leaves room for far larger tiles.
_RANK_MASK = 0xFFFF
_INVERSE_SHIFT = 16


def default_cache_dir() -> Path:
    """``<repo>/.stbn`` — the gitignored STBN cache the downloader writes to."""
    return Path(__file__).resolve().parents[4] / ".stbn"


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

    def validate(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(
                f"unknown STBN source {self.source!r}; expected one of {list(SOURCES)}")
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
        """Filename stem identifying this spec's table on disk."""
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
                f"_t{self.tile_width}x{self.tile_height}_{tag}")


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
    """``(slices, height, width)`` uint16 ranks; each tile holds a permutation
    of ``0..tile_area-1``."""
    spec.validate()
    if spec.source == SOURCE_WHITE:
        return _white_ranks(spec)
    return rank_volume_from_scalar(load_scalar_volume(spec), spec)


# ----------------------------------------------------------------------------
# GPU table packing + cache
# ----------------------------------------------------------------------------

def pack_table(ranks: np.ndarray, spec: StbnMaskSpec) -> np.ndarray:
    """Pack a rank volume into the block-major ``uint32`` table the shaders read.

    ``packed[block_base + i] = rank(pixel i) | (pixel of rank i) << 16`` — the
    permutation and its inverse, one fetch each. See the module docstring for
    the index formula and shaders/common/stbn.slang for the reader.
    """
    index = _block_voxel_index(spec)
    per_block = ranks.reshape(-1)[index].astype(np.uint32)   # (t, by, bx, area)
    inverse = np.argsort(per_block, axis=-1).astype(np.uint32)
    return np.ascontiguousarray((per_block | (inverse << _INVERSE_SHIFT)).reshape(-1))


def unpack_table(packed: np.ndarray,
                 spec: StbnMaskSpec) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_table`, block-major ``(ranks, inverse)``. Test
    affordance: lets a test assert the GPU-visible bytes, not just the source
    array they were derived from."""
    blocks = packed.reshape(spec.slices, spec.blocks_y, spec.blocks_x, spec.tile_area)
    return ((blocks & _RANK_MASK).astype(np.uint16),
            (blocks >> _INVERSE_SHIFT).astype(np.uint16))


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
                if cached.shape == (spec.voxels,) and cached.dtype == np.uint32:
                    return cached
            except (OSError, ValueError):
                pass                      # fall through and rebuild

    packed = pack_table(build_rank_volume(spec), spec)
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


# ----------------------------------------------------------------------------
# Density resolution + quality metrics (used by the schedule and the tests)
# ----------------------------------------------------------------------------

def pixels_per_tile(fraction: float, tile_area: int = TILE_AREA) -> int:
    """The integer per-tile budget k realizing a requested pixel fraction.

    Rounds to the nearest achievable density and never returns 0: a tile that
    contributed nothing at all would leave its splats with no gradient
    whatsoever, a qualitatively worse failure than a coarse budget.
    ``fraction >= 1`` returns the full tile, i.e. the mask off.
    """
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
