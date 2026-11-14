# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Spatiotemporal blue-noise (STBN) tables: which fraction rho of the pixels a sparse step
supervises (reweighted by 1/rho). Design and measurements: docs/stbn_pixel_sampling.md."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trainer.backend.compute.execution import TILE_AREA, TILE_HEIGHT, TILE_WIDTH

SOURCE_NVIDIA = "nvidia"    # github.com/NVIDIA-RTX/STBN scalar set (non-commercial, never vendored)
SOURCE_WHITE = "white"      # independent noise: the control
SOURCE_FILE = "file"        # any (slices, height, width) scalar volume
SOURCES = (SOURCE_NVIDIA, SOURCE_WHITE, SOURCE_FILE)

SELECT_GLOBAL = "global"
SELECT_TILE_RANK = "tile_rank"
SELECT_MODES = (SELECT_GLOBAL, SELECT_TILE_RANK)

# The published asset; it wraps in x, y and t, so it tiles the screen seamlessly.
NVIDIA_WIDTH = 128
NVIDIA_HEIGHT = 128
NVIDIA_SLICES = 64
NVIDIA_VOLUME_NAME = "stbn_scalar_2Dx1Dx1D_128x128x64.npy"

_FETCH_HINT = (
    "run `python scripts/data/download_stbn.py` to fetch it from "
    "https://github.com/NVIDIA-RTX/STBN (NON-COMMERCIAL license — see the "
    "script's docstring), or point --stbn-source file --stbn-file at your own "
    "scalar STBN volume")

# Packed table: one uint32 per voxel, block-major, mirrored by shaders/common/stbn.slang.
#   bits  0..7   slot of this pixel in its tile's selected list, SLOT_NONE if unselected
#   bits  8..15  pixel-in-tile index of the slot-i selected pixel
#   bits 16..23  selected pixels in this tile (replicated per voxel)
SLOT_MASK = 0xFF
SLOT_NONE = 0xFF
COMPACT_SHIFT = 8
COMPACT_MASK = 0xFF
COUNT_SHIFT = 16
COUNT_MASK = 0xFF


def default_cache_dir() -> Path:
    """Downloads and built tables live in the checkout (or the working directory)."""
    root = Path(__file__).resolve().parents[4]
    return (root if (root / "pyproject.toml").is_file() else Path.cwd()) / ".stbn"


def nvidia_volume_path(cache_dir: Path | str | None = None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return root / "assets" / NVIDIA_VOLUME_NAME


def pixels_per_tile(fraction: float, tile_area: int = TILE_AREA) -> int:
    if fraction >= 1.0:
        return tile_area
    return int(min(tile_area, max(1, round(float(fraction) * tile_area))))


@dataclass(frozen=True)
class StbnSpec:
    """Everything that determines a table's content; equal specs build equal tables."""

    width: int = NVIDIA_WIDTH
    height: int = NVIDIA_HEIGHT
    slices: int = NVIDIA_SLICES
    source: str = SOURCE_NVIDIA
    select_mode: str = SELECT_GLOBAL
    # The density a table is built for (a selected set is density-specific).
    fraction: float = 0.25
    seed: int = 0               # white noise only
    path: str = ""              # file source, or an override of the cached nvidia asset

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(
                f"unknown STBN source {self.source!r}; expected one of {list(SOURCES)}")
        if self.select_mode not in SELECT_MODES:
            raise ValueError(
                f"unknown STBN select_mode {self.select_mode!r}; expected one of "
                f"{list(SELECT_MODES)}")
        if self.width % TILE_WIDTH or self.height % TILE_HEIGHT:
            raise ValueError(
                f"STBN texture {self.width}x{self.height} must be a whole number "
                f"of {TILE_WIDTH}x{TILE_HEIGHT} tiles so the mask tiles "
                f"the screen without straddling tile boundaries.")
        if self.slices < 1:
            raise ValueError(f"STBN needs >= 1 temporal slice, got {self.slices}.")
        if self.blocks_x < 1 or self.blocks_y < 1:
            raise ValueError(
                f"STBN texture {self.width}x{self.height} is smaller than one tile.")
        if self.source == SOURCE_FILE and not self.path:
            raise ValueError("--stbn-source file requires --stbn-file <path.npy>.")
        if self.source == SOURCE_NVIDIA and (
                self.width, self.height, self.slices) != (NVIDIA_WIDTH, NVIDIA_HEIGHT, NVIDIA_SLICES):
            raise ValueError(
                f"--stbn-source nvidia is fixed at {NVIDIA_WIDTH}x{NVIDIA_HEIGHT}"
                f"x{NVIDIA_SLICES} (the published asset); got "
                f"{self.width}x{self.height}x{self.slices}. Cropping it would "
                f"break the mask's toroidal wrap.")

    @property
    def tile_area(self) -> int:
        return TILE_AREA

    @property
    def blocks_x(self) -> int:
        return self.width // TILE_WIDTH

    @property
    def blocks_y(self) -> int:
        return self.height // TILE_HEIGHT

    @property
    def voxels(self) -> int:
        return self.slices * self.height * self.width

    @property
    def selected_per_slice(self) -> int:
        """Pixels a global threshold keeps per slice (never 0).

        A whole number of per-tile budgets, so the shaders' per-tile 1/rho weight matches the
        selected count exactly (a mismatch silently scales every gradient).
        """
        total = self.height * self.width
        if self.fraction >= 1.0:
            return total
        per_tile = pixels_per_tile(float(self.fraction))
        return int(min(total, max(1, per_tile * self.blocks_x * self.blocks_y)))

    def resolved_path(self) -> Path:
        return Path(self.path) if self.path else nvidia_volume_path()

    def cache_key(self) -> str:
        """Table filename stem: mode and density, since the table holds a selected set."""
        if self.source == SOURCE_WHITE:
            tag = f"white_s{self.seed}"
        else:
            source_path = self.resolved_path()
            stat = source_path.stat()
            tag = f"{self.source}_{source_path.stem}_{stat.st_size}_{int(stat.st_mtime)}"
        return (f"stbn_{self.width}x{self.height}x{self.slices}"
                f"_t{TILE_WIDTH}x{TILE_HEIGHT}"
                f"_{self.select_mode}{self.selected_per_slice}_{tag}")


def block_voxel_index(spec: StbnSpec) -> np.ndarray:
    """``(slices, blocks_y, blocks_x, tile_area)`` -> flat voxel index.

    The last axis is the shaders' pixel-in-tile index (``row * TILE_WIDTH + col``).
    """
    t = spec.slices
    flat = np.arange(spec.voxels, dtype=np.int64).reshape(
        t, spec.blocks_y, TILE_HEIGHT, spec.blocks_x, TILE_WIDTH)
    return np.ascontiguousarray(flat.transpose(0, 1, 3, 2, 4)).reshape(
        t, spec.blocks_y, spec.blocks_x, TILE_AREA)


def _ranks_from_order(order: np.ndarray, index: np.ndarray, spec: StbnSpec) -> np.ndarray:
    """Scatter per-tile orderings (``order[..., j]`` = j-th smallest) into a rank volume."""
    ranks = np.empty(spec.voxels, dtype=np.uint16)
    ranks[np.take_along_axis(index, order, axis=-1).reshape(-1)] = np.broadcast_to(
        np.arange(TILE_AREA, dtype=np.uint16), index.shape).reshape(-1)
    return ranks.reshape(spec.slices, spec.height, spec.width)


def load_scalar_volume(spec: StbnSpec) -> np.ndarray:
    """The raw volume behind an ``nvidia`` / ``file`` spec."""
    path = spec.resolved_path()
    if not path.is_file():
        if spec.source == SOURCE_NVIDIA:
            raise FileNotFoundError(
                f"NVIDIA scalar STBN asset not found at {path}. {_FETCH_HINT}.")
        raise FileNotFoundError(f"--stbn-file {path} does not exist.")
    return np.load(path)


def rank_volume_from_scalar(volume: np.ndarray, spec: StbnSpec) -> np.ndarray:
    """Per-tile ranks of a scalar volume (cropped to the spec)."""
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
    index = block_voxel_index(spec)
    # Stable: the asset's 8-bit levels tie inside a tile; break ties by pixel index.
    order = np.argsort(array.reshape(-1)[index], axis=-1, kind="stable")
    return _ranks_from_order(order, index, spec)


def build_rank_volume(spec: StbnSpec) -> np.ndarray:
    """``(slices, height, width)`` uint16 per-tile ranks (the tile_rank construction)."""
    if spec.source == SOURCE_WHITE:
        index = block_voxel_index(spec)
        rng = np.random.default_rng(spec.seed)
        return _ranks_from_order(np.argsort(rng.random(index.shape), axis=-1), index, spec)
    return rank_volume_from_scalar(load_scalar_volume(spec), spec)


def selected_mask(ranks: np.ndarray, pixels_per_tile: int) -> np.ndarray:
    """The shaders' ``rank < k`` at a per-tile budget."""
    return ranks < pixels_per_tile


def global_selected_mask(spec: StbnSpec) -> np.ndarray:
    """``(slices, height, width)`` bool: the lowest values of each whole slice.

    A stable argsort (not a value threshold) keeps the count exact despite the
    asset's 8-bit quantization. Per-tile counts then vary, which the shaders handle.
    """
    if spec.source == SOURCE_WHITE:
        volume = np.random.default_rng(spec.seed).random((spec.slices, spec.height, spec.width))
    else:
        volume = np.asarray(load_scalar_volume(spec), dtype=np.float64)
        volume = volume[:spec.slices, :spec.height, :spec.width]
    keep = spec.selected_per_slice
    mask = np.zeros((spec.slices, spec.height * spec.width), dtype=bool)
    for index in range(spec.slices):
        order = np.argsort(volume[index].reshape(-1), kind="stable")
        mask[index, order[:keep]] = True
    return mask.reshape(spec.slices, spec.height, spec.width)


def selected_volume(spec: StbnSpec, pixels_per_tile_budget: int | None = None) -> np.ndarray:
    """The boolean selection a spec realizes, for either mode."""
    if spec.select_mode == SELECT_GLOBAL:
        return global_selected_mask(spec)
    budget = (pixels_per_tile_budget if pixels_per_tile_budget is not None
              else pixels_per_tile(spec.fraction))
    return selected_mask(build_rank_volume(spec), budget)


def pack_selection(selected: np.ndarray, spec: StbnSpec) -> np.ndarray:
    """Pack a boolean selection into the block-major table (see the layout above).

    Slots follow ascending pixel-in-tile order, so the table carries only the set:
    the shaders cannot tell which selection mode produced it.
    """
    index = block_voxel_index(spec)
    flags = selected.reshape(-1)[index]
    counts = flags.sum(axis=-1).astype(np.uint32)
    if int(counts.max(initial=0)) > COUNT_MASK:
        raise ValueError(
            f"per-tile selected count {int(counts.max())} exceeds the "
            f"{COUNT_MASK} the packed layout can hold")
    slots = np.where(flags, np.cumsum(flags, axis=-1) - 1, SLOT_NONE).astype(np.uint32)
    order = np.argsort(~flags, axis=-1, kind="stable").astype(np.uint32)
    packed = slots | (order << COMPACT_SHIFT) | (counts[..., None] << COUNT_SHIFT)
    return np.ascontiguousarray(packed.reshape(-1))


def unpack_table(packed: np.ndarray, spec: StbnSpec
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_selection`: block-major ``(slots, compacted, counts)``."""
    blocks = packed.reshape(spec.slices, spec.blocks_y, spec.blocks_x, TILE_AREA)
    return ((blocks & SLOT_MASK).astype(np.uint16),
            ((blocks >> COMPACT_SHIFT) & COMPACT_MASK).astype(np.uint16),
            ((blocks >> COUNT_SHIFT) & COUNT_MASK).astype(np.uint16)[..., 0])


def build_table(spec: StbnSpec) -> np.ndarray:
    return pack_selection(selected_volume(spec), spec)


def _table_matches(packed: np.ndarray, spec: StbnSpec) -> bool:
    """Does a cached table hold the spec's density? (Shape alone cannot tell.)"""
    try:
        counts = ((packed >> COUNT_SHIFT) & COUNT_MASK).reshape(
            spec.slices, spec.blocks_y, spec.blocks_x, TILE_AREA)[..., 0]
        return int(counts.sum(axis=(1, 2)).max()) == spec.selected_per_slice
    except Exception:                                      # noqa: BLE001
        return False


def load_or_build(spec: StbnSpec, cache_dir: Path | str | None) -> np.ndarray:
    """The packed table, from the disk cache when valid; a bad entry is rebuilt."""
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{spec.cache_key()}.npy"
        if path.is_file():
            try:
                cached = np.load(path)
                if (cached.shape == (spec.voxels,) and cached.dtype == np.uint32
                        and _table_matches(cached, spec)):
                    return cached
                print(f"[stbn] rebuilding {path.name}: cached table does not "
                      f"match the requested spec", flush=True)
            except (OSError, ValueError):
                pass
    packed = build_table(spec)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, packed)
        except OSError:
            pass
    return packed


def low_frequency_power(mask: np.ndarray, cutoff: float = 0.25) -> float:
    """Share of a 2D mask's spectral power (DC removed) below ``cutoff`` of Nyquist.

    Lower is bluer; white noise sits near the band's area fraction.
    """
    centered = mask.astype(np.float64) - mask.mean()
    power = np.abs(np.fft.fft2(centered)) ** 2
    fy = np.fft.fftfreq(mask.shape[0])[:, None]
    fx = np.fft.fftfreq(mask.shape[1])[None, :]
    radius = np.sqrt((fy / 0.5) ** 2 + (fx / 0.5) ** 2)
    total = power.sum()
    if total <= 0.0:
        return 0.0
    return float(power[radius <= cutoff].sum() / total)
