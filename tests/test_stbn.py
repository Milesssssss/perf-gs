"""Properties of the STBN pixel-sampling mask (CPU only, no GPU).

Two jobs here.

The first is the INVARIANTS the rest of the system leans on. The exactly-k
per-tile count is what makes the loss's TILE_AREA/k reweighting exact rather than
an expectation, and the packed table's inverse permutation is what lets the
backward walk a compacted list; if either drifts, the gradient is quietly wrong
by a factor nobody chose.

The second is that the mask is actually BLUE. The whole argument for spending a
third-party asset and a tile-ranking pass is that STBN spreads the sampled pixels
in space and across a camera's repeat visits far better than random. That claim
should fail loudly if the asset changes or the ranking breaks, so it is pinned
here against the white-noise control (which shares the exact same per-tile count,
so the comparison isolates the ARRANGEMENT).
"""

from __future__ import annotations

import numpy as np
import pytest

from trainer.backend.render import stbn
from trainer.contracts.codegen import TILE_AREA, TILE_HEIGHT, TILE_WIDTH


def _asset_or_skip() -> stbn.StbnMaskSpec:
    spec = stbn.StbnMaskSpec()
    if not spec.resolved_path().is_file():
        pytest.skip("NVIDIA STBN asset absent; run scripts/data/download_stbn.py")
    return spec


def _white(seed: int = 0) -> stbn.StbnMaskSpec:
    return stbn.StbnMaskSpec(source=stbn.SOURCE_WHITE, seed=seed)


# ---------------------------------------------------------------- invariants

@pytest.mark.parametrize("source", [stbn.SOURCE_NVIDIA, stbn.SOURCE_WHITE])
def test_every_tile_of_every_slice_is_a_permutation(source: str) -> None:
    """The core invariant: ranks 0..TILE_AREA-1 exactly once per (slice, tile).

    Everything downstream follows from it — "rank < k" selecting exactly k
    pixels, the 1/rho weight being exact, and slots [0,k) of the inverse
    permutation being precisely the selected pixels.
    """
    spec = _asset_or_skip() if source == stbn.SOURCE_NVIDIA else _white()
    ranks = stbn.build_rank_volume(spec)

    assert ranks.shape == (spec.slices, spec.height, spec.width)
    per_tile = ranks.reshape(-1)[stbn._block_voxel_index(spec)]
    assert (np.sort(per_tile, axis=-1) == np.arange(spec.tile_area)).all()


@pytest.mark.parametrize("budget", [1, 6, 16, 32, 63])
def test_selected_count_is_exactly_the_budget_in_every_tile(budget: int) -> None:
    """Not "on average k": k in EVERY tile of EVERY slice. A globally thresholded
    mask gives 2..11 per tile at rho=0.1 (see the module docstring in stbn.py),
    which is the count jitter the tile ranking exists to remove."""
    spec = _asset_or_skip()
    counts = stbn.tile_counts(stbn.build_rank_volume(spec), budget, spec)
    assert counts.min() == budget and counts.max() == budget


def test_selection_is_nested_across_densities() -> None:
    """A density SCHEDULE walks a progressive sequence: dropping from 25% to 10%
    keeps a subset of the pixels it was already sampling, rather than jumping to
    an unrelated mask."""
    ranks = stbn.build_rank_volume(_asset_or_skip())
    coarse = stbn.selected_mask(ranks, 6)
    fine = stbn.selected_mask(ranks, 16)
    assert (fine | coarse == fine).all()      # coarse subset of fine


def test_packed_table_carries_the_permutation_and_its_inverse() -> None:
    """The GPU only ever sees these bytes, so assert them rather than the array
    they came from: low half = rank by pixel (the loss's test), high half = pixel
    by slot (the backward's compacted walk)."""
    spec = _asset_or_skip()
    ranks = stbn.build_rank_volume(spec)
    packed = stbn.pack_table(ranks, spec)

    assert packed.dtype == np.uint32 and packed.shape == (spec.voxels,)
    unpacked_ranks, inverse = stbn.unpack_table(packed, spec)

    block_ranks = ranks.reshape(-1)[stbn._block_voxel_index(spec)]
    assert (unpacked_ranks == block_ranks).all()
    # rank(inverse[j]) == j for every slot, i.e. slot j IS the j-th selected px.
    round_trip = np.take_along_axis(unpacked_ranks, inverse.astype(np.intp), axis=-1)
    assert (round_trip == np.arange(spec.tile_area)).all()


def test_pixels_per_tile_never_starves_a_tile_completely() -> None:
    """A tile budget of 0 would leave its splats with no gradient at all — a
    qualitatively worse failure than a coarse budget, so tiny fractions clamp
    to 1 rather than rounding to nothing."""
    assert stbn.pixels_per_tile(1.0) == TILE_AREA
    assert stbn.pixels_per_tile(0.5) == TILE_AREA // 2
    assert stbn.pixels_per_tile(0.0001) == 1
    assert stbn.pixels_per_tile(2.0) == TILE_AREA        # clamped, not extrapolated


def test_spec_rejects_layouts_the_shaders_cannot_index() -> None:
    with pytest.raises(ValueError, match="whole number"):
        stbn.StbnMaskSpec(source=stbn.SOURCE_WHITE,
                          width=TILE_WIDTH * 2 + 1, height=TILE_HEIGHT).validate()
    with pytest.raises(ValueError, match="toroidal wrap"):
        # Cropping NVIDIA's asset would break the wrap that lets it tile.
        stbn.StbnMaskSpec(width=64, height=64, slices=32).validate()
    with pytest.raises(ValueError, match="unknown STBN source"):
        stbn.StbnMaskSpec(source="perlin").validate()


def test_missing_asset_names_the_fetch_script() -> None:
    """The asset is an opt-in download, so its absence has to be actionable."""
    spec = stbn.StbnMaskSpec(path="does/not/exist.npy")
    with pytest.raises(FileNotFoundError, match="download_stbn"):
        stbn.load_scalar_volume(spec)


# ---------------------------------------------------------- blue-noise quality

def test_slices_are_spatially_blue_not_white() -> None:
    """Low-frequency spectral power, DC removed: white noise spreads power evenly
    so ~5% lands in the lowest quarter-Nyquist disc; blue noise pushes it out.

    Measured at the time of writing: STBN ~0.005-0.007 vs white ~0.034 — i.e.
    5x+ less clumping. The bound is loose enough to survive an asset revision and
    tight enough to fail if the ranking ever stops preserving the arrangement.
    """
    ranks = stbn.build_rank_volume(_asset_or_skip())
    white_ranks = stbn.build_rank_volume(_white())

    for budget in (6, 16):
        blue = np.mean([stbn.low_frequency_power(
            stbn.selected_mask(ranks, budget)[t]) for t in (0, 7, 31)])
        white = np.mean([stbn.low_frequency_power(
            stbn.selected_mask(white_ranks, budget)[t]) for t in (0, 7, 31)])
        assert blue < 0.015, f"budget {budget}: not spatially blue ({blue:.4f})"
        assert blue < white / 3.0, (
            f"budget {budget}: only {white / blue:.1f}x better than white noise")


def test_repeat_visits_of_a_pixel_are_spread_over_time() -> None:
    """The reason to use a SPATIOtemporal mask rather than one static blue-noise
    mask reused on every visit.

    A pixel sampled on visit t should be unlikely to be sampled again on t+1: for
    an independent mask the expected number of such consecutive pairs is
    (slices-1) * rho^2, and STBN comes in far below it (measured: 0.00 vs 0.55 at
    rho=0.094). Without this the temporal axis would be decoration.
    """
    spec = _asset_or_skip()
    ranks = stbn.build_rank_volume(spec)
    white_ranks = stbn.build_rank_volume(_white())

    for budget in (6, 16):
        rho = budget / spec.tile_area
        independent = (spec.slices - 1) * rho * rho
        blue = stbn.temporal_adjacency(ranks, budget)
        white = stbn.temporal_adjacency(white_ranks, budget)
        assert blue < independent / 4.0, (
            f"budget {budget}: temporal clumping {blue:.3f} vs independent "
            f"{independent:.3f}")
        assert blue < white, f"budget {budget}: no better than white ({white:.3f})"


def test_pixels_get_a_fair_share_of_visits_over_one_temporal_cycle() -> None:
    """How close the 1/rho reweighting is to unbiased in practice.

    The estimator is unbiased when the slice index is uniform over the mask's
    temporal period. It is exactly uniform once a camera has been drawn a
    multiple of `slices` times; short of that, what matters is how evenly the
    DETERMINISTIC per-pixel exposure is spread over one cycle. Mean is exactly
    rho by construction (each tile contributes k of TILE_AREA on every slice);
    this pins the SPREAD, which STBN roughly halves versus white noise
    (measured: std 0.020 vs 0.037 at rho=0.094).
    """
    spec = _asset_or_skip()
    for budget in (6, 16):
        rho = budget / spec.tile_area
        frequency = stbn.selected_mask(
            stbn.build_rank_volume(spec), budget).mean(axis=0)
        white_frequency = stbn.selected_mask(
            stbn.build_rank_volume(_white()), budget).mean(axis=0)
        assert frequency.mean() == pytest.approx(rho, abs=1e-12)
        assert frequency.std() < white_frequency.std()
        assert frequency.std() < 0.5 * rho


def test_white_control_shares_every_invariant_with_the_real_mask() -> None:
    """The control has to differ ONLY in arrangement — same per-tile count, same
    nesting, same table layout — or an A/B against it measures the plumbing
    instead of the blue noise."""
    spec = _white(seed=7)
    ranks = stbn.build_rank_volume(spec)
    counts = stbn.tile_counts(ranks, 6, spec)
    assert counts.min() == counts.max() == 6
    assert (stbn.selected_mask(ranks, 6) <= stbn.selected_mask(ranks, 16)).all()


def test_white_mask_is_reproducible_and_reseedable() -> None:
    a = stbn.build_rank_volume(_white(seed=3))
    b = stbn.build_rank_volume(_white(seed=3))
    c = stbn.build_rank_volume(_white(seed=4))
    assert (a == b).all()
    assert not (a == c).all()


# ------------------------------------------------------------------ ranking

def test_ranking_a_scalar_volume_preserves_its_local_order() -> None:
    """Tile stratification must only REORDER within a tile — it is what makes the
    per-tile count exact, and it would be worthless if it also scrambled the
    arrangement the source volume encodes."""
    spec = stbn.StbnMaskSpec(source=stbn.SOURCE_WHITE,
                             width=TILE_WIDTH, height=TILE_HEIGHT, slices=2)
    rng = np.random.default_rng(0)
    volume = rng.random((2, TILE_HEIGHT, TILE_WIDTH))
    ranks = stbn.rank_volume_from_scalar(volume, spec)

    for t in range(2):
        order = np.argsort(volume[t].reshape(-1))
        # The pixel with the smallest value must get rank 0, and so on.
        assert (ranks[t].reshape(-1)[order] == np.arange(TILE_AREA)).all()


def test_cache_returns_the_same_bytes_it_stored(tmp_path) -> None:
    spec = _white(seed=11)
    first = stbn.load_or_build(spec, tmp_path)
    assert (tmp_path / f"{spec.cache_key()}.npy").is_file()
    assert np.array_equal(first, stbn.load_or_build(spec, tmp_path))
    assert np.array_equal(first, stbn.pack_table(stbn.build_rank_volume(spec), spec))


def test_corrupt_cache_entry_is_rebuilt_not_raised(tmp_path) -> None:
    """The cache is an optimization; a truncated file (interrupted run, half-
    written .npy) must not take a training run down with it."""
    spec = _white(seed=12)
    expected = stbn.load_or_build(spec, tmp_path)
    (tmp_path / f"{spec.cache_key()}.npy").write_bytes(b"not an npy")
    assert np.array_equal(stbn.load_or_build(spec, tmp_path), expected)
