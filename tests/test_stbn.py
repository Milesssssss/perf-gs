# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

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

import dataclasses

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


@pytest.mark.parametrize("mode", [stbn.SELECT_GLOBAL, stbn.SELECT_TILE_RANK])
def test_packed_table_carries_the_selection_its_list_and_its_length(mode) -> None:
    """The GPU only ever sees these bytes, so assert them rather than the array
    they came from: slot field = "is this pixel supervised?" (the loss's test),
    compacted field = "which pixel is the i-th selected one?" (the backward's
    walk), count field = that list's length.

    Parameterized over both selection modes because the whole point of the layout
    is that the shader cannot tell them apart — it reads a set and a length, and
    the mode is settled entirely on the host.
    """
    spec = dataclasses.replace(_asset_or_skip(), select_mode=mode, fraction=0.25)
    selected = stbn.selected_volume(spec)
    packed = stbn.pack_selection(selected, spec)

    assert packed.dtype == np.uint32 and packed.shape == (spec.voxels,)
    slots, compacted, counts = stbn.unpack_table(packed, spec)

    block_selected = selected.reshape(-1)[stbn._block_voxel_index(spec)]
    # The slot field agrees with the selection, pixel for pixel.
    assert ((slots != stbn._SLOT_NONE) == block_selected).all()
    # The count field is the number of selected pixels in the tile.
    assert (counts == block_selected.sum(axis=-1)).all()

    # The compacted list's first `count` entries are exactly the selected pixels,
    # each once — that equivalence is the backward's licence to walk the list
    # instead of testing all 64 pixels.
    for index in np.ndindex(counts.shape):
        count = int(counts[index])
        listed = compacted[index][:count]
        expected = np.flatnonzero(block_selected[index])
        assert sorted(listed.tolist()) == expected.tolist()


@pytest.mark.parametrize("fraction", [0.10, 0.25, 0.50, 0.33, 0.0001, 0.9])
def test_global_density_matches_the_weight_the_shaders_apply(fraction) -> None:
    """The 1/rho weight and the actual selected density must agree EXACTLY.

    ``stbnWeight`` derives the unbiasing factor from the per-tile budget, which
    has 1/TILE_AREA resolution. If a global threshold kept ``round(rho*W*H)``
    pixels instead of a whole number of per-tile budgets, the weight would be off
    by the quantization error — at rho=0.10 that is 6 vs 6.4, a +6.6% inflation of
    every gradient. Adam absorbs a uniform scale, but the densifier's
    ``--grad-thresh`` is absolute, so inflated gradients densify more eagerly and
    the bug would masquerade as a result ("sparse supervision grows more
    Gaussians"). This is the guard on that.
    """
    spec = stbn.StbnMaskSpec(select_mode=stbn.SELECT_GLOBAL, fraction=fraction)
    density = spec.selected_per_slice / (spec.height * spec.width)
    budget = stbn.pixels_per_tile(fraction, spec.tile_area)
    assert density == pytest.approx(budget / spec.tile_area, abs=1e-12)
    # ...which is exactly the reciprocal of the weight TILE_AREA/budget.
    assert 1.0 / density == pytest.approx(spec.tile_area / budget, abs=1e-9)


@pytest.mark.parametrize("fraction", [0.10, 0.25, 0.50])
def test_the_two_modes_select_the_same_number_of_pixels(fraction) -> None:
    """Density-matched by construction, so the ablation compares ARRANGEMENT
    rather than confounding it with a different pixel count."""
    base = stbn.StbnMaskSpec(source=stbn.SOURCE_WHITE, seed=0, fraction=fraction)
    counts = set()
    for mode in (stbn.SELECT_GLOBAL, stbn.SELECT_TILE_RANK):
        volume = stbn.selected_volume(dataclasses.replace(base, select_mode=mode))
        counts.add(int(volume.sum()))
    assert len(counts) == 1


def test_global_mode_keeps_the_exact_fraction_of_each_whole_slice() -> None:
    """A global threshold is defined on the slice, so the per-SLICE count must be
    exact — that is what makes the uniform 1/rho weight correct."""
    spec = dataclasses.replace(_asset_or_skip(), select_mode=stbn.SELECT_GLOBAL,
                               fraction=0.25)
    selected = stbn.selected_volume(spec)
    per_slice = selected.reshape(spec.slices, -1).sum(axis=1)
    assert (per_slice == spec.selected_per_slice).all()
    assert spec.selected_per_slice == round(0.25 * spec.height * spec.width)


def test_global_mode_lets_per_tile_counts_vary_and_tile_rank_does_not() -> None:
    """The defining difference between the two modes.

    Constant per-tile counts are exactly the artificial stratification the global
    mode removes: it must NOT pin every 8x8 tile to k, and the tile_rank mode must.
    """
    base = _asset_or_skip()
    glob = stbn.selected_volume(
        dataclasses.replace(base, select_mode=stbn.SELECT_GLOBAL, fraction=0.25))
    tiled = stbn.selected_volume(
        dataclasses.replace(base, select_mode=stbn.SELECT_TILE_RANK,
                            fraction=0.25))
    index = stbn._block_voxel_index(base)
    glob_counts = glob.reshape(-1)[index].sum(axis=-1)
    tile_counts = tiled.reshape(-1)[index].sum(axis=-1)

    assert tile_counts.min() == tile_counts.max() == 16
    assert glob_counts.min() < 16 < glob_counts.max()
    # ...while still averaging to the requested density.
    assert glob_counts.mean() == pytest.approx(16.0, abs=1e-9)


def test_global_mode_is_bluer_than_per_tile_ranking() -> None:
    """The measurement that motivated the global mode.

    Forcing exactly k per tile imposes a per-tile threshold, which lets pixels on
    either side of a tile boundary both survive and clump — precisely what the
    global ranking avoids. Asserted as a strict inequality on low-frequency power
    (lower = bluer) so a regression to per-tile semantics fails here.
    """
    base = _asset_or_skip()
    glob = stbn.selected_volume(
        dataclasses.replace(base, select_mode=stbn.SELECT_GLOBAL, fraction=0.25))
    tiled = stbn.selected_volume(
        dataclasses.replace(base, select_mode=stbn.SELECT_TILE_RANK,
                            fraction=0.25))

    def low_power(mask):
        return float(np.mean([stbn.low_frequency_power(m) for m in mask]))

    assert low_power(glob) < low_power(tiled)


def test_per_tile_ranking_flatters_the_white_control() -> None:
    """The other half of the compression, and the reason it mattered.

    Forcing 16 of every 64 pixels on WHITE noise — whose natural per-tile counts
    span roughly 5..31 — suppresses its low frequencies too, making the control
    artificially bluer and shrinking the blue-vs-white contrast from both ends.
    """
    white = stbn.StbnMaskSpec(source=stbn.SOURCE_WHITE, seed=0, fraction=0.25)
    glob = stbn.selected_volume(
        dataclasses.replace(white, select_mode=stbn.SELECT_GLOBAL))
    tiled = stbn.selected_volume(
        dataclasses.replace(white, select_mode=stbn.SELECT_TILE_RANK))

    def low_power(mask):
        return float(np.mean([stbn.low_frequency_power(m) for m in mask]))

    assert low_power(tiled) < low_power(glob)


def test_cache_key_separates_every_mode_and_density() -> None:
    """The cache is keyed by content, and the packed table encodes the selected
    SET — so it is density-specific in BOTH modes.

    Regression guard for a real failure: the key once omitted the density for
    ``tile_rank``, so a table built at rho=0.09375 was served for a rho=0.25 run.
    The mask was 9.4% dense while the shaders applied the 25% weight, making every
    gradient 2.67x too small; the only symptom was ~0.5 dB of lost quality.
    """
    base = _asset_or_skip()
    keys = set()
    for mode in (stbn.SELECT_GLOBAL, stbn.SELECT_TILE_RANK):
        for fraction in (0.10, 0.25, 0.50):
            spec = dataclasses.replace(base, select_mode=mode,
                                       fraction=fraction)
            keys.add(spec.cache_key())
    assert len(keys) == 6, "every (mode, density) pair needs its own cache entry"


def test_a_cached_table_of_the_wrong_density_is_rejected(tmp_path) -> None:
    """Shape and dtype are too weak a check to protect the cache.

    Every table of a given volume size has the same shape and dtype whatever its
    density, so a key collision would hand back a valid-looking, wrong mask. The
    loader must verify the count the shaders will actually read.
    """
    spec25 = stbn.StbnMaskSpec(source=stbn.SOURCE_WHITE, seed=0,
                               select_mode=stbn.SELECT_TILE_RANK, fraction=0.25)
    spec10 = dataclasses.replace(spec25, fraction=0.10)

    # Plant the WRONG table under the right name, exactly as the old key
    # collision did, then confirm the loader rebuilds instead of trusting it.
    path = tmp_path / f"{spec25.cache_key()}.npy"
    np.save(path, stbn.build_table(spec10))
    loaded = stbn.load_or_build(spec25, tmp_path)

    _slots, _compact, counts = stbn.unpack_table(loaded, spec25)
    assert int(counts.min()) == int(counts.max()) == 16
    assert np.array_equal(loaded, stbn.build_table(spec25))


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
    assert np.array_equal(first, stbn.build_table(spec))


def test_corrupt_cache_entry_is_rebuilt_not_raised(tmp_path) -> None:
    """The cache is an optimization; a truncated file (interrupted run, half-
    written .npy) must not take a training run down with it."""
    spec = _white(seed=12)
    expected = stbn.load_or_build(spec, tmp_path)
    (tmp_path / f"{spec.cache_key()}.npy").write_bytes(b"not an npy")
    assert np.array_equal(stbn.load_or_build(spec, tmp_path), expected)
