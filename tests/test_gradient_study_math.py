# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Tests for the gradient-study metric math and the temporal-only construction.

The study's three statistics are derived analytically from three accumulated
scalars (dot, |a|^2, |b|^2) rather than from a materialized concatenated gradient
vector, and the accumulation is reported as a prefix MEAN while the device holds a
prefix SUM. Both shortcuts are correct, and both are exactly the kind of thing
that is wrong by a factor of K without anything visibly failing — so they are
checked against hand-computable answers and against the direct computation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_RUNNING = Path(__file__).resolve().parent.parent / "scripts" / "running"
if str(_RUNNING) not in sys.path:
    sys.path.insert(0, str(_RUNNING))

import paper_gradient_study as study  # noqa: E402


def _direct(a: np.ndarray, b: np.ndarray) -> dict:
    """The metrics computed the obvious way, on explicit vectors."""
    return {
        "relative_gradient_error": float(np.linalg.norm(a - b)
                                         / np.linalg.norm(b)),
        "cosine_similarity": float(np.dot(a, b)
                                   / (np.linalg.norm(a) * np.linalg.norm(b))),
        "gradient_norm_ratio": float(np.linalg.norm(a) / np.linalg.norm(b)),
    }


def test_identical_vectors_give_zero_error_and_unit_cosine() -> None:
    stats = study.GroupStats()
    v = np.array([3.0, -4.0, 12.0])
    stats.add("g", v, v)
    m = stats.metrics("g")
    assert m["relative_gradient_error"] == pytest.approx(0.0, abs=1e-12)
    assert m["cosine_similarity"] == pytest.approx(1.0, abs=1e-12)
    assert m["gradient_norm_ratio"] == pytest.approx(1.0, abs=1e-12)


def test_doubling_keeps_direction_but_not_magnitude() -> None:
    """The case that separates "right direction" from "right gradient" — and the
    sanity check the brief asks for explicitly."""
    stats = study.GroupStats()
    b = np.array([1.0, 2.0, 3.0])
    stats.add("g", 2.0 * b, b)
    m = stats.metrics("g")
    assert m["cosine_similarity"] == pytest.approx(1.0, abs=1e-12)
    assert m["gradient_norm_ratio"] == pytest.approx(2.0, abs=1e-12)
    assert m["relative_gradient_error"] == pytest.approx(1.0, abs=1e-12)


def test_negation_gives_cosine_minus_one() -> None:
    stats = study.GroupStats()
    b = np.array([1.0, -2.0, 0.5])
    stats.add("g", -b, b)
    assert stats.metrics("g")["cosine_similarity"] == pytest.approx(-1.0,
                                                                    abs=1e-12)
def test_orthogonal_vectors_give_zero_cosine() -> None:
    stats = study.GroupStats()
    stats.add("g", np.array([1.0, 0.0]), np.array([0.0, 5.0]))
    assert stats.metrics("g")["cosine_similarity"] == pytest.approx(0.0,
                                                                    abs=1e-12)


@pytest.mark.parametrize("k", [1, 2, 4, 8, 64])
def test_scale_parameter_matches_dividing_by_k_directly(k) -> None:
    """The device holds a SUM of K estimates; the reported quantity is the mean.

    ``metrics(scale=K)`` must equal computing the metrics on ``sum/K``. Getting
    this wrong would leave cosine right (it is scale-invariant) while making the
    norm ratio and the relative error wrong by exactly K — which looks like a
    catastrophic method failure rather than a bookkeeping bug.
    """
    rng = np.random.default_rng(7)
    dense = rng.normal(size=500)
    summed = sum(dense + 0.3 * rng.normal(size=500) for _ in range(k))

    stats = study.GroupStats()
    stats.add("g", summed, dense)
    derived = stats.metrics("g", scale=float(k))
    expected = _direct(summed / k, dense)
    for key, value in expected.items():
        assert derived[key] == pytest.approx(value, rel=1e-10, abs=1e-12)


def test_group_stats_are_additive_over_components() -> None:
    """Accumulating component-by-component must equal one pass over the
    concatenation — that equivalence is what lets the study avoid allocating a
    giant parameter vector."""
    rng = np.random.default_rng(11)
    parts_a = [rng.normal(size=50) for _ in range(4)]
    parts_b = [rng.normal(size=50) for _ in range(4)]

    stats = study.GroupStats()
    for a, b in zip(parts_a, parts_b):
        stats.add("g", a, b)
    derived = stats.metrics("g")
    expected = _direct(np.concatenate(parts_a), np.concatenate(parts_b))
    for key, value in expected.items():
        assert derived[key] == pytest.approx(value, rel=1e-10, abs=1e-12)


def test_fixed_point_scale_cancels_in_all_three_metrics() -> None:
    """All three statistics are ratios, so the study may work in raw int64
    fixed-point units without dividing by 65536.

    Only the RATIOS cancel. ``dense_norm``/``sparse_norm`` are absolute and scale
    with the units by construction, so they are deliberately excluded here.
    """
    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    plain = study.GroupStats()
    plain.add("g", a, b)
    scaled = study.GroupStats()
    scaled.add("g", a * 65536.0, b * 65536.0)
    ratios = ("relative_gradient_error", "cosine_similarity",
              "gradient_norm_ratio")
    for key in ratios:
        assert scaled.metrics("g")[key] == pytest.approx(
            plain.metrics("g")[key], rel=1e-9)
    # ...and the absolute norms scale, confirming the units really did change.
    assert scaled.metrics("g")["dense_norm"] == pytest.approx(
        plain.metrics("g")["dense_norm"] * 65536.0, rel=1e-9)


def test_averaging_independent_unbiased_estimates_reduces_error() -> None:
    """Sanity floor for the whole experiment: if averaging K unbiased estimates
    did NOT reduce the relative error, the harness would be measuring nothing."""
    rng = np.random.default_rng(5)
    dense = rng.normal(size=2000)
    errors = []
    for k in (1, 4, 16, 64):
        summed = sum(dense + rng.normal(size=2000) for _ in range(k))
        stats = study.GroupStats()
        stats.add("g", summed, dense)
        errors.append(stats.metrics("g", scale=float(k))["relative_gradient_error"])
    assert errors == sorted(errors, reverse=True)


# ---------------------------------------------------------------------------
# temporal-only construction
# ---------------------------------------------------------------------------

def _spec(fraction: float):
    from trainer.backend.render import stbn_table as stbn_module
    return stbn_module.StbnSpec(fraction=fraction)


@pytest.mark.parametrize("fraction", [0.25, 0.50])
def test_temporal_only_visits_every_pixel_exactly_once_per_period(fraction) -> None:
    """That IS the temporal structure: a perfectly stratified per-pixel schedule
    with period P = 1/rho."""
    spec = _spec(fraction)
    volume = study.build_temporal_only_volume(spec, seed=0)
    period = int(round(1.0 / fraction))
    for start in range(0, spec.slices - period + 1, period):
        window = volume[start:start + period]
        assert (window.sum(axis=0) == 1).all()


def test_temporal_only_has_the_requested_density() -> None:
    spec = _spec(0.25)
    volume = study.build_temporal_only_volume(spec, seed=0)
    assert volume.mean() == pytest.approx(0.25, abs=1e-12)


def test_temporal_only_is_white_spatially_not_blue() -> None:
    """The control must have NO spatial structure — otherwise it would not
    isolate the temporal axis. Compared against the NVIDIA blue mask, its
    low-frequency power must be far higher (i.e. much whiter).
    """
    from trainer.backend.render import stbn_table as stbn_module

    spec = _spec(0.25)
    temporal = study.build_temporal_only_volume(spec, seed=0)
    low_temporal = float(np.mean([stbn_module.low_frequency_power(f)
                                  for f in temporal[:8]]))
    blue_spec = stbn_module.StbnSpec(
        fraction=0.25, select_mode=stbn_module.SELECT_GLOBAL)
    if not blue_spec.resolved_path().is_file():
        pytest.skip("NVIDIA STBN asset absent")
    blue = stbn_module.selected_volume(blue_spec)
    low_blue = float(np.mean([stbn_module.low_frequency_power(f)
                              for f in blue[:8]]))
    assert low_temporal > 5.0 * low_blue


def test_temporal_only_packs_into_a_valid_table() -> None:
    """It must round-trip through the same packer the GPU table uses, or the
    diagnostic would be measuring a mask the shader cannot read."""
    from trainer.backend.render import stbn_table as stbn_module

    spec = _spec(0.25)
    volume = study.build_temporal_only_volume(spec, seed=0)
    packed = stbn_module.pack_selection(volume, spec)
    assert packed.dtype == np.uint32 and packed.shape == (spec.voxels,)
    slots, _compacted, counts = stbn_module.unpack_table(packed, spec)
    index = stbn_module.block_voxel_index(spec)
    assert ((slots != stbn_module.SLOT_NONE)
            == volume.reshape(-1)[index]).all()
    # Unlike a tile-stratified mask, per-tile counts must VARY here (it is white
    # spatially), which is also what exercises the variable-count shader path.
    assert counts.min() < counts.max()


def test_method_table_covers_the_four_variants() -> None:
    names = [m[0] for m in study.METHODS]
    assert names == ["iid_white", "spatial_blue", "temporal_only", "stbn"]
    sources = {m[0]: m[1] for m in study.METHODS}
    traversal = {m[0]: m[2] for m in study.METHODS}
    # spatial_blue and stbn MUST share the noise source and differ only in
    # traversal — that is the temporal contrast.
    assert sources["spatial_blue"] == sources["stbn"] == "nvidia"
    assert traversal["spatial_blue"] == "independent"
    assert traversal["stbn"] == "consecutive"


def test_scalar_groups_partition_the_gradient_components() -> None:
    study._validate_groups()          # raises if the mapping is not total
