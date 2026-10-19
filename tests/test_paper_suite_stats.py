# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Tests for the paper suite's statistical core and cell identity.

These functions decide the numbers that would go in a paper, and every one of
them fails silently when wrong: a sign flip inverts a conclusion, a broken pairing
inflates an interval, an extrapolated time-to-quality invents the most quotable
figure in the section, and a non-injective cache key serves one cell's results
under another cell's name. So they are tested against hand-computable answers
rather than against themselves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_RUNNING = Path(__file__).resolve().parent.parent / "scripts" / "running"
if str(_RUNNING) not in sys.path:
    sys.path.insert(0, str(_RUNNING))

from paper_suite import config as cfg
from paper_suite import stats


# --------------------------------------------------------------------------
# mean_std
# --------------------------------------------------------------------------

def test_mean_std_uses_sample_standard_deviation() -> None:
    """ddof=1, not 0. With the three seeds this suite runs, the difference is
    22% — not a rounding detail in a column printed as "mean ± s.d."."""
    summary = stats.mean_std([1.0, 2.0, 3.0])
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["std"] == pytest.approx(1.0)             # ddof=1; ddof=0 = 0.816
    assert summary["n"] == 3


def test_mean_std_ignores_missing_and_non_finite() -> None:
    summary = stats.mean_std([1.0, None, float("nan"), 3.0])
    assert summary["n"] == 2
    assert summary["mean"] == pytest.approx(2.0)


def test_mean_std_of_a_single_value_reports_zero_spread_not_nan() -> None:
    summary = stats.mean_std([5.0])
    assert summary["mean"] == pytest.approx(5.0)
    assert summary["std"] == pytest.approx(0.0)


def test_mean_std_of_nothing_is_none_not_zero() -> None:
    """None, because 0.0 in a PSNR column reads as a catastrophic result rather
    than as an absent one."""
    assert stats.mean_std([])["mean"] is None


# --------------------------------------------------------------------------
# bootstrap_ci
# --------------------------------------------------------------------------

def test_bootstrap_ci_brackets_the_mean_and_is_reproducible() -> None:
    values = [0.10, 0.12, 0.08, 0.15, 0.09, 0.11, 0.13, 0.07, 0.14, 0.10]
    first = stats.bootstrap_ci(values, resamples=2000, seed=7)
    second = stats.bootstrap_ci(values, resamples=2000, seed=7)
    assert first["ci_low"] == second["ci_low"] == pytest.approx(second["ci_low"])
    assert first["ci_low"] < first["mean"] < first["ci_high"]
    assert first["excludes_zero"] is True


def test_bootstrap_ci_reports_containing_zero_when_it_does() -> None:
    """A spread straddling zero must NOT be reported as resolving a sign."""
    values = [-0.5, 0.4, -0.3, 0.6, -0.6, 0.2, -0.1, 0.3]
    result = stats.bootstrap_ci(values, resamples=4000, seed=11)
    assert result["ci_low"] < 0.0 < result["ci_high"]
    assert result["excludes_zero"] is False


def test_bootstrap_ci_refuses_fewer_than_three_points() -> None:
    """An interval from two points would look like evidence and be none."""
    result = stats.bootstrap_ci([1.0, 2.0])
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["excludes_zero"] is None
    assert "too few" in result["note"]


def test_bootstrap_ci_of_identical_values_is_degenerate_not_broken() -> None:
    result = stats.bootstrap_ci([0.25] * 6, resamples=500, seed=3)
    assert result["ci_low"] == pytest.approx(0.25)
    assert result["ci_high"] == pytest.approx(0.25)


# --------------------------------------------------------------------------
# sign_test
# --------------------------------------------------------------------------

def test_sign_test_matches_the_exact_binomial() -> None:
    """8 of 8 positive: two-sided p = 2 * (1/2)^8 = 0.0078125."""
    result = stats.sign_test([1.0] * 8)
    assert result["positive"] == 8 and result["negative"] == 0
    assert result["p_two_sided"] == pytest.approx(2 * 0.5 ** 8)


def test_sign_test_on_an_even_split_is_not_significant() -> None:
    result = stats.sign_test([1.0, -1.0, 1.0, -1.0])
    assert result["p_two_sided"] == pytest.approx(1.0)


def test_sign_test_drops_ties_and_says_so() -> None:
    result = stats.sign_test([1.0, 1.0, 0.0, 0.0])
    assert result["ties"] == 2
    assert result["n"] == 2


# --------------------------------------------------------------------------
# paired_summary
# --------------------------------------------------------------------------

def _maps():
    treatment = {("a", 1): 10.5, ("a", 2): 10.7, ("b", 1): 20.1, ("b", 2): 20.3,
                 ("c", 1): 30.0, ("c", 2): 30.4}
    baseline = {("a", 1): 10.0, ("a", 2): 10.0, ("b", 1): 20.0, ("b", 2): 20.0,
                ("c", 1): 30.0, ("c", 2): 30.0}
    return treatment, baseline


def test_paired_summary_averages_seeds_within_a_scene_before_inference() -> None:
    """Scenes are the independent unit; seeds are repeated measures.

    Three scenes with two seeds each must give THREE per-scene differences, not
    six — treating them as six would shrink the interval by ~sqrt(2) for free.
    """
    treatment, baseline = _maps()
    result = stats.paired_summary("t", "b", "psnr_srgb",
                                  treatment_values=treatment,
                                  baseline_values=baseline, resamples=500)
    assert result.n_cells == 6
    assert result.n_scenes == 3
    assert result.scene_deltas["a"] == pytest.approx(0.6)   # mean(0.5, 0.7)
    assert result.scene_deltas["c"] == pytest.approx(0.2)   # mean(0.0, 0.4)
    assert result.ci["n"] == 3


def test_paired_summary_counts_wins_per_cell_and_ties_exactly() -> None:
    treatment, baseline = _maps()
    result = stats.paired_summary("t", "b", "psnr_srgb",
                                  treatment_values=treatment,
                                  baseline_values=baseline, resamples=500)
    # ("c", 1) is exactly equal -> a tie, not a win and not a loss.
    assert (result.wins, result.losses, result.ties) == (5, 0, 1)


def test_paired_summary_respects_direction_for_lower_is_better_metrics() -> None:
    """LPIPS is lower-better: a NEGATIVE delta is a win. A sign error here would
    invert the perceptual conclusion while leaving PSNR looking right."""
    treatment = {("a", 1): 0.10, ("b", 1): 0.20}
    baseline = {("a", 1): 0.15, ("b", 1): 0.18}
    result = stats.paired_summary("t", "b", "lpips_alex",
                                  treatment_values=treatment,
                                  baseline_values=baseline, resamples=200)
    assert result.higher_is_better is False
    assert (result.wins, result.losses) == (1, 1)


def test_paired_summary_drops_unpaired_cells_and_records_which() -> None:
    """A cell missing on one side must not enter the comparison, and must not
    vanish silently — that is how an incomplete grid becomes a biased result."""
    treatment = {("a", 1): 1.0, ("b", 1): 2.0, ("orphan", 1): 3.0}
    baseline = {("a", 1): 0.0, ("b", 1): 0.0}
    result = stats.paired_summary("t", "b", "psnr_srgb",
                                  treatment_values=treatment,
                                  baseline_values=baseline, resamples=200)
    assert result.n_cells == 2
    assert "orphan@s1" in result.unpaired_dropped


def test_paired_summary_treats_descriptive_metrics_as_directionless() -> None:
    """Gaussian count is neither better high nor low; it must not accrue wins."""
    result = stats.paired_summary(
        "t", "b", "n_gaussians",
        treatment_values={("a", 1): 100.0}, baseline_values={("a", 1): 90.0},
        resamples=200)
    assert result.higher_is_better is None
    assert (result.wins, result.losses, result.ties) == (0, 0, 0)


# --------------------------------------------------------------------------
# time_to_target
# --------------------------------------------------------------------------

def test_time_to_target_interpolates_between_bracketing_checkpoints() -> None:
    iters = [250, 500, 750]
    times = [10.0, 20.0, 30.0]
    values = [20.0, 22.0, 24.0]
    # Target 21.0 sits halfway between the first two samples.
    out = stats.time_to_target(iters, times, values, 21.0)
    assert out["status"] == "reached"
    assert out["seconds"] == pytest.approx(15.0)
    assert out["iteration"] == pytest.approx(375.0)
    assert out["interpolated"] is True


def test_time_to_target_never_extrapolates_past_the_measured_range() -> None:
    """The whole point: an unreached target is `not_reached` with its shortfall,
    never a projected number."""
    out = stats.time_to_target([250, 500], [10.0, 20.0], [20.0, 21.0], 25.0)
    assert out["status"] == "not_reached"
    assert out["seconds"] is None
    assert out["shortfall"] == pytest.approx(4.0)
    assert out["best_value"] == pytest.approx(21.0)


def test_time_to_target_returns_the_first_crossing_not_the_last() -> None:
    """Quality can dip after a densification or an opacity reset; "time to reach
    X" means the first time it got there."""
    out = stats.time_to_target([1, 2, 3, 4], [1.0, 2.0, 3.0, 4.0],
                               [25.0, 20.0, 20.0, 25.0], 24.0)
    assert out["status"] == "reached_at_first_checkpoint"
    assert out["seconds"] == pytest.approx(1.0)


def test_time_to_target_handles_a_flat_bracket_without_dividing_by_zero() -> None:
    out = stats.time_to_target([1, 2], [1.0, 2.0], [20.0, 20.0], 20.0)
    assert out["status"] == "reached_at_first_checkpoint"


def test_time_to_target_supports_lower_is_better_metrics() -> None:
    out = stats.time_to_target([1, 2], [1.0, 2.0], [0.30, 0.10], 0.20,
                               higher_is_better=False)
    assert out["status"] == "reached"
    assert out["seconds"] == pytest.approx(1.5)


def test_time_to_target_with_no_usable_points_says_so() -> None:
    assert stats.time_to_target([], [], [], 1.0)["status"] == "no_data"


# --------------------------------------------------------------------------
# group_aggregate
# --------------------------------------------------------------------------

def test_group_aggregate_flags_an_incomplete_group() -> None:
    """An aggregate over 11 of 13 scenes must never look like one over 13."""
    rows = [{"psnr_srgb": 20.0}, {"psnr_srgb": 22.0}, {"psnr_srgb": None}]
    summary = stats.group_aggregate(rows, "psnr_srgb", expected_n=3)
    assert summary["n"] == 2
    assert summary["complete"] is False
    assert summary["missing"] == 1


def test_group_aggregate_is_complete_when_it_is() -> None:
    rows = [{"psnr_srgb": 20.0}, {"psnr_srgb": 22.0}]
    assert stats.group_aggregate(rows, "psnr_srgb", expected_n=2)["complete"]


# --------------------------------------------------------------------------
# cell identity
# --------------------------------------------------------------------------

def test_fraction_tag_is_injective_over_the_fractions_in_use() -> None:
    """0.10 and 1.00 must not collide. A scheme that rendered both as "100"
    would alias a sparse cell's directory onto the dense one's."""
    fractions = [0.10, 0.25, 0.50, 0.75, 1.0]
    tags = [cfg.fraction_tag(f) for f in fractions]
    assert len(set(tags)) == len(tags)
    assert cfg.fraction_tag(0.10) == "f0100"
    assert cfg.fraction_tag(0.25) == "f0250"
    assert cfg.fraction_tag(1.0) == "f1000"


def test_fraction_tags_sort_in_density_order() -> None:
    tags = [cfg.fraction_tag(f) for f in (0.10, 0.25, 0.50, 1.0)]
    assert tags == sorted(tags)


def test_cell_keys_are_unique_across_the_whole_plan() -> None:
    """The key is the run directory AND the cache key; a collision would have two
    configurations overwrite each other's results."""
    plan = cfg.build_plan()
    keys = [c.key for c in plan]
    assert len(set(keys)) == len(keys)


def test_cell_key_distinguishes_the_instrumentation_profile() -> None:
    """A cell carrying the convergence ladder is not interchangeable with one
    without it: its wall-clock contains eight held-out sweeps."""
    plain = cfg.Cell(scene="bicycle", regime="stbn", seed=42, fraction=0.25)
    conv = cfg.Cell(scene="bicycle", regime="stbn", seed=42, fraction=0.25,
                    convergence=True)
    timing = cfg.Cell(scene="bicycle", regime="stbn", seed=42, fraction=0.25,
                      stage_timing=True)
    assert len({plain.key, conv.key, timing.key}) == 3


def test_config_hash_is_stable_and_sensitive() -> None:
    cell = cfg.Cell(scene="bicycle", regime="stbn", seed=42, fraction=0.25)
    assert cell.config_hash("py") == cell.config_hash("py")
    # The interpreter is excluded on purpose: a venv path change must not
    # invalidate 9 hours of training.
    assert cell.config_hash("python") == cell.config_hash("other-python")
    # Anything that changes the RESULT must change the hash.
    other = cfg.Cell(scene="bicycle", regime="stbn", seed=43, fraction=0.25)
    assert cell.config_hash("py") != other.config_hash("py")
    fraction = cfg.Cell(scene="bicycle", regime="stbn", seed=42, fraction=0.50)
    assert cell.config_hash("py") != fraction.config_hash("py")


def test_consumers_do_not_affect_identity_or_hash() -> None:
    """Adding an experiment as a consumer of an existing cell must not invalidate
    it — otherwise every new experiment retrains the whole grid."""
    cell = cfg.Cell(scene="bicycle", regime="stbn", seed=42, fraction=0.25)
    tagged = cell.with_consumer("Z")
    assert tagged.key == cell.key
    assert tagged.config_hash("py") == cell.config_hash("py")


def test_dense_cells_are_shared_across_every_pixel_fraction() -> None:
    """A pixel fraction does not apply to dense supervision, so there must be one
    dense cell per (scene, seed) rather than one per fraction."""
    plan = cfg.build_plan(scenes=("bicycle",), seeds=(42,),
                          fractions=(0.10, 0.25, 0.50), with_timing=False)
    dense = [c for c in plan if c.regime == "full"]
    assert len(dense) == 1
    assert dense[0].fraction == pytest.approx(1.0)
    # ...and it is recorded as serving both experiments.
    assert {"A", "B"} <= set(dense[0].consumers)


def test_cell_key_separates_the_selection_modes() -> None:
    """The two modes are different sampling processes; sharing a run directory
    would let one overwrite the other, and the resume logic would then serve a
    per-tile-rank result as if it were a global one."""
    glob = cfg.Cell(scene="bicycle", regime="stbn", seed=42, fraction=0.25,
                    select_mode=cfg.SELECT_GLOBAL)
    tiled = cfg.Cell(scene="bicycle", regime="stbn", seed=42, fraction=0.25,
                     select_mode=cfg.SELECT_TILE_RANK)
    assert glob.key != tiled.key
    assert glob.config_hash("py") != tiled.config_hash("py")
    # Both modes are named explicitly: suffixing only the ablation would make an
    # untagged legacy directory look like a global run.
    assert cfg.SELECT_GLOBAL in glob.key
    assert cfg.SELECT_TILE_RANK in tiled.key


def test_dense_cells_ignore_the_selection_mode() -> None:
    """Dense supervises every pixel, so the mode is meaningless there — one run
    must serve as the shared baseline for both arms rather than being trained
    twice for a bit-identical result."""
    glob = cfg.Cell(scene="bicycle", regime="full", seed=42, fraction=1.0,
                    select_mode=cfg.SELECT_GLOBAL)
    tiled = cfg.Cell(scene="bicycle", regime="full", seed=42, fraction=1.0,
                     select_mode=cfg.SELECT_TILE_RANK)
    assert glob.key == tiled.key
    assert glob.config_hash("py") == tiled.config_hash("py")
    assert "--stbn-select-mode" not in glob.command("py")


def test_stratification_arm_covers_every_scene_and_sparse_regime() -> None:
    plan = cfg.build_plan(scenes=("bicycle", "kitchen"), seeds=(42,),
                          fractions=(0.25,), with_timing=False,
                          with_stratification=True)
    tiled = [c for c in plan if c.select_mode == cfg.SELECT_TILE_RANK]
    assert {c.scene for c in tiled} == {"bicycle", "kitchen"}
    assert {c.regime for c in tiled} == set(cfg.SPARSE_REGIMES)
    assert all("S" in c.consumers for c in tiled)
    # ...and it can be turned off without disturbing the rest of the plan.
    without = cfg.build_plan(scenes=("bicycle", "kitchen"), seeds=(42,),
                             fractions=(0.25,), with_timing=False,
                             with_stratification=False)
    assert not [c for c in without if c.select_mode == cfg.SELECT_TILE_RANK]


def test_regime_flags_carry_the_mode_for_sparse_regimes_only() -> None:
    for regime in cfg.SPARSE_REGIMES:
        flags = cfg.regime_flags(regime, 0.25, cfg.SELECT_TILE_RANK)
        assert "--stbn-select-mode" in flags
        assert cfg.SELECT_TILE_RANK in flags
    assert "--stbn-select-mode" not in cfg.regime_flags("full", 0.25)


def test_regime_flags_isolate_exactly_one_variable_at_a_time() -> None:
    """The two contrasts the paper rests on are only valid if the flag sets differ
    in precisely one thing each."""
    iid = set(cfg.regime_flags("iid_white", 0.25))
    indep = set(cfg.regime_flags("indep_blue", 0.25))
    stbn = set(cfg.regime_flags("stbn", 0.25))
    # spatial: same independence, different noise source
    assert iid ^ indep == {"white", "nvidia"}
    # temporal: same noise source, independence dropped
    assert indep ^ stbn == {"--stbn-independent"}
