"""ESC aggregation math: geometric mean, zero-catastrophe, partial renormalise."""

from __future__ import annotations

import math

from esc.config import EscWeights
from esc.model import SubScores
from esc.scoring.aggregate import compute_esc, compute_esc_partial


def _subs(value: float) -> SubScores:
    return SubScores(value, value, value, value, value, value)


def test_all_ones_is_one_hundred():
    assert compute_esc(_subs(1.0)) == 100.0


def test_uniform_subscores_pass_through():
    # weights sum to 1, so a uniform value v yields 100 * v.
    assert math.isclose(compute_esc(_subs(0.5)), 50.0, rel_tol=1e-9)


def test_single_zero_is_catastrophic():
    subs = SubScores(0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert compute_esc(subs) == 0.0


def test_geometric_mean_penalises_high_weight_collapse():
    balanced = compute_esc(_subs(0.5))
    # A near-zero in the HIGHEST-weight dimension (C, 0.40) collapses the score
    # far below the balanced case, even with five excellent dimensions.
    collapse_c = compute_esc(SubScores(0.01, 0.9, 0.9, 0.9, 0.9, 0.9))
    assert collapse_c < balanced
    # The same near-zero in the LOWEST-weight dimension (S, 0.05) barely dents
    # it — the geometric mean weights the collapse by that dimension's weight.
    collapse_s = compute_esc(SubScores(0.9, 0.9, 0.9, 0.9, 0.9, 0.01))
    assert collapse_s > collapse_c


def test_partial_drops_unmeasured_and_renormalises():
    esc, effective = compute_esc_partial(
        {"correctness": 0.5, "static_quality": 0.5,
         "edit_efficiency": None, "understandability": None,
         "locality": None, "testability": None}
    )
    assert math.isclose(esc, 50.0, rel_tol=1e-9)
    assert math.isclose(sum(effective.values()), 1.0, rel_tol=1e-9)


def test_partial_weights_match_ratio():
    esc, effective = compute_esc_partial(
        {"correctness": 0.6, "static_quality": 0.6}, EscWeights()
    )
    assert set(effective) == {"correctness", "static_quality"}
    assert effective["correctness"] > effective["static_quality"]
