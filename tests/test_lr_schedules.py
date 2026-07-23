"""Characterization test: freeze the trainer's per-step schedules.

Two inline schedules in v2_train.py drive per-iteration scalars:
  * ``_mean_lr``: log-linear (exponential) mean/position LR decay init -> final
    over a FIXED horizon, clamped at the ends.
  * ``_extinction_hardness``: hold at 1.0 (alpha-over) until the first densify,
    then log-linear ramp 1 -> target over a warmup window, clamped at target.

The refactor moves these into a pure schedule module; these golden values pin
the CURRENT math (formula + concrete outputs) so the extracted functions must
match. ``_reference_*`` mirror today's inline implementations.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trainer.training.lr_schedule import (
    ExtinctionHardnessSchedule,
    MeanLearningRateSchedule,
)


def _reference_mean_lr(step, lr_init, lr_final, max_steps):
    t = min(max(step / max(max_steps, 1), 0.0), 1.0)
    return float(np.exp(np.log(lr_init) * (1.0 - t) + np.log(lr_final) * t))


def test_mean_lr_endpoints_and_geometric_midpoint() -> None:
    init, final, max_steps = 1e-3, 1e-6, 1000
    assert _reference_mean_lr(0, init, final, max_steps) == pytest.approx(init)
    assert _reference_mean_lr(max_steps, init, final, max_steps) == pytest.approx(final)
    # Log-linear => the midpoint is the GEOMETRIC mean of the endpoints.
    mid = _reference_mean_lr(max_steps // 2, init, final, max_steps)
    assert mid == pytest.approx(math.sqrt(init * final), rel=1e-6)
    assert mid == pytest.approx(3.16227766e-5, rel=1e-6)


def test_mean_lr_clamps_past_horizon() -> None:
    init, final, max_steps = 1e-3, 1e-6, 1000
    assert _reference_mean_lr(2000, init, final, max_steps) == pytest.approx(final)
    assert _reference_mean_lr(-5, init, final, max_steps) == pytest.approx(init)


def test_mean_lr_zero_horizon_guarded() -> None:
    # max(max_steps, 1) guard: step 0 -> init, any step >= 1 -> final.
    assert _reference_mean_lr(0, 1e-3, 1e-6, 0) == pytest.approx(1e-3)
    assert _reference_mean_lr(1, 1e-3, 1e-6, 0) == pytest.approx(1e-6)


def _reference_extinction_hardness(step, target, warmup, first_densify_it):
    if target <= 1.0 or warmup < 0:
        return target
    if first_densify_it is None or step <= first_densify_it:
        return 1.0
    if warmup == 0:
        return target
    t = min((step - first_densify_it) / warmup, 1.0)
    return float(np.exp(np.log(target) * t))


def test_extinction_holds_at_one_until_first_densify() -> None:
    # Before any densify (first_densify_it is None): always 1.0.
    assert _reference_extinction_hardness(1, 4.0, 500, None) == 1.0
    assert _reference_extinction_hardness(500, 4.0, 500, None) == 1.0
    # After first densify recorded at 200, still 1.0 up to and including it.
    assert _reference_extinction_hardness(199, 4.0, 500, 200) == 1.0
    assert _reference_extinction_hardness(200, 4.0, 500, 200) == 1.0


def test_extinction_log_linear_ramp() -> None:
    # target=4, warmup=500, first densify at 200: 4**t, t=(step-200)/500.
    assert _reference_extinction_hardness(325, 4.0, 500, 200) == pytest.approx(
        4.0 ** 0.25)  # t=0.25 -> sqrt(2)
    assert _reference_extinction_hardness(450, 4.0, 500, 200) == pytest.approx(2.0)
    assert _reference_extinction_hardness(700, 4.0, 500, 200) == pytest.approx(4.0)
    assert _reference_extinction_hardness(900, 4.0, 500, 200) == pytest.approx(4.0)


def test_extinction_disabled_paths() -> None:
    # target <= 1 => hardening off, constant target.
    assert _reference_extinction_hardness(500, 1.0, 500, 200) == 1.0
    assert _reference_extinction_hardness(500, 0.5, 500, 200) == 0.5
    # warmup < 0 => schedule disabled, constant target from step 0.
    assert _reference_extinction_hardness(0, 4.0, -1, 200) == 4.0
    assert _reference_extinction_hardness(10, 4.0, -1, None) == 4.0


def test_extinction_zero_warmup_jumps_after_first_densify() -> None:
    assert _reference_extinction_hardness(150, 4.0, 0, 200) == 1.0   # still <= first
    assert _reference_extinction_hardness(300, 4.0, 0, 200) == 4.0   # jump to target


# --- the extracted schedule CLASSES reproduce the frozen reference -----------

def test_mean_lr_class_matches_reference() -> None:
    s = MeanLearningRateSchedule(lr_init=1e-3, lr_final=1e-6, max_steps=1000)
    for step in (0, 250, 500, 750, 1000, 2000, -5):
        assert s.at(step) == pytest.approx(
            _reference_mean_lr(step, 1e-3, 1e-6, 1000))


def test_extinction_class_matches_reference() -> None:
    s = ExtinctionHardnessSchedule(target=4.0, warmup=500)
    assert s.target == 4.0
    # Before the first densify: held at 1.0 regardless of step.
    assert s.at(1) == 1.0
    assert s.at(500) == 1.0
    s.notify_first_densification(200)
    for step in (199, 200, 325, 450, 700, 900):
        assert s.at(step) == pytest.approx(
            _reference_extinction_hardness(step, 4.0, 500, 200))


def test_extinction_class_notify_is_idempotent() -> None:
    s = ExtinctionHardnessSchedule(target=4.0, warmup=500)
    s.notify_first_densification(200)
    s.notify_first_densification(999)   # ignored — only the first call counts
    # Ramp is anchored at 200, so step 700 is fully ramped (t=1 -> target). If
    # the second notify had won, 700 <= 999 would still read 1.0.
    assert s.at(700) == pytest.approx(4.0)


def test_extinction_class_disabled_paths() -> None:
    assert ExtinctionHardnessSchedule(target=1.0, warmup=500).at(500) == 1.0
    disabled = ExtinctionHardnessSchedule(target=4.0, warmup=-1)
    assert disabled.at(0) == 4.0
    assert disabled.at(10) == 4.0
