# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

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

from trainer.config.config import AdamLRs, LossConfig
from trainer.frontend.schedule import (
    ExtinctionHardnessSchedule,
    LearningRateSchedule,
    StbnDensitySchedule,
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
    s = LearningRateSchedule(base=AdamLRs(mean=1e-3), extent=1.0, mean_steps=1000,
                             iterations=1000, scale_rotation_decay=(1.0, 1.0))
    for step in (0, 250, 500, 750, 1000, 2000, -5):
        assert s.at(step).mean == pytest.approx(
            _reference_mean_lr(step, 1e-3, 1e-5, 1000))


def test_mean_lr_scales_with_the_scene_extent() -> None:
    s = LearningRateSchedule(base=AdamLRs(mean=1e-3), extent=4.0, mean_steps=100,
                             iterations=100, scale_rotation_decay=(1.0, 1.0))
    assert s.at(0).mean == pytest.approx(4e-3)
    assert s.at(100).mean == pytest.approx(4e-5)


def test_constant_decay_keeps_every_other_lr_exactly() -> None:
    base = AdamLRs()
    s = LearningRateSchedule(base=base, extent=1.0, mean_steps=100, iterations=100,
                             scale_rotation_decay=(1.0, 1.0))
    for step in (0, 50, 100):
        lrs = s.at(step)
        assert (lrs.scale, lrs.rot, lrs.opacity, lrs.color, lrs.sh_rest) == (
            base.scale, base.rot, base.opacity, base.color, base.sh_rest)


def test_scale_and_rotation_decay_log_linearly_over_the_run() -> None:
    base = AdamLRs(scale=0.01, rot=0.002, opacity=0.05)
    s = LearningRateSchedule(base=base, extent=1.0, mean_steps=3000, iterations=3000,
                             scale_rotation_decay=(3.0, 0.5))
    assert s.at(0).scale == pytest.approx(0.03)
    assert s.at(3000).rot == pytest.approx(0.001)
    # Log-linear: halfway is the geometric mean of 3x and 0.5x.
    assert s.at(1500).scale == pytest.approx(0.01 * math.sqrt(1.5))
    assert s.at(4000).scale == pytest.approx(0.005)
    assert s.at(1500).opacity == 0.05


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


# --- STBN pixel-density ramp ------------------------------------------------
# Same log-linear shape as the hardness ramp, but descending: full supervision
# first (sparse pixels hurt the thin structures early densification is trying to
# create), then geometric decay to the steady-state fraction.

def test_stbn_density_ramps_geometrically_to_the_target() -> None:
    schedule = StbnDensitySchedule(target_fraction=0.1, warmup=1000)
    assert schedule.target == 0.1
    assert schedule.at(1) == pytest.approx(1.0)          # step 1 = full density
    assert schedule.at(1001) == pytest.approx(0.1)
    assert schedule.at(5000) == pytest.approx(0.1)       # clamped past the ramp
    # Log-linear => the midpoint is the geometric mean of 1.0 and the target.
    assert schedule.at(501) == pytest.approx(math.sqrt(0.1), rel=1e-6)
    # Monotone descent, so the ramp passes through the intermediate densities
    # (1.0 -> 0.5 -> 0.25 -> 0.1) instead of sitting near one end.
    values = [schedule.at(s) for s in range(1, 1002, 50)]
    assert all(b <= a for a, b in zip(values, values[1:]))


def test_stbn_density_without_a_warmup_pins_the_target() -> None:
    schedule = StbnDensitySchedule(target_fraction=0.25, warmup=0)
    assert schedule.at(1) == 0.25
    assert schedule.at(9999) == 0.25


def test_stbn_density_disabled_target_never_ramps() -> None:
    """target 1.0 IS the off switch, so a warmup must not dip below full density
    and quietly enable sparse supervision for a run that asked for none."""
    schedule = StbnDensitySchedule(target_fraction=1.0, warmup=500)
    assert schedule.at(1) == 1.0
    assert schedule.at(250) == 1.0
    assert schedule.at(500) == 1.0


@pytest.mark.parametrize(("step", "weight"), [(1, 0.0), (1501, 0.0), (2190, 0.0), (2595, 0.5), (3000, 1.0)])
def test_short_objective_matches_the_three_thousand_step_recipe(step, weight):
    base = AdamLRs(scale=0.01, rot=0.002, opacity=0.05)
    loss = LossConfig(l2_weight=9, ssim_weight=0.22, alpha_weight=0.1)
    schedule = LearningRateSchedule(base, 2.4, 3000, 3000, (1, 1))
    ordinary = schedule.at(step - 1)
    rates, objective = schedule.short_objective(step, ordinary, loss)
    progress = (step - 1) / 3000
    geometry = 0.08 ** weight
    color = 0.3 ** progress * 0.6 ** weight
    assert rates.mean == pytest.approx(2 * ordinary.mean * geometry)
    assert rates.scale == pytest.approx(ordinary.scale * geometry)
    assert rates.rot == pytest.approx(base.rot * 3 * (1 / 6) ** progress * geometry)
    assert rates.opacity == pytest.approx(base.opacity * 0.7 * color)
    assert rates.color == pytest.approx(base.color * color)
    assert rates.sh_rest == pytest.approx(base.sh_rest * color)
    assert objective.l2_weight == pytest.approx(loss.l2_weight * (1 + 1.5 * weight))
    assert (objective.ssim_weight, objective.alpha_weight) == (loss.ssim_weight, loss.alpha_weight)
