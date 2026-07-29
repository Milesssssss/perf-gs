"""CPU tests for trainer.config_types.validate_training_config + the schedule
zero-interval guard.

The CLI/config-correctness refactor moves configuration validation UP-FRONT: an
invalid combination now fails fast with a readable ``ConfigError`` BEFORE any GPU
allocation, instead of a ``ZeroDivisionError`` deep in the schedule or an empty
train split that silently trains on nothing. These pin the guarded cases and
prove the CLI defaults still pass.
"""

from __future__ import annotations

import pytest
from v2_train import build_parser

from trainer.config.config_types import (
    ConfigError,
    DensificationMode,
    ScheduleConfig,
    validate_training_config,
)
from trainer.frontend.schedule import TrainingSchedule


def _args(**overrides):
    args = build_parser().parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _validate(args, *, n_cams=10, n_train=9, n_test=1):
    return validate_training_config(args, n_cams=n_cams, n_train=n_train,
                                    n_test=n_test)


def test_defaults_pass_and_return_structure_aware() -> None:
    assert _validate(_args()) == DensificationMode.STRUCTURE_AWARE


def test_returns_selected_mode() -> None:
    assert _validate(_args(densify_mode="none")) == DensificationMode.NONE
    assert _validate(_args(densify_mode="simple")) == DensificationMode.SIMPLE


@pytest.mark.parametrize("overrides", [
    {"iters": 0},
    {"max_splats": 0},
    {"densify_every": 0},
    {"densify_from": -1},
    {"densify_from": 2000, "densify_to": 1000},
])
def test_fatal_scalar_combinations_raise(overrides) -> None:
    with pytest.raises(ConfigError):
        _validate(_args(**overrides))


def test_zero_structure_cadence_raises_only_on_structure_mode() -> None:
    # sad_structure_every / sad_accum_every are modulo divisors on the
    # structure-aware path, so 0 is fatal there...
    with pytest.raises(ConfigError):
        _validate(_args(densify_mode="structure_aware", sad_structure_every=0))
    with pytest.raises(ConfigError):
        _validate(_args(densify_mode="structure_aware", sad_accum_every=0))
    # ...but irrelevant when structure-aware densify is not selected.
    assert _validate(_args(densify_mode="simple", sad_structure_every=0)) \
        == DensificationMode.SIMPLE


def test_empty_train_split_raises() -> None:
    with pytest.raises(ConfigError):
        _validate(_args(), n_cams=0, n_train=0, n_test=0)


def test_empty_holdout_raises_only_when_eval_on() -> None:
    with pytest.raises(ConfigError):
        _validate(_args(eval_holdout=1), n_cams=5, n_train=5, n_test=0)
    # eval off => no test views is fine (all cams are trained + evaluated).
    assert _validate(_args(eval_holdout=0), n_cams=5, n_train=5, n_test=0) \
        == DensificationMode.STRUCTURE_AWARE


def test_schedule_preview_survives_zero_reset_every() -> None:
    # Regression: the preview-buffering modulo used to divide by reset_every, so
    # reset_every=0 (resets disabled) raised ZeroDivisionError. It is now
    # guarded — all previews fall back to the +50 offset.
    cfg = ScheduleConfig.from_args(_args(reset_every=0, preview_every=1000))
    sched = TrainingSchedule(cfg)
    fired = [it for it in range(1, cfg.iterations + 1)
             if sched.events_at(it).preview]
    assert 1050 in fired  # 1000 + 50 (no +350 since resets are disabled)
