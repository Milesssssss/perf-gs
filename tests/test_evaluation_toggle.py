# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Opt-in evaluation on the CLI and in the hub, independent of the training split."""
from __future__ import annotations

import pytest

from trainer.config.resolve import RunConfig
from trainer.frontend.cli import parse_args
from trainer.frontend.reporting.outputs import RunArtifacts
from trainer.frontend.schedule import TrainingSchedule
from trainer.frontend.settings import TrainingSetup


@pytest.mark.parametrize("preset", ["ultrafast", "balanced", "quality"])
def test_default_run_keeps_holdout_without_scoring_or_dumping_images(preset, tmp_path):
    args = parse_args(["--preset", preset])
    assert (args.eval_holdout, args.eval_holdout_stride) == (1, 8)
    assert not (args.evaluation or args.write_ground_truths or args.compute_final_metrics
                or args.final_test_psnr or args.run_report or args.eval_lpips)
    setup = TrainingSetup.from_args(args)
    assert setup.get("evaluation") is False
    schedule = TrainingSchedule(RunConfig.from_args(args, 1.0).schedule)
    for step in range(1, args.iters + 1):
        e = schedule.events_at(step)
        assert not (e.log or e.preview or e.render_all or e.eval or e.psnr_curve)
    artifacts = RunArtifacts.create(tmp_path, args)
    try:
        assert artifacts.eval_curve_json is None and artifacts.psnr_curve_json is None
    finally:
        artifacts.close_all()

    setup.set("evaluation", True)
    setup.apply_to(args)
    assert RunConfig.from_args(args, 1.0).schedule.eval_every == args.iters


@pytest.mark.parametrize(("flags", "steps"), [
    (["--evaluation"], [20]),
    (["--evaluation", "--eval-every", "0"], []),
    (["--eval-every", "10"], [10, 20]),
    (["--eval-iters", "3", "8"], [3, 8]),
    (["--eval-iters", "3", "--no-evaluation"], []),
])
def test_eval_sweep_only_runs_at_requested_steps(flags, steps, tmp_path):
    args = parse_args(["--iters", "20", *flags])
    schedule = TrainingSchedule(RunConfig.from_args(args, 1.0).schedule)
    assert [it for it in range(1, 21) if schedule.events_at(it).eval] == steps
    artifacts = RunArtifacts.create(tmp_path, args)
    try:
        assert (artifacts.eval_curve_json is not None) == bool(steps)
    finally:
        artifacts.close_all()


@pytest.mark.parametrize("flag", ["--run-report", "--compute-final-metrics", "--final-test-psnr"])
def test_metric_flag_enables_only_requested_pass_and_explicit_veto_wins(flag):
    args = parse_args([flag])
    assert args.evaluation and args.eval_every == 0 and args.preview_every == 0
    assert not parse_args([flag, "--no-evaluation"]).evaluation
    assert not parse_args(["--no-evaluation", flag]).evaluation
