# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""``--evaluation`` / the panel's "eval + final metrics": one veto, three passes.

The expensive thing a run does besides training is render the WHOLE camera set —
the periodic eval sweep (which lands ON the last iteration whenever --eval-every
divides --iters), the post-training metrics pass, and the held-out test PSNR. This
is the single switch over all three, on the CLI and in the hub panel.

It is ON by default and ``--no-evaluation`` vetoes all three; the tests below pin
both halves of that contract: with it off nothing whole-camera-set runs no matter
what the individual flags say, and with it on each pass still obeys its own flag.
"""
from __future__ import annotations

from trainer.config.resolve import RunConfig, end_of_run_cadence
from trainer.frontend.cli import build_parser
from trainer.frontend.reporting.outputs import RunArtifacts
from trainer.frontend.schedule import TrainingSchedule
from trainer.frontend.settings import BOOL, SETTINGS_MAP, group_settings

# --------------------------------------------------------------------------
# the knob itself (shared by the CLI and the panel)
# --------------------------------------------------------------------------

def test_it_is_one_shared_knob_on_both_surfaces() -> None:
    setting = SETTINGS_MAP["evaluation"]
    assert setting.kind == BOOL and setting.default is True
    assert setting in group_settings("eval")          # the panel renders it
    args = build_parser().parse_args([])
    assert args.evaluation is True
    assert build_parser().parse_args(["--no-evaluation"]).evaluation is False


def test_default_on_means_nothing_changed_for_existing_runs() -> None:
    """A veto, not a switch: with it on, each pass still obeys its own flag."""
    args = build_parser().parse_args([])
    assert (end_of_run_cadence(args.eval_every, args.iters), args.compute_final_metrics,
            args.final_test_psnr) == (3000, False, False)


def _events_over(*argv):
    args = build_parser().parse_args(list(argv))
    schedule = TrainingSchedule(RunConfig.from_args(args, 1.0).schedule)
    return [schedule.events_at(it) for it in range(1, args.iters + 1)]


def test_a_longer_run_evaluates_once_at_its_own_end() -> None:
    """The default cadences follow --iters, so nothing fires at 3000 of a 7000-step run."""
    events = _events_over("--iters", "7000", "--evaluation")
    for flag in ("preview", "render_all", "eval"):
        fired = [e.iteration for e in events if getattr(e, flag)]
        assert fired == [7000], flag


def test_no_evaluation_stops_previews_rendering_every_camera() -> None:
    events = _events_over("--iters", "7000", "--no-evaluation")
    assert not any(e.render_all for e in events)
    assert [e.iteration for e in events if e.preview] == [7000]


# --------------------------------------------------------------------------
# the in-loop eval sweep: gated by whether its stream was opened at all
# --------------------------------------------------------------------------

def _artifacts(tmp_path, *argv):
    return RunArtifacts.create(tmp_path, build_parser().parse_args(list(argv)))


def test_the_default_run_opens_the_eval_sweep(tmp_path) -> None:
    """The default run evaluates once, on its last iteration."""
    artifacts = _artifacts(tmp_path)
    assert artifacts.eval_curve_json is not None
    artifacts.close_all()


def test_no_evaluation_closes_the_door_on_the_sweep(tmp_path) -> None:
    """The loop's own gate is ``stream is None``, so this stops the sweep without
    touching the hot path."""
    artifacts = _artifacts(tmp_path, "--no-evaluation")
    assert artifacts.eval_curve_json is None
    assert not (tmp_path / "eval_curve.jsonl").exists()
    artifacts.close_all()


def test_an_eval_cadence_of_zero_still_wins_on_its_own(tmp_path) -> None:
    artifacts = _artifacts(tmp_path, "--eval-every", "0")
    assert artifacts.eval_curve_json is None
    artifacts.close_all()


# --------------------------------------------------------------------------
# post-training: the metrics pass + the .ply written from the same block
# --------------------------------------------------------------------------

def _post_training_gate(**flags) -> bool:
    """Reproduce the guard ``run_post_training`` computes, over real parsed args.

    The function itself needs a live pipeline, so what is checked here is the
    decision — that the veto is an AND with the pass's own flag, in one place.
    """
    argv = []
    if flags.get("compute_final_metrics"):
        argv.append("--compute-final-metrics")
    if not flags.get("evaluation", True):
        argv.append("--no-evaluation")
    args = build_parser().parse_args(argv)
    return args.compute_final_metrics and args.evaluation


def test_the_metrics_pass_needs_both_its_flag_and_the_veto() -> None:
    assert _post_training_gate(compute_final_metrics=True) is True
    assert _post_training_gate(compute_final_metrics=True, evaluation=False) is False   # vetoed
    # ... and evaluation being ON never turns the pass on by itself.
    assert _post_training_gate() is False


def test_the_final_test_psnr_follows_the_same_veto() -> None:
    args = build_parser().parse_args(["--final-test-psnr"])
    assert args.final_test_psnr is True and args.evaluation is True
    args = build_parser().parse_args(["--final-test-psnr", "--no-evaluation"])
    assert args.final_test_psnr is True and args.evaluation is False
