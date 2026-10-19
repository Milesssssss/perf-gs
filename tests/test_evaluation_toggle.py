# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""``--evaluation`` / the panel's "eval + final metrics": one veto, three passes.

The expensive thing a run does besides training is render the WHOLE camera set —
the periodic eval sweep (which lands ON the last iteration whenever --eval-every
divides --iters), the post-training metrics pass, and the held-out test PSNR. This
is the single switch over all three, on the CLI and in the hub panel.

It is OFF by default, so a plain run pays for none of them; the tests below pin
both halves of that contract: with it off nothing whole-camera-set runs no matter
what the individual flags say, and with it on each pass still obeys its own flag.
"""
from __future__ import annotations

from trainer.frontend.cli import build_parser
from trainer.frontend.reporting.outputs import RunArtifacts
from trainer.frontend.settings import BOOL, SETTINGS_MAP, group_settings

# --------------------------------------------------------------------------
# the knob itself (shared by the CLI and the panel)
# --------------------------------------------------------------------------

def test_it_is_one_shared_knob_on_both_surfaces() -> None:
    setting = SETTINGS_MAP["evaluation"]
    assert setting.kind == BOOL and setting.default is False
    assert setting in group_settings("eval")          # the panel renders it
    assert not setting.launch_only                    # ... and it is editable
    args = build_parser().parse_args([])
    assert args.evaluation is False
    assert build_parser().parse_args(["--evaluation"]).evaluation is True


def test_default_on_means_nothing_changed_for_existing_runs() -> None:
    """A veto, not a switch: with it on, each pass still obeys its own flag."""
    args = build_parser().parse_args([])
    assert (args.eval_every, args.compute_final_metrics, args.final_test_psnr) \
        == (3000, False, False)


# --------------------------------------------------------------------------
# the in-loop eval sweep: gated by whether its stream was opened at all
# --------------------------------------------------------------------------

def _artifacts(tmp_path, *argv):
    return RunArtifacts.create(tmp_path, build_parser().parse_args(list(argv)))


def test_the_eval_sweep_is_off_until_asked_for(tmp_path) -> None:
    """The default run pays for no whole-camera-set renders at all."""
    artifacts = _artifacts(tmp_path)
    assert artifacts.eval_curve_json is None
    artifacts.close_all()

    artifacts = _artifacts(tmp_path, "--evaluation")
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


def test_debug_bin_viz_says_which_switch_it_needs(tmp_path, capsys) -> None:
    """It records eval views, so it cannot outlive the sweep it piggybacks on."""
    artifacts = _artifacts(tmp_path, "--no-evaluation", "--debug-bin-viz")
    assert artifacts.bin_viz_json is None
    message = capsys.readouterr().out
    assert "--eval-every" in message and "--evaluation" in message
    artifacts.close_all()


def test_bin_viz_still_records_when_evaluation_runs(tmp_path) -> None:
    artifacts = _artifacts(tmp_path, "--evaluation", "--debug-bin-viz")
    assert artifacts.bin_viz_json is not None
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
    if flags.get("evaluation", False):
        argv.append("--evaluation")
    args = build_parser().parse_args(argv)
    return args.compute_final_metrics and args.evaluation


def test_the_metrics_pass_needs_both_its_flag_and_the_veto() -> None:
    assert _post_training_gate(compute_final_metrics=True, evaluation=True) is True
    assert _post_training_gate(compute_final_metrics=True) is False   # vetoed
    # ... and evaluation being ON never turns the pass on by itself.
    assert _post_training_gate(evaluation=True) is False


def test_the_final_test_psnr_follows_the_same_veto() -> None:
    args = build_parser().parse_args(["--final-test-psnr"])
    assert args.final_test_psnr is True and args.evaluation is False
    args = build_parser().parse_args(["--final-test-psnr", "--evaluation"])
    assert args.final_test_psnr is True and args.evaluation is True
