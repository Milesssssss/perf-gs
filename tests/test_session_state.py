# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The application state machine: three lifecycles, one derived UI contract.

These are the spec's acceptance tests expressed against state rather than pixels —
at every point a user must be able to answer "what am I working on", "what is
happening", "what did my last action do" and "what can I do next", and each of those
is a property here. If a transition leaves the UI unable to answer one of them, a
test below fails.
"""
from __future__ import annotations

import pytest

from trainer.frontend.viewer.session import (
    ERROR,
    KIND_COLMAP,
    KIND_IMAGES,
    MODE_HUB,
    MODE_TRAIN,
    OK,
    WARN,
    Confirm,
    DatasetInfo,
    DatasetState,
    ResultInfo,
    ResultState,
    Session,
    ToolsState,
    TrainingState,
    format_count,
    format_duration,
    format_rate,
)


class _Clock:
    """Controllable time, so notice expiry is tested rather than slept through."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def hub(**kwargs) -> Session:
    return Session(MODE_HUB, **kwargs)


def ready_dataset(session: Session, *, kind: str = KIND_COLMAP,
                  images: int = 148) -> None:
    """Drive a session to a trainable dataset the way the orchestrator does."""
    session.dataset_inspecting("D:/data/kitchen")
    session.dataset_detected(kind, scene_dir="D:/data/kitchen")
    session.dataset_ready(DatasetInfo(path="D:/data/kitchen", kind=kind,
                                      images=images, width=1920, height=1080,
                                      points=1_700_000,
                                      scene_dir="D:/data/kitchen"))


def trained(session: Session, *, iterations: int = 30_000) -> None:
    session.training_initializing(iterations)
    session.training_running()
    session.publish_training(step=iterations, total=iterations,
                             gaussians=7_820_000, psnr=26.13, elapsed_s=94.0)
    session.training_completed(ResultInfo(
        iterations=iterations, target_iterations=iterations, wall_s=94.0,
        splats=7_820_000, run_dir="outputs/train/kitchen"))


# --------------------------------------------------------------------------
# first use: an empty session has to explain itself
# --------------------------------------------------------------------------

def test_an_empty_session_says_what_to_do() -> None:
    snap = hub().snapshot()
    assert snap.dataset_state == DatasetState.EMPTY
    assert snap.status_badge == "EMPTY"
    assert snap.primary_action == "drop"
    assert "Drop a dataset" in snap.next_step()
    assert not snap.has_dataset and not snap.has_result
    assert not snap.can_start_training and not snap.can_export


def test_nothing_destructive_is_offered_on_an_empty_session() -> None:
    snap = hub().snapshot()
    assert not any((snap.can_clear, snap.can_restart_training, snap.can_reconstruct,
                    snap.can_stop, snap.can_pause))


# --------------------------------------------------------------------------
# dataset lifecycle
# --------------------------------------------------------------------------

def test_a_drop_is_acknowledged_before_anything_is_read() -> None:
    """The drop must change the UI on the frame it lands — the classify + load that
    follow can take seconds."""
    session = hub()
    session.dataset_inspecting("D:/data/kitchen")
    snap = session.snapshot()
    assert snap.dataset_state == DatasetState.INSPECTING
    assert snap.dataset.name == "kitchen"          # named immediately
    assert snap.activity.active and not snap.activity.determinate
    assert snap.busy and not snap.can_drop         # no second drop mid-inspect


def test_what_was_detected_is_published_before_it_is_loaded() -> None:
    session = hub()
    session.dataset_inspecting("D:/data/kitchen")
    session.dataset_detected(KIND_COLMAP, scene_dir="D:/data/kitchen")
    assert session.snapshot().dataset.kind_label == "COLMAP reconstruction"
    session.dataset_loading("Reading the COLMAP model")
    assert session.snapshot().activity.label == "Reading the COLMAP model"


def test_a_loaded_dataset_reports_what_it_contains() -> None:
    session = hub()
    ready_dataset(session)
    snap = session.snapshot()
    assert snap.dataset_state == DatasetState.READY
    assert snap.trainable and snap.can_start_training
    assert snap.dataset.summary() == "148 images · 1920x1080 · 1.70M points"
    assert snap.status_badge == "READY"
    assert snap.primary_action == "train"
    assert not snap.activity.active                 # the busy state is over


def test_a_failed_load_explains_itself_and_offers_a_way_out() -> None:
    session = hub()
    session.dataset_inspecting("D:/data/junk")
    session.dataset_failed("Missing cameras.bin")
    snap = session.snapshot()
    assert snap.dataset_state == DatasetState.ERROR
    assert snap.dataset_error == "Missing cameras.bin"
    assert snap.status_badge == "ERROR"
    assert snap.primary_action == "drop"            # the way out is another folder
    assert snap.can_drop and not snap.can_start_training
    assert not snap.activity.active                 # and it is not still "loading"


def test_images_need_reconstruction_before_they_can_train() -> None:
    session = hub()
    ready_dataset(session, kind=KIND_IMAGES)
    session.update_dataset(scene_dir="")            # images have no scene yet
    snap = session.snapshot()
    assert not snap.trainable and not snap.can_start_training
    assert snap.primary_action == "install_tools"   # tools missing -> that first
    session.tools(ToolsState.READY)
    snap = session.snapshot()
    assert snap.can_reconstruct and snap.primary_action == "reconstruct"
    assert "Reconstruct" in snap.next_step()


def test_reconstruction_is_a_dataset_state_with_progress() -> None:
    session = hub()
    ready_dataset(session, kind=KIND_IMAGES)
    session.tools(ToolsState.READY)
    session.dataset_reconstructing()
    snap = session.snapshot()
    assert snap.dataset_state == DatasetState.RECONSTRUCTING
    assert snap.busy and snap.activity.determinate  # a real bar, not a spinner
    assert snap.status_badge == "RECONSTRUCTING"
    assert not snap.can_start_training              # nothing to train yet


# --------------------------------------------------------------------------
# training lifecycle
# --------------------------------------------------------------------------

def test_pressing_start_stops_the_session_looking_idle() -> None:
    session = hub()
    ready_dataset(session)
    session.training_initializing(30_000)
    snap = session.snapshot()
    assert snap.training_state == TrainingState.INITIALIZING
    assert snap.status_badge == "INITIALIZING"
    assert snap.busy and snap.training_live
    assert not snap.can_start_training              # no double-start
    assert snap.stats.total == 30_000               # the target is known already


def test_a_running_run_exposes_every_number_the_spec_asks_for() -> None:
    session = hub()
    ready_dataset(session)
    session.training_initializing(30_000)
    session.training_running()
    session.publish_training(step=3042, total=30_000, it_per_s=327.0,
                             elapsed_s=9.3, gaussians=7_450_000, psnr=24.95,
                             loss=0.0182, sh_degree=3, densify_step=2900)
    stats = session.snapshot().stats
    assert (stats.step, stats.total) == (3042, 30_000)
    assert round(stats.percent, 1) == 10.1
    assert format_rate(stats.it_per_s) == "327 it/s"
    assert format_duration(stats.elapsed_s) == "9s"
    assert format_duration(stats.eta_s) == "1m 22s"
    assert format_count(stats.gaussians) == "7.45M"
    assert (stats.psnr, stats.sh_degree, stats.densify_step) == (24.95, 3, 2900)


def test_pause_and_resume_are_distinct_states_not_a_toggle_guess() -> None:
    session = hub()
    ready_dataset(session)
    session.training_initializing(30_000)
    session.training_running()
    assert session.snapshot().can_pause and not session.snapshot().can_resume

    session.publish_training(step=8421)
    session.training_paused()
    snap = session.snapshot()
    assert snap.training_state == TrainingState.PAUSED
    assert snap.state_label == "paused at iteration 8,421"
    assert snap.can_resume and not snap.can_pause
    assert snap.can_stop                            # stopping a paused run works
    # A paused run is NOT busy: it is waiting for the user, so the things that make
    # sense while waiting stay available.
    assert not snap.busy and snap.can_configure


def test_stopping_keeps_the_partial_result_rather_than_cancelling() -> None:
    session = hub()
    ready_dataset(session)
    session.training_initializing(30_000)
    session.training_running()
    session.publish_training(step=8421, total=30_000, gaussians=5_000_000)
    session.training_stopping()
    snap = session.snapshot()
    assert snap.training_state == TrainingState.STOPPING
    assert snap.activity.label == "Stopping training"
    assert snap.busy
    session.training_completed(ResultInfo(iterations=8421, target_iterations=30_000,
                                          splats=5_000_000, wall_s=30.0))
    snap = session.snapshot()
    assert snap.has_result and snap.result.iterations == 8421
    assert snap.can_export


def test_finalizing_is_not_a_finished_run() -> None:
    """The eval sweep and metrics pass happen AFTER the last iteration; the panel
    must not look done while the orchestrator cannot serve a command."""
    session = hub()
    ready_dataset(session)
    session.training_initializing(3000)
    session.training_running()
    session.publish_training(step=3000, total=3000)
    session.training_finalizing("Final metrics")
    snap = session.snapshot()
    assert snap.training_state == TrainingState.FINALIZING
    assert snap.busy and not snap.can_export and not snap.can_restart_training
    assert snap.activity.label == "Final metrics"


def test_paused_final_metrics_resume_finalizing_with_their_activity() -> None:
    session = hub()
    ready_dataset(session)
    session.training_initializing(3000)
    session.training_running()
    session.publish_training(step=3000, total=3000)
    session.training_finalizing("Final metrics")
    session.set_progress("Final metrics", 0.5, detail="camera 40/194")
    activity = session.snapshot().activity
    assert session.snapshot().can_pause and session.snapshot().can_stop

    session.training_paused()
    snap = session.snapshot()
    assert snap.training_state == TrainingState.PAUSED
    assert snap.can_resume and snap.can_stop and not snap.can_pause
    assert snap.activity == activity

    session.training_finalizing("Late final metrics")
    snap = session.snapshot()
    assert snap.training_state == TrainingState.PAUSED
    assert snap.activity == activity

    session.training_resumed()
    snap = session.snapshot()
    assert snap.training_state == TrainingState.FINALIZING
    assert snap.activity == activity
    assert snap.can_pause and snap.can_stop and not snap.can_resume


def test_paused_in_loop_evaluation_resumes_running_with_its_activity() -> None:
    session = hub()
    ready_dataset(session)
    session.training_initializing(3000)
    session.training_running()
    session.publish_training(step=1500, total=3000)
    session.begin_activity("Evaluating every camera", 0.25)
    session.set_detail("camera 48/194")
    activity = session.snapshot().activity

    session.training_paused()
    snap = session.snapshot()
    assert snap.training_state == TrainingState.PAUSED
    assert snap.activity == activity

    session.training_resumed()
    snap = session.snapshot()
    assert snap.training_state == TrainingState.RUNNING
    assert snap.activity == activity
    assert snap.can_pause and snap.can_stop and not snap.can_resume


def test_stopping_paused_finalization_ignores_late_resume_and_progress() -> None:
    session = hub()
    ready_dataset(session)
    session.training_initializing(3000)
    session.training_running()
    session.publish_training(step=3000, total=3000)
    session.training_finalizing("Final metrics")
    session.training_paused()
    session.training_stopping()
    stopping = session.snapshot()
    assert stopping.training_state == TrainingState.STOPPING
    assert stopping.activity.label == "Stopping training"
    assert not any((stopping.can_pause, stopping.can_resume, stopping.can_stop))

    session.training_resumed()
    assert session.snapshot() == stopping
    session.training_finalizing("Late final metrics")
    assert session.snapshot() == stopping

    result = ResultInfo(iterations=3000, target_iterations=3000,
                        splats=5_000_000, wall_s=30.0)
    session.training_completed(result)
    snap = session.snapshot()
    assert snap.training_state == TrainingState.COMPLETED
    assert snap.result_state == ResultState.TRAINED and snap.result == result
    assert snap.has_result and snap.can_export
    assert not snap.activity.active
    assert not any((snap.can_pause, snap.can_resume, snap.can_stop))


def test_a_failed_run_says_so_where_training_lives() -> None:
    session = hub()
    ready_dataset(session)
    session.training_initializing(3000)
    session.training_failed("building the run: out of memory")
    snap = session.snapshot()
    assert snap.training_state == TrainingState.FAILED
    assert "out of memory" in snap.training_error
    assert snap.status_badge == "FAILED"
    assert not snap.busy                            # the session is usable again
    assert snap.can_start_training                  # ... and can try again


# --------------------------------------------------------------------------
# result lifecycle
# --------------------------------------------------------------------------

def test_a_finished_run_becomes_the_important_object() -> None:
    session = hub()
    ready_dataset(session)
    trained(session)
    snap = session.snapshot()
    assert snap.training_state == TrainingState.COMPLETED
    assert snap.result_state == ResultState.TRAINED
    assert snap.has_result and snap.unexported
    assert snap.result.summary() == "30,000 iterations · 1m 34s"
    # NO quality metric: the only PSNR the panel has live is one camera's re-render,
    # and beside "30,000 iterations" that reads as the dataset-wide test score.
    assert snap.result.quality() == "7.82M splats"
    assert not hasattr(snap.result, "psnr")
    assert snap.primary_action == "export"          # the unexported result leads
    assert snap.can_export and snap.can_restart_training and snap.can_clear


def test_exporting_is_recorded_so_the_result_is_no_longer_at_risk() -> None:
    session = hub()
    ready_dataset(session)
    trained(session)
    session.result_exported("D:/out/kitchen.ply", 412.5)
    snap = session.snapshot()
    assert snap.result_state == ResultState.EXPORTED
    assert snap.has_result and not snap.unexported
    assert snap.result.exported_path.endswith("kitchen.ply")
    assert snap.result.exported_mb == 412.5
    assert snap.primary_action == "restart"         # nothing left to protect
    assert snap.state_label == "result exported"


def test_double_clicking_retrain_cannot_replace_the_next_result() -> None:
    session = hub()
    ready_dataset(session)
    trained(session)
    session.request("restart")
    assert session.take_command() == "restart"
    session.request("restart")
    session.reset_for_new_run()
    trained(session)
    assert session.take_command() is None
    assert session.snapshot().unexported


def test_an_export_failure_surfaces_on_the_result_not_in_the_log() -> None:
    session = hub()
    ready_dataset(session)
    trained(session)
    session.result_failed("Could not write D:/ro/kitchen.ply: permission denied")
    snap = session.snapshot()
    assert "permission denied" in snap.result_error
    assert snap.result_state == ResultState.TRAINED   # still exportable elsewhere
    assert snap.can_export


def test_initial_points_are_a_result_state_but_not_a_reconstruction() -> None:
    session = hub()
    ready_dataset(session)
    session.result_initialized(1_700_000)
    snap = session.snapshot()
    assert snap.result_state == ResultState.INITIALIZED
    assert not snap.has_result                       # nothing to export yet
    assert not snap.can_export and snap.can_start_training


# --------------------------------------------------------------------------
# acknowledgement: notices
# --------------------------------------------------------------------------

def test_success_notices_expire_and_failures_do_not() -> None:
    clock = _Clock()
    session = hub(clock=clock)
    session.notify("Training complete", OK)
    session.notify("Export failed", ERROR)
    assert len(session.snapshot().notices) == 2
    clock.advance(30.0)
    remaining = session.snapshot().notices
    assert [n.text for n in remaining] == ["Export failed"]


def test_notices_are_bounded_and_keep_the_newest() -> None:
    session = hub()
    for index in range(6):
        session.notify(f"event {index}", OK)
    texts = [n.text for n in session.snapshot().notices]
    assert texts == ["event 3", "event 4", "event 5"]


def test_a_notice_carries_its_severity_marker() -> None:
    session = hub()
    session.notify("Dataset loaded", OK)
    session.notify("Will be replaced", WARN)
    lines = [n.line() for n in session.snapshot().notices]
    assert lines == ["+ Dataset loaded", "! Will be replaced"]


# --------------------------------------------------------------------------
# confirmations: destructive actions do not happen silently
# --------------------------------------------------------------------------

def test_a_pending_question_blocks_the_actions_it_is_about() -> None:
    session = hub()
    ready_dataset(session)
    trained(session)
    session.ask(Confirm(key="dataset", title="Save the current model?", body="...",
                        command="export_then_load", confirm_label="Export and open"))
    snap = session.snapshot()
    assert snap.confirm is not None
    assert snap.primary_action == "confirm"
    assert snap.next_step() == "Save the current model?"
    # Everything else stands down until it is answered.
    assert not any((snap.can_export, snap.can_restart_training, snap.can_clear,
                    snap.can_drop))


def test_accepting_returns_the_command_and_cancelling_returns_nothing() -> None:
    session = hub()
    session.ask(Confirm(key="dataset", title="Load a new dataset?", body="...",
                        command="load_dropped", alt_label="Export first",
                        alt_command="export"))
    assert session.take_confirm(accept=True) == "load_dropped"
    assert session.snapshot().confirm is None

    session.ask(Confirm(key="dataset", title="t", body="b", command="load_dropped"))
    assert session.take_confirm(accept=False) is None
    assert session.snapshot().confirm is None
    assert session.take_confirm(accept=True) is None      # nothing pending


def test_the_alternative_lets_the_user_rescue_the_work_first() -> None:
    session = hub()
    session.ask(Confirm(key="result", title="Clear the session?", body="...",
                        command="clear!", alt_label="Export first",
                        alt_command="export"))
    assert session.take_confirm(accept=True, alt=True) == "export"


# --------------------------------------------------------------------------
# resets: three different meanings, three different operations
# --------------------------------------------------------------------------

def test_clearing_the_session_returns_to_the_drop_target() -> None:
    session = hub()
    ready_dataset(session)
    trained(session)
    session.tools(ToolsState.READY)
    session.set_matcher_index(2)
    session.log("something happened")
    session.clear_session()
    snap = session.snapshot()
    assert snap.dataset_state == DatasetState.EMPTY
    assert snap.training_state == TrainingState.IDLE
    assert snap.result_state == ResultState.NONE
    assert snap.stats.step == 0 and snap.stats.total == 0
    assert snap.primary_action == "drop"
    # Kept: what is not this dataset's story.
    assert snap.tools_state == ToolsState.READY
    assert snap.matcher == "vocab_tree"
    assert snap.log


def test_resetting_for_a_new_run_keeps_the_dataset() -> None:
    """Train Again must not re-read the dataset from disk."""
    session = hub()
    ready_dataset(session)
    trained(session)
    session.reset_for_new_run()
    snap = session.snapshot()
    assert snap.dataset_state == DatasetState.READY   # still loaded
    assert snap.dataset.images == 148
    assert snap.trainable and snap.can_start_training
    assert snap.result_state == ResultState.NONE      # the old result is gone
    assert snap.training_state == TrainingState.IDLE
    assert snap.stats.step == 0


@pytest.mark.parametrize("transition,args,expected_state", [
    ("training_initializing", (3000,), TrainingState.INITIALIZING),
    ("reset_for_new_run", (), TrainingState.IDLE),
    ("clear_session", (), TrainingState.IDLE),
    ("training_completed", (ResultInfo(iterations=3000),), TrainingState.COMPLETED),
], ids=["initialize", "reset", "clear", "complete"])
def test_lifecycle_changes_do_not_resume_a_previous_finalization(
        transition, args, expected_state) -> None:
    session = hub()
    ready_dataset(session)
    session.training_initializing(3000)
    session.training_running()
    session.training_finalizing("Final metrics")
    session.set_progress("Final metrics", 0.5, detail="camera 40/194")
    session.training_paused()

    getattr(session, transition)(*args)
    snap = session.snapshot()
    assert snap.training_state == expected_state
    assert not snap.can_resume
    session.training_resumed()
    assert session.snapshot() == snap


def test_a_pending_click_cannot_act_on_a_cleared_session() -> None:
    session = hub()
    ready_dataset(session)
    session.request("train")
    session.drop("D:/data/other")
    session.clear_session()
    assert session.take_command() is None
    assert session.take_dropped() is None


# --------------------------------------------------------------------------
# configuration lock
# --------------------------------------------------------------------------

@pytest.mark.parametrize("state,editable", [
    (TrainingState.IDLE, True),
    (TrainingState.INITIALIZING, False),
    (TrainingState.RUNNING, False),
    (TrainingState.PAUSED, True),
    (TrainingState.STOPPING, False),
    (TrainingState.FINALIZING, False),
    (TrainingState.COMPLETED, True),
    (TrainingState.FAILED, True),
])
def test_parameters_lock_only_while_a_run_is_consuming_them(state, editable) -> None:
    session = hub()
    session._snap = session._snap.__class__(mode=MODE_HUB, training_state=state)
    assert session.snapshot().can_configure is editable


def test_parameters_are_never_editable_outside_the_hub() -> None:
    assert not Session(MODE_TRAIN).snapshot().can_configure


# --------------------------------------------------------------------------
# activity: every long operation is visible
# --------------------------------------------------------------------------

def test_an_indeterminate_activity_is_not_a_bar_at_zero() -> None:
    session = hub()
    session.begin_activity("Exporting kitchen.ply")
    activity = session.snapshot().activity
    assert activity.active and not activity.determinate
    assert activity.caption() == "Exporting kitchen.ply"


def test_a_determinate_activity_reports_its_percentage() -> None:
    session = hub()
    session.begin_activity("Installing reconstruction tools", 0.0)
    session.set_progress("Downloading colmap", 0.42)
    activity = session.snapshot().activity
    assert activity.determinate and activity.percent() == 42
    assert activity.caption() == "Downloading colmap  42%"


def test_a_detail_refines_the_activity_without_losing_its_progress() -> None:
    session = hub()
    session.begin_activity("Final metrics", 0.5)
    session.set_detail("camera 40/194")
    activity = session.snapshot().activity
    assert activity.caption() == "Final metrics · camera 40/194  50%"
    session.set_detail("")
    assert session.snapshot().activity.caption() == "Final metrics  50%"


def test_ending_an_activity_retires_the_readout() -> None:
    session = hub()
    session.begin_activity("Loading dataset")
    session.end_activity()
    assert not session.snapshot().activity.active


# --------------------------------------------------------------------------
# formatting: no fake precision
# --------------------------------------------------------------------------

def test_values_are_formatted_for_reading() -> None:
    assert format_count(7_820_000) == "7.82M"
    assert format_count(148_000) == "148k"
    assert format_count(940) == "940"
    assert format_duration(94) == "1m 34s"
    assert format_duration(3 * 3600 + 240) == "3h 04m"
    assert format_duration(None) == "--"
    assert format_rate(326.7) == "327 it/s"
    assert format_rate(0.0) == "-- it/s"


def test_the_view_caches_one_snapshot_per_frame() -> None:
    session = hub()
    view = session.view()
    ready_dataset(session)
    assert view.dataset_state == DatasetState.EMPTY   # frame not resampled yet
    view.sample()
    assert view.dataset_state == DatasetState.READY
