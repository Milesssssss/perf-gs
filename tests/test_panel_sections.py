"""What the panel SHOWS in each state, and what each control does.

Built with :class:`~tests._ui_testkit.RecordingPanel`, so these are the real section
builders and the real bindings — only the ImGui objects are stand-ins. Assertions are
written at the level the UX contract lives at ("the panel says the dataset is ready
and offers Start Training"), not at widget indices.
"""
from __future__ import annotations

from _ui_testkit import Recorder, RecordingPanel

from trainer.frontend.settings import SETTINGS, SETTINGS_MAP, TrainingSetup
from trainer.frontend.viewer.session import (
    KIND_IMAGES,
    LOSS_HISTORY,
    MODE_HUB,
    OK,
    WARN,
    Confirm,
    DatasetInfo,
    ResultInfo,
    Session,
    ToolsState,
)
from trainer.frontend.viewer.ui.components import (
    bar_scale,
    bars,
    bucket,
    notice_line,
)
from trainer.frontend.viewer.ui.sections import (
    Actions,
    CameraControls,
    configuration_section,
    dataset_section,
    diagnostics_section,
    header,
    loss_section,
    result_section,
    training_section,
    viewer_section,
)

EDITABLE = tuple(s for s in SETTINGS if not s.launch_only)


def _session() -> Session:
    return Session(MODE_HUB, setup=TrainingSetup())


def _actions(recorder: Recorder) -> Actions:
    """Every action wired to a recorder, so a click is observable."""
    names = ("install_tools", "reconstruct", "new_dataset", "clear_session",
             "start_training", "restart_training", "pause", "resume", "stop",
             "export", "confirm", "confirm_alt", "cancel", "on_config_change",
             "clear_log", "browse", "reveal_export")
    kwargs = {name: recorder.action(name) for name in names}
    kwargs["select_matcher"] = recorder.action("select_matcher")
    kwargs["set_recon_cuda"] = recorder.action("set_recon_cuda")
    kwargs["set_export_name"] = recorder.action("set_export_name")
    kwargs["toggle_log"] = recorder.action("toggle_log")
    return Actions(**kwargs)


def build(session: Session, which: str = "all"):
    """Build the panel (or one section) against ``session``; returns the panel,
    the live view and the action recorder."""
    panel, recorder = RecordingPanel(), Recorder()
    view, actions = session.view(), _actions(recorder)
    if which in ("all", "header"):
        header(panel, view)
    if which in ("all", "dataset"):
        dataset_section(panel, view, actions)
    if which in ("all", "training"):
        training_section(panel, view, actions)
    if which in ("all", "result"):
        result_section(panel, view, actions)
    if which in ("all", "configuration"):
        configuration_section(panel, view, session.setup, actions,
                              show_all=lambda: view.show_all_config,
                              toggle_all=session.toggle_config_detail)
    if which in ("all", "loss"):
        loss_section(panel, view)
    if which in ("all", "diagnostics"):
        diagnostics_section(panel, view, actions, show=lambda: view.show_log)
    return panel, view, recorder


def ready(session: Session, *, kind: str = "colmap", images: int = 148) -> None:
    session.dataset_ready(DatasetInfo(path="D:/data/kitchen", kind=kind,
                                      images=images, width=1920, height=1080,
                                      points=1_700_000,
                                      scene_dir="D:/data/kitchen"
                                      if kind == "colmap" else ""))


def trained(session: Session) -> None:
    session.training_initializing(30_000)
    session.training_running()
    session.publish_training(step=30_000, total=30_000, gaussians=7_820_000,
                             psnr=26.13, elapsed_s=94.0)
    session.training_completed(ResultInfo(iterations=30_000, wall_s=94.0,
                                          splats=7_820_000,
                                          run_dir="outputs/train/kitchen"))


# --------------------------------------------------------------------------
# empty: the drop target has to be unmissable
# --------------------------------------------------------------------------

def test_the_empty_state_is_a_drop_target_and_says_what_is_accepted() -> None:
    panel, _, _ = build(_session(), "dataset")
    screen = panel.screen()
    assert "DROP A DATASET HERE" in screen
    assert "image folder or COLMAP data" in screen
    assert "sparse/" in screen                      # what a COLMAP folder needs


def test_the_header_leads_with_the_state_and_the_next_step() -> None:
    panel, _, _ = build(_session(), "header")
    screen = panel.screen()
    assert "[ EMPTY ]" in screen
    assert "Drop a dataset folder onto the window to begin." in screen


def test_nothing_actionable_is_offered_before_a_dataset_exists() -> None:
    panel, _, _ = build(_session())
    assert panel.visible_buttons() == ["Browse...", "Start Training"]
    assert not panel.find("Start Training").enabled


# --------------------------------------------------------------------------
# drop -> inspect -> ready
# --------------------------------------------------------------------------

def test_the_drop_target_is_replaced_by_progress_the_moment_it_lands() -> None:
    session = _session()
    panel, view, _ = build(session, "dataset")
    session.dataset_inspecting("D:/data/kitchen")
    view.sample()
    screen = panel.screen()
    assert "DROP A DATASET HERE" not in screen
    assert "[ INSPECTING ]" in screen and "kitchen" in screen


def test_the_detected_type_is_shown_while_it_is_still_loading() -> None:
    session = _session()
    panel, view, _ = build(session, "dataset")
    session.dataset_inspecting("D:/data/kitchen")
    session.dataset_detected("colmap", scene_dir="D:/data/kitchen")
    session.dataset_loading("Reading the COLMAP model")
    view.sample()
    assert "detected: COLMAP reconstruction" in panel.screen()


def test_a_ready_dataset_shows_a_card_not_a_path_dump() -> None:
    session = _session()
    panel, view, _ = build(session, "dataset")
    ready(session)
    view.sample()
    screen = panel.screen()
    assert "+ kitchen" in screen
    assert "COLMAP reconstruction" in screen
    assert "148 images · 1920x1080 · 1.70M points" in screen
    assert "D:/data/kitchen" in screen              # the path is still available


def test_the_section_header_summarizes_the_dataset_when_collapsed() -> None:
    session = _session()
    panel, view, _ = build(session, "dataset")
    ready(session)
    view.sample()
    assert panel.of("section")[0].label.startswith("DATASET")
    assert "kitchen" in panel.of("section")[0].label


def test_a_dataset_error_is_shown_where_the_dataset_is() -> None:
    session = _session()
    panel, view, _ = build(session, "dataset")
    session.dataset_inspecting("D:/data/junk")
    session.dataset_failed("Missing cameras.bin")
    view.sample()
    screen = panel.screen()
    assert "x Could not load dataset" in screen
    assert "Missing cameras.bin" in screen
    assert "Try another folder" in panel.visible_buttons()


def test_colmap_controls_appear_only_for_images_that_need_them() -> None:
    session = _session()
    panel, view, _ = build(session, "dataset")
    ready(session)                                  # a COLMAP dataset
    view.sample()
    assert "Reconstruct to COLMAP" not in panel.visible_buttons()

    ready(session, kind=KIND_IMAGES)                # ... now an image folder
    view.sample()
    assert "Install reconstruction tools" in panel.visible_buttons()
    session.tools(ToolsState.READY)
    view.sample()
    buttons = panel.visible_buttons()
    assert "Reconstruct to COLMAP" in buttons
    assert "Install reconstruction tools" not in buttons   # nothing left to install


def test_the_reconstruct_button_fires_its_action() -> None:
    session = _session()
    panel, view, recorder = build(session, "dataset")
    ready(session, kind=KIND_IMAGES)
    session.tools(ToolsState.READY)
    view.sample()
    button = panel.find("Reconstruct to COLMAP")
    assert button.enabled
    button.click()
    assert recorder.names == ["reconstruct"]


# --------------------------------------------------------------------------
# training status
# --------------------------------------------------------------------------

def test_start_training_is_the_primary_action_on_a_ready_dataset() -> None:
    session = _session()
    panel, view, recorder = build(session, "training")
    ready(session)
    view.sample()
    assert "Ready to train." in panel.screen()
    button = panel.find("Start Training")
    assert button.shown and button.enabled
    button.click()
    assert recorder.names == ["start_training"]


def test_the_run_status_carries_every_number_in_one_block() -> None:
    session = _session()
    panel, view, _ = build(session, "training")
    ready(session)
    session.training_initializing(30_000)
    session.training_running()
    session.publish_training(step=3042, total=30_000, it_per_s=327.0, elapsed_s=9.3,
                             gaussians=7_450_000, psnr=24.95, sh_degree=3,
                             densify_step=2900, densify_kind="structure")
    view.sample()
    screen = panel.screen()
    assert "[ RUNNING ]" in screen
    assert "3,042 / 30,000" in screen and "10.1%" in screen
    assert "327 it/s" in screen and "9s elapsed" in screen and "ETA 1m 22s" in screen
    assert "7.45M splats" in screen and "24.95 dB" in screen
    assert "SH 3" in screen and "densify @ 2,900 (structure)" in screen
    assert panel.of("progress")[0].binds["fraction"]() > 0.10


def test_pause_stop_replace_start_while_a_run_is_live() -> None:
    session = _session()
    panel, view, recorder = build(session, "training")
    ready(session)
    session.training_initializing(30_000)
    session.training_running()
    view.sample()
    assert panel.visible_buttons() == ["Pause", "Stop"]
    panel.find("Pause").click()
    panel.find("Stop").click()
    assert recorder.names == ["pause", "stop"]


def test_a_paused_run_offers_resume_instead_of_pause() -> None:
    session = _session()
    panel, view, recorder = build(session, "training")
    ready(session)
    session.training_initializing(30_000)
    session.training_running()
    session.publish_training(step=8421, total=30_000)
    session.training_paused()
    view.sample()
    assert panel.visible_buttons() == ["Resume", "Stop"]
    assert "[ PAUSED ]" in panel.screen()
    panel.find("Resume").click()
    assert recorder.names == ["resume"]


def test_the_eval_detail_is_shown_under_the_run_status() -> None:
    """The bar is pinned at N/N during the final sweep, so the detail says why."""
    session = _session()
    panel, view, _ = build(session, "training")
    ready(session)
    session.training_initializing(3000)
    session.training_running()
    session.publish_training(step=3000, total=3000)
    session.set_detail("evaluating camera 12/194")
    view.sample()
    assert "evaluating camera 12/194" in panel.screen()


def test_a_training_failure_is_reported_in_the_training_section() -> None:
    session = _session()
    panel, view, _ = build(session, "training")
    ready(session)
    session.training_initializing(3000)
    session.training_failed("building the run: out of memory")
    view.sample()
    screen = panel.screen()
    assert "x Training failed" in screen and "out of memory" in screen


# --------------------------------------------------------------------------
# result: export / retrain / move on
# --------------------------------------------------------------------------

def test_a_finished_run_presents_the_result_and_its_next_actions() -> None:
    session = _session()
    panel, view, _ = build(session, "result")
    ready(session)
    trained(session)
    view.sample()
    screen = panel.screen()
    assert "+ Training complete" in screen
    assert "30,000 iterations · 1m 34s" in screen
    assert "7.82M splats" in screen
    assert panel.visible_buttons() == ["Export PLY", "Train Again (fresh run)",
                                       "Load New Dataset", "Clear Session"]


def test_train_again_says_what_it_does_rather_than_being_ambiguous() -> None:
    session = _session()
    panel, view, _ = build(session, "result")
    ready(session)
    trained(session)
    view.sample()
    screen = panel.screen()
    assert "Train Again (fresh run)" in screen
    assert "discards this reconstruction" in screen
    assert "from scratch" in screen


def test_exporting_names_the_file_and_reports_where_it_went() -> None:
    session = _session()
    panel, view, recorder = build(session, "result")
    ready(session)
    trained(session)
    session.set_export_name("kitchen_30000.ply")
    view.sample()
    field = panel.of("input_text")[0]
    assert field.info["value"]() == "kitchen_30000.ply"
    panel.find("Export PLY").click()
    assert recorder.names == ["export"]

    session.result_exported("D:/out/kitchen_30000.ply", 412.5)
    view.sample()
    screen = panel.screen()
    assert "+ Exported" in screen
    assert "D:/out/kitchen_30000.ply" in screen
    assert "412.5 MB" in screen
    assert "Show in folder" in panel.visible_buttons()


def test_an_export_failure_appears_next_to_the_export_button() -> None:
    session = _session()
    panel, view, _ = build(session, "result")
    ready(session)
    trained(session)
    session.result_failed("Could not write D:/ro/out.ply: permission denied")
    view.sample()
    screen = panel.screen()
    assert "x Export failed" in screen and "permission denied" in screen


def test_a_stale_result_warns_that_the_parameters_moved_on() -> None:
    session = _session()
    panel, view, _ = build(session, "result")
    ready(session)
    trained(session)
    session.mark_config_edited()
    view.sample()
    assert "! Parameters changed since this run" in panel.screen()


def test_the_result_section_is_quiet_before_anything_is_trained() -> None:
    session = _session()
    panel, view, _ = build(session, "result")
    ready(session)
    view.sample()
    assert "No reconstruction yet." in panel.screen()
    assert panel.visible_buttons() == []


# --------------------------------------------------------------------------
# confirmations, rendered inline by the section that owns them
# --------------------------------------------------------------------------

def test_a_replace_question_appears_in_the_dataset_section() -> None:
    session = _session()
    panel, view, recorder = build(session, "dataset")
    ready(session)
    trained(session)
    session.ask(Confirm(key="dataset", title="Load a new dataset?",
                        body="The current reconstruction has not been exported "
                             "and will be discarded.",
                        command="load_dropped", confirm_label="Discard and load",
                        alt_label="Export first", alt_command="export"))
    view.sample()
    screen = panel.screen()
    assert "! Load a new dataset?" in screen
    assert "has not been exported" in screen
    assert panel.visible_buttons() == ["Discard and load", "Export first", "Cancel"]
    panel.find("Cancel").click()
    assert recorder.names == ["cancel"]


def test_a_restart_question_appears_in_the_result_section() -> None:
    session = _session()
    panel, view, recorder = build(session, "result")
    ready(session)
    trained(session)
    session.ask(Confirm(key="result", title="Restart training?",
                        body="The current reconstruction is discarded.",
                        command="train!", confirm_label="Discard and restart"))
    view.sample()
    assert "! Restart training?" in panel.screen()
    panel.find("Discard and restart").click()
    assert recorder.names == ["confirm"]
    # ... and the actions it is about are not clickable underneath it.
    assert not panel.find("Export PLY").enabled


def test_a_question_without_an_alternative_offers_only_two_answers() -> None:
    session = _session()
    panel, view, _ = build(session, "result")
    ready(session)
    trained(session)
    session.result_exported("D:/out/kitchen.ply", 1.0)
    session.ask(Confirm(key="result", title="Restart training?", body="b",
                        command="train!", confirm_label="Restart"))
    view.sample()
    assert panel.visible_buttons().count("Cancel") == 1
    assert "Export first" not in panel.visible_buttons()


# --------------------------------------------------------------------------
# configuration: separated from status, progressively disclosed
# --------------------------------------------------------------------------

def test_the_common_parameters_are_visible_and_the_rest_are_behind_a_toggle() -> None:
    session = _session()
    panel, view, _ = build(session, "configuration")
    screen = panel.screen()
    assert "Applies to the next run." in screen
    for key in ("iters", "densify_mode", "sh_degree", "lambda_ssim"):
        assert SETTINGS_MAP[key].caption in screen
    # The full table is hidden until asked for.
    groups = [w for w in panel.of("section") if w.label.startswith("SCHEDULE")]
    assert groups and not groups[0].shown

    session.toggle_config_detail(True)
    view.sample()
    assert groups[0].shown


def test_every_editable_parameter_is_offered_exactly_once() -> None:
    session = _session()
    panel, _, _ = build(session, "configuration")
    controls = [w for w in panel.widgets
                if w.kind in ("combo", "checkbox", "drag_int", "drag_float")]
    # +1 for the disclosure checkbox itself.
    assert len(controls) == len(EDITABLE) + 1


def test_a_parameter_edit_writes_the_store_and_marks_the_result_stale() -> None:
    session = _session()
    panel, view, recorder = build(session, "configuration")
    session.toggle_config_detail(True)
    view.sample()
    field = next(w for w in panel.of("drag_int") if w.info["value"]() == 3000)
    field.change(9000)
    assert session.setup.get("iters") == 9000
    assert "on_config_change" in recorder.names


def test_parameters_grey_out_while_a_run_consumes_them() -> None:
    session = _session()
    panel, view, _ = build(session, "configuration")
    ready(session)
    session.training_initializing(3000)
    session.training_running()
    view.sample()
    controls = [w for w in panel.widgets
                if w.kind in ("combo", "drag_int", "drag_float")]
    assert controls and not any(w.enabled for w in controls)
    assert "Locked while a run is training." in panel.screen()


def test_a_launch_only_parameter_is_shown_read_only_with_the_reason() -> None:
    session = _session()
    panel, view, _ = build(session, "configuration")
    session.toggle_config_detail(True)
    view.sample()
    screen = panel.screen()
    assert "backward pixel stride" in screen
    assert "launch-only" in screen


def test_a_collapsed_group_still_says_what_it_holds() -> None:
    session = _session()
    panel, view, _ = build(session, "configuration")
    session.toggle_config_detail(True)
    view.sample()
    densify = next(w for w in panel.of("section")
                   if w.label.startswith("DENSIFICATION"))
    assert "structure_aware" in densify.label


# --------------------------------------------------------------------------
# notices + diagnostics
# --------------------------------------------------------------------------

def test_notices_fill_from_the_bottom_so_the_newest_never_moves() -> None:
    session = _session()
    session.notify("first", OK)
    notices = session.snapshot().notices
    assert notice_line(notices, 2, 3) == "+ first"
    assert notice_line(notices, 0, 3) == "" and notice_line(notices, 1, 3) == ""
    session.notify("second", WARN)
    notices = session.snapshot().notices
    assert notice_line(notices, 1, 3) == "+ first"
    assert notice_line(notices, 2, 3) == "! second"


def test_the_header_shows_acknowledgements_and_they_disappear() -> None:
    clock = type("C", (), {"t": 100.0, "__call__": lambda s: s.t})()
    session = Session(MODE_HUB, clock=clock)
    panel, view, _ = build(session, "header")
    session.notify("Training started", OK)
    view.sample()
    assert "+ Training started" in panel.screen()
    clock.t += 60.0
    view.sample()
    assert "+ Training started" not in panel.screen()


def test_the_log_is_off_by_default_and_summarized_in_its_header() -> None:
    session = _session()
    panel, view, recorder = build(session, "diagnostics")
    session.log("colmap chatter")
    view.sample()
    assert "colmap chatter" not in panel.screen()
    assert "DIAGNOSTICS" in panel.of("section")[0].label
    assert "1 lines" in panel.of("section")[0].label

    session.toggle_log(True)
    view.sample()
    assert "colmap chatter" in panel.screen()
    panel.find("Clear log").click()
    assert recorder.names == ["clear_log"]


# --------------------------------------------------------------------------
# viewer section
# --------------------------------------------------------------------------

def test_the_viewer_section_reports_the_camera_it_is_looking_through() -> None:
    panel = RecordingPanel()
    viewer_section(panel, CameraControls(
        names=lambda: ["free fly", "cam_0001"], selection=lambda: 1,
        select=lambda _i: None, label=lambda: "cam_0001 / 148",
        speed=lambda: 1.25, fovy=lambda: 45.0))
    screen = panel.screen()
    assert "cam_0001" in screen
    assert "1.25" in screen and "45 deg" in screen


# --------------------------------------------------------------------------
# LOSS graph
# --------------------------------------------------------------------------

def test_bars_are_built_from_one_repeated_glyph() -> None:
    """The panel font is proportional, so only a single-glyph bar has a width
    proportional to its value — a mixed-glyph plot would shear per row."""
    drawn = bars([0.0, 0.5, 1.0], rows=3, cells=10)
    assert drawn == ["", "#####", "##########"]
    assert all(set(line) <= {"#"} for line in drawn)


def test_bars_autoscale_to_the_window_and_the_range_is_printed() -> None:
    """Late in a run the loss moves in the third decimal; a zero-based axis would
    be a flat wall. The scale line is what keeps that honest."""
    drawn = bars([0.0302, 0.0301, 0.0300], rows=3, cells=10)
    assert drawn[0] == "#" * 10 and drawn[-1] == ""
    assert bar_scale([0.0302, 0.0300]) == "scale 0.03000 .. 0.03020"


def test_a_flat_or_tiny_window_still_draws_something_truthful() -> None:
    assert bars([0.5] * 5, rows=3) == ["#", "#", "#"]
    assert bar_scale([0.5, 0.5]) == "flat at 0.50000"
    assert bars([], rows=3) == []


def test_long_series_compress_into_buckets_without_dropping_spikes() -> None:
    flat = [1.0] * 20
    spiked = list(flat)
    spiked[10] = 5.0
    assert bucket(flat, 4) == [1.0, 1.0, 1.0, 1.0]
    assert max(bucket(spiked, 4)) > 1.0        # the spike raised its bucket
    assert len(bucket(list(range(97)), 10)) == 10


def test_the_loss_section_plots_what_the_reporter_published() -> None:
    session = _session()
    panel, view, _ = build(session, "loss")
    for step in range(1, 61):
        session.publish_training(step=step * 10, loss=0.03 - step * 0.0004)
    view.sample()
    screen = panel.screen()
    assert "#" in screen                                  # a curve is drawn
    assert "scale 0.00600 .. 0.02960" in screen
    assert "iters 10-600" in screen and "60 samples" in screen
    assert "-80% over the window" in panel.of("section")[0].label


def test_the_empty_graph_says_why_it_is_empty() -> None:
    """--log-every 0 is the default, so this is the state most runs start in: the
    graph must not read as broken."""
    session = _session()
    panel, view, _ = build(session, "loss")
    view.sample()
    screen = panel.screen()
    assert "no samples" in panel.of("section")[0].label
    assert "--log-every 50" in screen
    assert "#" not in screen


def test_history_is_bounded_and_belongs_to_its_run() -> None:
    session = _session()
    for step in range(1, 400):
        session.publish_training(step=step, loss=1.0 / step)
    assert len(session.snapshot().loss_history) == LOSS_HISTORY
    session.training_initializing(3000)
    assert session.snapshot().loss_history == ()


def test_throughput_ticks_do_not_pollute_the_curve() -> None:
    """The reporter publishes rate and loss on different cadences; only a real
    loss value is a sample."""
    session = _session()
    session.publish_training(step=10, loss=0.5)
    session.publish_training(step=11, it_per_s=300.0)
    session.publish_training(step=12, it_per_s=310.0)
    assert session.snapshot().loss_history == ((10, 0.5),)


def test_the_finished_result_reports_no_quality_metric() -> None:
    """THE BUG: the run summary showed the PSNR of the single camera the last
    iteration happened to sample, next to "30,000 iterations", which reads as the
    held-out test score. There is no honest dataset-wide number available without a
    full sweep, so the panel states none."""
    session = _session()
    panel, view, _ = build(session, "result")
    ready(session)
    session.publish_training(step=30_000, total=30_000, psnr=26.13, loss=0.0031)
    trained(session)
    view.sample()
    screen = panel.screen()
    assert "7.82M splats" in screen
    assert "dB" not in screen
    assert "loss" not in screen


def test_a_live_per_camera_number_is_labelled_as_one_camera() -> None:
    """During a run the same PSNR IS useful as a live signal — it just may not sit
    beside the splat count, where it reads as a dataset metric."""
    session = _session()
    panel, view, _ = build(session, "training")
    ready(session)
    session.training_initializing(30_000)
    session.training_running()
    session.publish_training(step=100, total=30_000, gaussians=1_200_000,
                             psnr=21.40, loss=0.0182)
    view.sample()
    lines = [w.rendered() for w in panel.of("text") if w.shown]
    model = next(line for line in lines if "splats" in line)
    sampled = next(line for line in lines if "dB" in line)
    assert "dB" not in model and "loss" not in model     # exact numbers only
    assert "sampled cam" in sampled and "loss 0.0182" in sampled


# --- the progress bar must track what is ACTUALLY running --------------------

def _visible_progress(panel):
    """(fraction, ...) of every progress bar the panel would currently draw."""
    return [w.binds["fraction"]() for w in panel.of("progress") if w.shown]


def test_a_running_step_shows_the_iteration_bar() -> None:
    session = Session()
    session.training_initializing(3000)
    session.training_running()
    session.publish_training(step=1500, total=3000)
    panel, _, _ = build(session, "training")
    assert _visible_progress(panel) == [0.5]


def test_the_eval_sweep_replaces_the_iteration_bar_with_its_own() -> None:
    """THE bug: --eval-every dividing --iters puts the full-camera sweep ON the
    last iteration, so the iteration bar read "3,000 / 3,000  100.0%" for the
    minutes that sweep took — a bar claiming the run was done while it plainly
    was not. Whatever is actually running owns the bar."""
    session = Session()
    session.training_initializing(3000)
    session.training_running()
    session.publish_training(step=3000, total=3000)
    session.begin_activity("Evaluating every camera", 0.0)
    session.set_progress("Evaluating every camera", 0.25,
                         detail="camera 48/194")

    panel, view, _ = build(session, "training")
    bars = _visible_progress(panel)
    assert bars == [0.25], \
        f"expected only the eval bar at 25%, got {bars}"
    assert 1.0 not in bars, "the 100% iteration bar is still on screen"
    # The label says what is running, and the iteration count is still readable
    # as a secondary row rather than as the headline.
    assert view.activity.label == "Evaluating every camera"
    assert view.activity.detail == "camera 48/194"


def test_finalizing_shows_the_finalize_bar_not_a_full_iteration_bar() -> None:
    session = Session()
    session.training_initializing(3000)
    session.training_running()
    session.publish_training(step=3000, total=3000)
    session.training_finalizing("held-out test PSNR")

    panel, _, _ = build(session, "training")
    assert _visible_progress(panel) == [0.0]


def test_indeterminate_work_parks_the_bar_at_zero_rather_than_lying() -> None:
    """Activity.fraction is None when no honest number exists; the bar must not
    invent one (and must not fall back to the iteration fraction)."""
    session = Session()
    session.training_initializing(3000)
    session.training_running()
    session.publish_training(step=3000, total=3000)
    session.begin_activity("Rendering every camera")     # no fraction

    panel, _, _ = build(session, "training")
    assert _visible_progress(panel) == [0.0]


def test_the_iteration_bar_returns_once_the_activity_ends() -> None:
    """The swap is not one-way: when the sweep finishes mid-run the iteration bar
    has to come back, or the rest of the run shows a stale eval bar."""
    session = Session()
    session.training_initializing(3000)
    session.training_running()
    session.publish_training(step=2000, total=3000)
    session.begin_activity("Evaluating every camera", 0.5)
    panel, view, _ = build(session, "training")
    assert _visible_progress(panel) == [0.5]

    session.end_activity()
    view.sample()          # what the render loop does once per frame
    assert _visible_progress(panel) == [2000 / 3000]
