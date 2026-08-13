"""Viewer UI logic: docking/resize math, text fitting, log tail, state policy.

Everything here is deliberately DEVICE-FREE — the parts of the UI that used to be
tangled into the render loop (and therefore only testable by eye) are now pure
functions and small state objects:

  * :mod:`trainer.frontend.viewer.ui.dock`  — the anchored-panel resize fix
  * :mod:`trainer.frontend.viewer.ui.text`  — wrap/elide (nothing gets clipped)
  * :mod:`trainer.frontend.viewer.state`    — log tail + enable/hint policy
  * :mod:`trainer.frontend.settings` — the hub's editable TRAINING SETUP
  * :mod:`trainer.frontend.reconstruction.progress` — determinate progress
"""
from __future__ import annotations

import contextlib

import pytest

from trainer.frontend import settings as settings_module
from trainer.frontend.cli import build_parser
from trainer.frontend.reconstruction.progress import (
    RECON_STAGES,
    reconstruction_parser,
    setup_parser,
)
from trainer.frontend.settings import (
    BOOL,
    SETTINGS_MAP,
    CHOICE,
    FLOAT,
    GROUPS,
    INT,
    SETTINGS,
    Setting,
    TrainingSetup,
    validate,
    viewer_stream_capacity,
)
from trainer.frontend.viewer import state as st
from trainer.frontend.viewer.ui import scale as sc
from trainer.frontend.viewer.ui import text as tx
from trainer.frontend.viewer.ui.dock import PADDING, SCROLLBAR, Dock, Rect
from trainer.frontend.viewer.ui.input import InputMap
from trainer.frontend.viewer.ui.panel import bullets
from trainer.frontend.viewer.ui.sections import (
    SessionActions,
    _progress_caption,
    _training_metrics,
    _training_timing,
    dataset_section,
    status_section,
    training_setup_section,
)

WINDOW = (1600.0, 900.0)


# --------------------------------------------------------------------------
# dock: anchoring + the resize bug
# --------------------------------------------------------------------------

@pytest.mark.parametrize("anchor", ["left", "right", "top", "bottom"])
def test_dock_pins_to_its_edge_and_spans_the_window(anchor):
    w, h = WINDOW
    rect = Dock(anchor, thickness=380.0, min_thickness=340.0).resolve(w, h)
    if anchor == "left":
        assert (rect.x, rect.y, rect.w, rect.h) == (0.0, 0.0, 380.0, h)
    elif anchor == "right":
        assert (rect.x, rect.y, rect.w, rect.h) == (w - 380.0, 0.0, 380.0, h)
    elif anchor == "top":
        assert (rect.x, rect.y, rect.w, rect.h) == (0.0, 0.0, w, 380.0)
    else:
        assert (rect.x, rect.y, rect.w, rect.h) == (0.0, h - 380.0, w, 380.0)


@pytest.mark.parametrize(
    "anchor,dragged",
    [("left", Rect(0.0, 0.0, 520.0, 900.0)),
     ("right", Rect(1600.0 - 520.0, 0.0, 520.0, 900.0)),
     ("top", Rect(0.0, 0.0, 1600.0, 520.0)),
     ("bottom", Rect(0.0, 900.0 - 520.0, 1600.0, 520.0))],
)
def test_dock_adopts_a_resize_of_the_free_edge(anchor, dragged):
    """THE BUG: the old layout re-asserted the size every frame, so the panel
    snapped back to its hard-coded thickness on mouse-up. The drag must stick."""
    w, h = WINDOW
    dock = Dock(anchor, thickness=380.0, min_thickness=340.0)
    dock.resolve(w, h)
    dock.resolve(w, h, dragged)                  # frame with the user's drag
    assert dock.thickness == 520.0
    settled = dock.resolve(w, h, dock.rect)      # ... and every frame after
    assert dock.thickness == 520.0
    assert settled.approx_eq(dock.rect, 1e-6)


def test_dock_does_not_write_while_the_panel_already_sits_there():
    """Suppressing the write is what stops us fighting an in-progress drag."""
    dock = Dock("left", thickness=380.0, min_thickness=340.0)
    rect = dock.resolve(*WINDOW)
    assert dock.needs_write(None) is True
    assert dock.needs_write(rect) is False
    assert dock.needs_write(Rect(0.0, 0.0, rect.w + 5.0, rect.h)) is True


def test_dock_ignores_a_moved_or_collapsed_panel():
    """A title-bar drag (unpinned) or a collapsed/corner-dragged panel must not
    be mistaken for a thickness change — that read a bogus size."""
    w, h = WINDOW
    dock = Dock("left", thickness=380.0, min_thickness=340.0)
    dock.resolve(w, h)
    dock.resolve(w, h, Rect(300.0, 120.0, 520.0, 900.0))   # moved off the edge
    assert dock.thickness == 380.0
    dock.resolve(w, h, Rect(0.0, 0.0, 520.0, 40.0))        # collapsed
    assert dock.thickness == 380.0


def test_dock_clamps_the_adopted_thickness():
    w, h = WINDOW
    dock = Dock("left", thickness=380.0, min_thickness=340.0, max_fraction=0.5)
    dock.resolve(w, h, Rect(0.0, 0.0, 40.0, h))            # dragged tiny
    assert dock.thickness == 340.0
    dock.resolve(w, h, Rect(0.0, 0.0, 5000.0, h))          # dragged huge
    assert dock.thickness == w * 0.5


def test_dock_content_width_excludes_padding_and_scrollbar():
    dock = Dock("left", thickness=380.0, min_thickness=340.0)
    dock.resolve(*WINDOW)
    assert dock.content_width() == 380.0 - 2 * PADDING - SCROLLBAR


def test_dock_rejects_an_unknown_anchor():
    with pytest.raises(ValueError):
        Dock("middle")


# --------------------------------------------------------------------------
# HiDPI: sgl renders the font at the monitor's content scale, so pixels and
# text-metric units are NOT the same thing (this is what clipped every message
# on a 150% display while an offscreen render of the panel looked perfect).
# --------------------------------------------------------------------------

def test_dock_sizes_the_panel_in_pixels_but_measures_text_in_font_units():
    w, h = WINDOW
    plain = Dock("left", thickness=380.0, min_thickness=340.0)
    hidpi = Dock("left", thickness=380.0, min_thickness=340.0, scale=1.5)
    plain.resolve(w, h)
    rect = hidpi.resolve(w, h)

    assert (hidpi.thickness, hidpi.thickness_px) == (380.0, 570.0)
    assert (rect.w, rect.h) == (570.0, h)            # 1.5x the pixels...
    # ... and the text budget stays comparable, because the font is 1.5x wider:
    # only ImGui's raw-pixel padding/scrollbar cost relatively less.
    assert hidpi.content_width() == pytest.approx(
        (570.0 - 2 * PADDING - SCROLLBAR) / 1.5)
    assert hidpi.content_width() > plain.content_width()
    assert hidpi.content_width() < plain.thickness


def test_dock_adopts_a_hidpi_drag_in_pixels():
    """ImGui reports the drag in device pixels; the thickness is logical."""
    w, h = WINDOW
    dock = Dock("left", thickness=380.0, min_thickness=340.0, scale=1.5)
    dock.resolve(w, h)
    dock.resolve(w, h, Rect(0.0, 0.0, 600.0, h))
    assert dock.thickness == pytest.approx(400.0)
    assert dock.thickness_px == pytest.approx(600.0)
    assert dock.needs_write(dock.rect) is False      # settled: nothing to push
    dock.resolve(w, h, Rect(0.0, 0.0, 300.0, h))     # dragged below the minimum
    assert dock.thickness == 340.0
    assert dock.thickness_px == pytest.approx(510.0)


def test_dock_clamps_a_hidpi_panel_against_the_window_in_pixels():
    dock = Dock("left", thickness=380.0, min_thickness=100.0,
                max_fraction=0.5, scale=2.0)
    dock.resolve(1000.0, 600.0, Rect(0.0, 0.0, 900.0, 600.0))
    assert dock.thickness_px == 500.0                # half the window, not 900


def test_display_scale_normalizes_a_dpi_reading():
    assert sc.normalize(96) == 1.0
    assert sc.normalize(144) == 1.5
    assert sc.normalize(240) == 2.5
    assert sc.normalize(0) == 1.0                    # nonsense -> no scaling
    assert sc.normalize(-96) == 1.0
    assert sc.normalize(96 * 99) == sc.MAX_SCALE     # never off a cliff
    assert sc.normalize(48) == 1.0                   # never SHRINK the panel


def test_font_scale_rounds_the_atlas_up_to_a_whole_pixel():
    """The atlas is rasterized at an integer pixel size, so a 1.5x display draws
    ImGui's 13px font at 20px = 1.538x — measured live, and the reason a plain
    1.5x still clipped. Rounding UP only ever wraps a line early."""
    assert sc.font_scale(1.0) == 1.0
    assert sc.font_scale(1.5) == pytest.approx(20.0 / 13.0)
    assert sc.font_scale(2.0) == 2.0
    for dpi_scale in (1.0, 1.25, 1.5, 1.75, 2.0, 3.0):
        assert sc.font_scale(dpi_scale) >= dpi_scale
        assert sc.font_scale(dpi_scale) - dpi_scale < 1.0 / sc.BASE_FONT_PX


def test_display_scale_is_usable_even_when_the_query_fails(monkeypatch):
    monkeypatch.setattr(sc, "query_dpi", lambda: None)
    assert sc.display_scale() == 1.0
    monkeypatch.setattr(sc, "query_dpi", lambda: 144.0)
    assert sc.display_scale() == pytest.approx(20.0 / 13.0)


# --------------------------------------------------------------------------
# text: wrap / elide (the "messages break down" symptom)
# --------------------------------------------------------------------------

def test_width_is_proportional_not_a_character_count():
    assert tx.width("lll") < tx.width("mmm")
    assert tx.width("") == 0.0


def test_wrap_never_exceeds_the_width():
    line = ("[recon] .recon-tools/colmap-cuda/bin/colmap.exe feature_extractor "
            "--database_path D:/data/garden/reconstruction/_recon_work/database.db "
            "--image_path D:/data/garden/images --SiftExtraction.use_gpu 1")
    lines = tx.wrap(line, 340.0)
    assert len(lines) > 1
    assert all(tx.fits(candidate, 340.0) for candidate in lines)
    # No content is lost, only whitespace/breaks are introduced.
    assert "".join(lines).replace(" ", "") == line.replace(" ", "")


def test_wrap_splits_an_unbreakable_token_at_separators():
    token = "D:/datasets/360_v2/bicycle/reconstruction/sparse/0/points3D.bin"
    lines = tx.wrap(token, 120.0)
    assert all(tx.fits(candidate, 120.0) for candidate in lines)
    assert "".join(lines) == token
    assert lines[0].endswith("/")     # broke after a separator, not mid-word


def test_wrap_elides_the_last_line_instead_of_dropping_content():
    lines = tx.wrap(" ".join(f"word{i}" for i in range(40)), 200.0, max_lines=3)
    assert len(lines) == 3
    assert lines[-1].endswith(tx.ELLIPSIS)
    assert all(tx.fits(candidate, 200.0) for candidate in lines)


def test_wrap_keeps_explicit_newlines_as_line_breaks():
    assert tx.wrap("a\nb\nc", 300.0) == ["a", "b", "c"]


def test_elide_start_keeps_the_path_leaf():
    path = "D:/datasets/360_v2/bicycle/reconstruction"
    out = tx.elide_start(path, 120.0)
    assert out.startswith(tx.ELLIPSIS) and out.endswith("reconstruction")
    assert tx.fits(out, 120.0)


def test_elide_end_and_middle_fit_and_mark_the_cut():
    long = "structure_tensor_texture_for_camera_number_seventeen"
    assert tx.fits(tx.elide_end(long, 150.0), 150.0)
    assert tx.elide_end(long, 150.0).endswith(tx.ELLIPSIS)
    middle = tx.elide_middle(long, 150.0)
    assert tx.fits(middle, 150.0)
    assert middle.startswith("str") and middle.endswith("teen")


def test_short_text_is_returned_untouched():
    for fn in (tx.elide_start, tx.elide_end, tx.elide_middle):
        assert fn("short", 500.0) == "short"


# --------------------------------------------------------------------------
# LogBuffer: what used to be a single overwritten status string
# --------------------------------------------------------------------------

def test_log_buffer_keeps_only_the_last_carriage_return_redraw():
    log = st.LogBuffer()
    log.append("downloading [10/360 MB]\rdownloading [90/360 MB]")
    assert log.last() == "downloading [90/360 MB]"


def test_log_buffer_drops_blank_lines_and_collapses_repeats():
    log = st.LogBuffer()
    assert log.append("   ") is None
    log.append("same")
    log.append("same")
    log.append("same")
    assert len(log) == 1
    assert log.last() == "same  (x3)"
    log.append("different")
    assert len(log) == 2


def test_log_buffer_is_bounded_and_tails_newest_last():
    log = st.LogBuffer(capacity=5)
    for i in range(20):
        log.append(f"line {i}")
    assert len(log) == 5
    assert log.tail(3) == ("line 17", "line 18", "line 19")
    assert log.tail(0) == ()


# --------------------------------------------------------------------------
# SessionState / SessionSnapshot: the UI policy, in one place
# --------------------------------------------------------------------------

def test_commands_are_one_shot():
    session = st.SessionState(st.MODE_HUB)
    session.drop("D:/data/garden")
    session.request("reconstruct")
    assert session.take_dropped() == "D:/data/garden"
    assert session.take_dropped() is None
    assert session.take_command() == "reconstruct"
    assert session.take_command() is None


def test_matcher_selection_round_trips_through_the_combo_index():
    session = st.SessionState(st.MODE_HUB)
    assert session.snapshot().matcher_index == 0
    session.set_matcher_index(2)
    assert session.matcher == st.MATCHERS[2]
    assert session.snapshot().matcher_index == 2
    session.set_matcher("not-a-matcher")            # ignored, not crashed
    assert session.matcher == st.MATCHERS[2]


def test_button_enablement_follows_the_pipeline_state():
    session = st.SessionState(st.MODE_HUB)
    snap = session.snapshot()
    assert (snap.can_setup, snap.can_reconstruct, snap.can_train) == (True, False, False)

    session.set_dataset("D:/data/raw", st.KIND_IMAGES)
    assert session.snapshot().can_reconstruct is False   # tools not installed yet
    session.set_tools_ready(True)
    assert session.snapshot().can_reconstruct is True
    assert session.snapshot().can_setup is False        # the venv is already there

    session.set_phase(st.PHASE_RECONSTRUCT)             # busy: nothing is clickable
    snap = session.snapshot()
    assert (snap.busy, snap.can_setup, snap.can_reconstruct) == (True, False, False)

    session.set_phase(st.PHASE_IDLE)
    session.set_scene_dir("D:/data/raw/reconstruction")
    assert session.snapshot().can_train is True


def test_a_finished_run_can_be_exported_relaunched_or_cleared():
    """The session outlives the run: its result is still in memory, so the panel
    offers the .ply, another Train on the same scene, and a reset."""
    session = st.SessionState(st.MODE_HUB)
    session.set_dataset("D:/data/scene", st.KIND_COLMAP)
    session.set_scene_dir("D:/data/scene")
    assert session.snapshot().can_export is False      # nothing trained yet

    session.set_phase(st.PHASE_TRAINING)
    assert session.snapshot().can_clear is False       # not while it runs
    session.set_trained_model(True)
    session.set_phase(st.PHASE_DONE)
    snap = session.snapshot()
    assert (snap.can_export, snap.can_train, snap.can_clear) == (True, True, True)
    assert snap.can_configure is True                  # editable for the next run


def test_clearing_the_session_forgets_the_dataset_but_not_the_tools():
    session = st.SessionState(st.MODE_HUB)
    session.set_tools_ready(True)
    session.set_matcher_index(2)
    session.set_dataset("D:/data/scene", st.KIND_COLMAP)
    session.set_scene_dir("D:/data/scene")
    session.set_trained_model(True)
    session.publish_training(step=3000, total=3000, psnr=27.5)
    session.set_phase(st.PHASE_DONE)
    session.log("a line worth keeping")
    session.request("export")

    session.clear_session()
    snap = session.snapshot()
    assert (snap.dataset_path, snap.dataset_kind) == ("", st.KIND_NONE)
    assert (snap.scene_dir, snap.has_trained_model) == ("", False)
    assert (snap.phase, snap.stats.psnr, snap.stats.step) == (st.PHASE_IDLE, None, 0)
    assert (snap.can_export, snap.can_train, snap.can_clear) == (False, False, False)
    # A queued click must not act on the session that was just cleared.
    assert session.take_command() is None
    # Install state, choices and the record of what happened all survive.
    assert (snap.tools_ready, snap.matcher) == (True, st.MATCHERS[2])
    assert snap.log[-1] == "a line worth keeping"
    assert "drag" in snap.hint()


def test_hints_tell_the_user_what_to_do_next():
    hub = st.SessionState(st.MODE_HUB)
    assert "drag" in hub.snapshot().hint()
    hub.set_dataset("D:/data/raw", st.KIND_IMAGES)
    assert "Set up" in hub.snapshot().hint()
    hub.set_tools_ready(True)
    assert "Reconstruct" in hub.snapshot().hint()
    hub.set_dataset("D:/data/scene", st.KIND_COLMAP)
    assert "Train" in hub.snapshot().hint()
    hub.set_phase(st.PHASE_DONE)
    assert "export" in hub.snapshot().hint()        # the run is over, not the session
    hub.set_phase(st.PHASE_ERROR)
    assert "log" in hub.snapshot().hint()
    assert "gaussians" in st.SessionState(st.MODE_TRAIN).snapshot().hint()

    # A CLI run has no panel to offer any of that, so it says what IS true there.
    cli = st.SessionState(st.MODE_TRAIN)
    cli.set_phase(st.PHASE_DONE)
    assert "viewer stays live" in cli.snapshot().hint()


def test_training_stats_drive_a_determinate_progress_bar():
    session = st.SessionState()
    session.set_phase(st.PHASE_TRAINING)
    session.publish_training(step=750, total=3000, it_per_s=50.0)
    snap = session.snapshot()
    assert snap.progress.active is True
    assert snap.progress.percent() == 25
    assert snap.progress.label == "iter 750/3000"
    assert snap.stats.eta_s == pytest.approx((3000 - 750) / 50.0)


def test_publish_training_merges_independent_ticks():
    """A throughput tick and a loss/psnr tick publish separately; neither may
    erase the other's numbers."""
    session = st.SessionState()
    session.publish_training(step=10, total=100, it_per_s=12.0)
    session.publish_training(loss=0.01, psnr=24.5)
    stats = session.snapshot().stats
    assert (stats.step, stats.total, stats.it_per_s) == (10, 100, 12.0)
    assert (stats.loss, stats.psnr) == (0.01, 24.5)


def test_leaving_a_busy_phase_retires_the_progress_bar():
    session = st.SessionState(st.MODE_HUB)
    session.set_phase(st.PHASE_SETUP)
    session.set_progress("installing glomap", 0.7)
    assert session.snapshot().progress.active is True
    session.set_phase(st.PHASE_IDLE)
    assert session.snapshot().progress.active is False


def test_log_tail_is_published_into_the_snapshot():
    session = st.SessionState(log_lines=3)
    for i in range(6):
        session.log(f"line {i}")
    snap = session.snapshot()
    assert snap.log == ("line 3", "line 4", "line 5")
    assert snap.message == "line 5"
    session.clear_log()
    assert session.snapshot().log == ()


def test_state_view_caches_one_snapshot_per_frame():
    session = st.SessionState()
    view = session.view()
    session.log("published after sample")
    assert view.log == ()          # the frame's snapshot is frozen...
    view.sample()
    assert view.log == ("published after sample",)   # ... until the next frame


def test_format_duration():
    assert st.format_duration(None) == "--"
    assert st.format_duration(0) == "0:00"
    assert st.format_duration(75) == "1:15"
    assert st.format_duration(3725) == "1:02:05"


# --------------------------------------------------------------------------
# reconstruction progress: text -> determinate fraction
# --------------------------------------------------------------------------

def test_recon_progress_follows_the_stage_markers_monotonically():
    parser = reconstruction_parser()
    fractions = []
    for stage in RECON_STAGES:
        parser.feed(f"[stage] {stage.marker}")
        fractions.append(parser.progress.fraction)
    assert fractions == sorted(fractions)
    assert parser.progress.fraction == 1.0
    assert parser.progress.label == "done"


def test_recon_progress_interpolates_within_a_stage_and_never_rewinds():
    parser = reconstruction_parser()
    parser.feed("[stage] features")
    start = parser.progress.fraction
    parser.feed("Processed file [50/100]")
    half = parser.progress.fraction
    assert start < half < 0.30
    parser.feed("[stage] matching")
    parser.feed("[stage] features")        # a late line from an earlier stage
    assert parser.progress.fraction >= half
    assert parser.progress.label == "matching images"


def test_lines_that_are_not_stages_or_counters_move_nothing():
    parser = reconstruction_parser()
    assert parser.feed("colmap chatter nobody parses") is None
    assert parser.feed("[stage] a stage we do not know") is None
    assert parser.progress.fraction == 0.0


def test_setup_progress_tracks_the_install_stages():
    parser = setup_parser()
    for marker, expect in (("colmap", "installing colmap"),
                           ("glomap", "installing glomap"),
                           ("done", "tools ready")):
        parser.feed(f"[stage] {marker}")
        assert parser.progress.label == expect
    assert parser.progress.fraction == 1.0


def test_the_big_colmap_download_moves_the_bar_while_it_runs():
    """Its CUDA zip is ~360 MB — the longest single step of a setup, so the MB
    counter the installer prints has to interpolate inside the stage."""
    parser = setup_parser()
    parser.feed("[stage] colmap")
    start = parser.progress.fraction
    parser.feed("[recon] downloading [180/360 MB]")
    assert start < parser.progress.fraction < 0.75
    assert parser.progress.label == "installing colmap"


# --------------------------------------------------------------------------
# panel-facing formatting + self-documenting input map
# --------------------------------------------------------------------------

def test_progress_caption_shows_the_phase_when_idle_and_the_percent_when_busy():
    session = st.SessionState(st.MODE_HUB)
    view = session.view()
    assert _progress_caption(view) == "idle"
    session.set_phase(st.PHASE_RECONSTRUCT)
    session.set_progress("matching images", 0.42)
    view.sample()
    assert _progress_caption(view) == "matching images  —  42%"


def test_training_readouts_degrade_gracefully_before_any_numbers():
    assert _training_metrics(st.TrainingStats()) == "no metrics yet"
    assert "-- it/s" in _training_timing(st.TrainingStats())
    assert "eta --" in _training_timing(st.TrainingStats())
    stats = st.TrainingStats(step=100, total=1000, gaussians=123456,
                             it_per_s=20.0, loss=0.0123, psnr=25.5, elapsed_s=5.0)
    metrics = _training_metrics(stats)
    assert "loss 0.0123" in metrics and "25.50 dB" in metrics
    assert "123,456 splats" in metrics
    assert "eta 0:45" in _training_timing(stats)


def test_input_map_documents_itself_for_the_help_section():
    keys = InputMap().hold("KEY_W", "w").document("WASD", "move")
    assert keys.describe() == [("WASD", "move")]
    assert "WASD" in bullets(keys.describe())


def test_input_map_tracks_held_keys_and_fires_presses():
    fired = []
    keys = InputMap().hold("KEY_W", "w")
    keys.press("KEY_C", lambda: fired.append("c"))

    keys.handle(_KeyEvent("KEY_W", "press"))
    assert keys.held == {"w"}
    keys.handle(_KeyEvent("KEY_W", "repeat"))
    assert keys.held == {"w"}
    keys.handle(_KeyEvent("KEY_W", "release"))
    assert keys.held == set()

    keys.handle(_KeyEvent("KEY_C", "press"))
    keys.handle(_KeyEvent("KEY_C", "release"))
    assert fired == ["c"]

    keys.handle(_KeyEvent("KEY_W", "press"))
    keys.release_all()
    assert keys.held == set()


# --------------------------------------------------------------------------
# section builders: the declared widgets and what drives them
# --------------------------------------------------------------------------

def test_status_section_shows_the_bar_only_while_something_runs():
    """An idle 0% bar reads as 'stuck at zero', so it is hidden when idle."""
    session = st.SessionState(st.MODE_HUB)
    view = session.view()
    panel = _RecordingPanel()
    status_section(panel, view)

    bar = panel.only("progress")
    assert bar.binds["visible"]() is False
    session.set_phase(st.PHASE_SETUP)
    session.set_progress("installing colmap", 0.5)
    view.sample()
    assert bar.binds["visible"]() is True
    assert bar.binds["fraction"]() == pytest.approx(0.5)


def test_dataset_section_buttons_gate_and_drive_the_session():
    """A control is declared with an ``enabled`` getter and its action — the
    render loop never touches either (the extensibility this refactor is for)."""
    session = st.SessionState(st.MODE_HUB)
    view = session.view()
    panel = _RecordingPanel()
    dataset_section(panel, view, SessionActions(
        setup=lambda: session.request("setup"),
        reconstruct=lambda: session.request("reconstruct"),
        train=lambda: session.request("train"),
        select_matcher=session.set_matcher_index))

    train = panel.only("button", label="Train")
    assert train.binds["enabled"]() is False
    session.set_dataset("D:/data/scene", st.KIND_COLMAP)
    session.set_scene_dir("D:/data/scene")
    view.sample()
    assert train.binds["enabled"]() is True
    train.click()
    assert session.take_command() == "train"

    panel.only("combo").change(2)          # the matcher dropdown drives state
    assert session.matcher == st.MATCHERS[2]


def test_dataset_section_exposes_the_install_time_cuda_build_choice():
    """Which COLMAP/GLOMAP builds get installed is an INSTALL-TIME choice (the CUDA
    and CPU zips are separate downloads), so it belongs next to the Set up button."""
    session = st.SessionState(st.MODE_HUB)
    view = session.view()
    panel = _RecordingPanel()
    dataset_section(panel, view, SessionActions(
        setup=lambda: None, reconstruct=lambda: None, train=lambda: None,
        select_matcher=session.set_matcher_index,
        set_recon_cuda=session.set_recon_cuda))

    box = panel.only("checkbox", caption="CUDA recon builds")
    assert box.info["value"]() is True      # CUDA by default: it is the only GPU path
    box.change(False)
    assert session.recon_cuda is False
    session.set_phase(st.PHASE_SETUP)
    view.sample()
    assert box.binds["enabled"]() is False     # can't switch build mid-install


def test_setup_is_offered_only_while_the_recon_tools_are_missing():
    """The button downloads the COLMAP / GLOMAP builds; with those already on disk,
    pressing it again would just re-fetch hundreds of MB. ``tools_ready`` is probed
    from disk when the hub starts, so the button stays greyed out across restarts
    too — and so does the build checkbox, which has no next install left to steer."""
    session = st.SessionState(st.MODE_HUB)
    view = session.view()
    panel = _RecordingPanel()
    dataset_section(panel, view, SessionActions(
        setup=lambda: session.request("setup"),
        reconstruct=lambda: None, train=lambda: None,
        select_matcher=session.set_matcher_index,
        set_recon_cuda=session.set_recon_cuda))

    button = panel.only("button", label="Set up recon tools")
    assert button.binds["enabled"]() is True
    button.click()
    assert session.take_command() == "setup"

    session.set_tools_ready(True)
    view.sample()
    assert button.binds["enabled"]() is False
    assert panel.only("checkbox",
                      caption="CUDA recon builds").binds["enabled"]() is False


def test_the_dataset_section_offers_the_post_training_actions():
    """What happens when a run finishes: the .ply can be written, the same scene
    re-trained (the button says so), or the whole session thrown away."""
    session = st.SessionState(st.MODE_HUB)
    view = session.view()
    panel = _RecordingPanel()
    dataset_section(panel, view, SessionActions(
        setup=lambda: None, reconstruct=lambda: None,
        train=lambda: session.request("train"),
        select_matcher=session.set_matcher_index,
        export_ply=lambda: session.request("export"),
        clear=lambda: session.request("clear")))

    export = panel.only("button", label="Export .ply")
    clear = panel.only("button", label="Clear session")
    train = panel.only("button", label="Train")
    assert export.binds["enabled"]() is False       # nothing trained yet
    assert clear.binds["enabled"]() is False        # and nothing to forget

    session.set_dataset("D:/data/scene", st.KIND_COLMAP)
    session.set_scene_dir("D:/data/scene")
    session.set_trained_model(True)
    session.set_phase(st.PHASE_DONE)
    view.sample()
    assert export.binds["enabled"]() is True
    assert clear.binds["enabled"]() is True
    assert train.binds["enabled"]() is True
    assert train.binds["label"]() == "Train again"  # a re-run, and it says so

    export.click()
    assert session.take_command() == "export"
    clear.click()
    assert session.take_command() == "clear"


def test_a_panel_without_the_post_training_actions_declares_no_such_buttons():
    """They are optional wiring, so a caller that offers neither gets neither —
    the buttons cannot be there without something behind them."""
    session = st.SessionState(st.MODE_HUB)
    panel = _RecordingPanel()
    dataset_section(panel, session.view(), SessionActions(
        setup=lambda: None, reconstruct=lambda: None, train=lambda: None,
        select_matcher=session.set_matcher_index))
    labels = [w.info["label"] for w in panel.widgets if w.kind == "button"]
    assert labels == ["Set up recon tools", "Reconstruct", "Train"]


# --------------------------------------------------------------------------
# TRAINING SETUP: the hub's editable hyperparameters (spec table + section)
# --------------------------------------------------------------------------

CONTROLS = ("checkbox", "combo", "drag_int", "drag_float")
EDITABLE = tuple(s for s in SETTINGS if not s.launch_only)


def _build_setup_section(mode: str = st.MODE_HUB):
    session = st.SessionState(mode)
    view = session.view()
    setup = TrainingSetup()
    panel = _RecordingPanel()
    training_setup_section(panel, view, setup)
    return session, view, setup, panel


def _controls(panel) -> list[_Declared]:
    return [w for w in panel.widgets if w.kind in CONTROLS]


def _texts(panel) -> list[str]:
    return [w.info["value"]() if callable(w.info["value"]) else w.info["value"]
            for w in panel.widgets if w.kind == "text"]


def _probe(setting):
    """A value distinct from the store's neutral default, in the WIDGET's units
    (a combo hands back an index; everything else hands back a value)."""
    if setting.kind == BOOL:
        return True
    if setting.kind == CHOICE:
        return len(setting.choices) - 1
    return setting.maximum


def _expected(setting):
    if setting.kind == BOOL:
        return True
    if setting.kind == CHOICE:
        return setting.choices[-1]
    return setting.coerce(setting.maximum)


def test_setup_section_groups_related_options_in_the_declared_order():
    """The grouping IS the feature: related knobs sit together, in a deliberate
    order (GROUPS) rather than whatever order the table happens to list dests."""
    *_, panel = _build_setup_section()
    titles = [w.info["title"] for w in panel.widgets if w.kind == "section"]
    assert titles == ["TRAINING SETUP"] + [title for _, title in GROUPS]


def test_setup_section_offers_every_editable_setting_exactly_once():
    *_, panel = _build_setup_section()
    assert len(_controls(panel)) == len(EDITABLE)


def test_every_setup_control_reads_and_writes_the_shared_store():
    """Each widget is bound to the store, not to a copy: it writes through its
    callback and pulls its displayed value back out every frame."""
    _, _, setup, panel = _build_setup_section()
    for setting, widget in zip(EDITABLE, _controls(panel), strict=True):
        assert widget.binds["enabled"]() is True
        widget.change(_probe(setting))
        assert setup.get(setting.setting_name) == _expected(setting)
        pulled = widget.info["value"]()
        assert pulled == (setup.index_of(setting.setting_name) if setting.kind == CHOICE
                          else setup.get(setting.setting_name))


def test_setup_controls_lock_while_a_run_is_being_built_or_trained():
    """A run resolves its schedules and buffer sizes when it is BUILT, so an edit
    made while it runs would be silently ignored — grey them out instead."""
    session, view, _, panel = _build_setup_section()
    for phase in (st.PHASE_SETUP, st.PHASE_RECONSTRUCT, st.PHASE_TRAINING):
        session.set_phase(phase)
        view.sample()
        assert all(w.binds["enabled"]() is False for w in _controls(panel))
    session.set_phase(st.PHASE_IDLE)
    view.sample()
    assert all(w.binds["enabled"]() is True for w in _controls(panel))


def test_setup_controls_reopen_after_a_run_so_it_can_be_relaunched():
    """The values are read again by the NEXT run, which is how the same scene is
    re-trained with different parameters — locking them at DONE would have made
    'Train again' a repeat of the previous settings and nothing else."""
    session, view, _, panel = _build_setup_section()
    session.set_phase(st.PHASE_DONE)
    view.sample()
    assert all(w.binds["enabled"]() is True for w in _controls(panel))
    assert "applied when you press Train" in _texts(panel)[0]


def test_setup_is_never_editable_outside_the_hub():
    """A plain training run already had its args consumed before the window."""
    assert st.SessionState(st.MODE_TRAIN).snapshot().can_configure is False
    _, _, _, panel = _build_setup_section(st.MODE_TRAIN)
    assert all(w.binds["enabled"]() is False for w in _controls(panel))


def test_a_compile_time_setting_is_shown_read_only_not_silently_ignored():
    """--backward-pixel-stride is a shader define injected when the DEVICE is
    created, i.e. before the hub panel exists. It must not look editable."""
    stride = SETTINGS_MAP["backward_pixel_stride"]
    assert stride.launch_only is True
    _, _, setup, panel = _build_setup_section()
    row = [t for t in _texts(panel) if t.startswith(stride.caption)]
    assert len(row) == 1
    assert stride.display(setup.get(stride.setting_name)) in row[0]
    assert stride.note in row[0]               # ... and says why it is read-only


def test_the_shared_settings_are_internally_consistent():
    """Every row names a group that exists and has a drag range containing its own
    default, and every row actually reached the parser."""
    assert validate() == []


def test_every_shared_row_reaches_the_cli_exactly_as_declared():
    """``Setting.add_to`` is the ONLY declaration of these flags, so what argparse
    ended up with must match the row: same flag string, type, default, choices."""
    declared = {action.dest: action for action in build_parser()._actions}
    for setting in SETTINGS:
        action = declared[setting.setting_name]
        assert setting.flag in action.option_strings
        assert action.default == setting.default
        assert tuple(action.choices or ()) == setting.choices
        if setting.kind == BOOL:
            # --x/--no-x carries no type; the pair is the declaration.
            assert f"--no-{setting.setting_name.replace('_', '-')}" in action.option_strings
        else:
            assert action.type is setting.type


def test_the_widget_kind_follows_the_shared_declaration():
    """One table, so the panel cannot render a combo for a float or vice versa."""
    for setting in SETTINGS:
        if setting.choices:
            assert setting.kind == CHOICE
        elif setting.type is bool:
            assert setting.kind == BOOL
        else:
            assert setting.kind == (INT if setting.type is int else FLOAT)


def test_the_store_opens_on_the_cli_defaults():
    """An un-seeded store is what a plain `v2_train.py` run would use — including
    the booleans (--mip-splatting defaults ON, and the panel must say so)."""
    defaults = vars(build_parser().parse_args([]))
    assert TrainingSetup().values() == {s.setting_name: defaults[s.setting_name] for s in SETTINGS}


def test_a_row_the_cli_never_mounted_is_reported(monkeypatch):
    """A knob added to the table but not to build_parser() would give a widget
    that edits an attribute the run never reads — validate() must say so."""
    ghost = Setting("no_such_flag", "ghost", "perf", int, 0, minimum=0, maximum=1)
    monkeypatch.setattr(settings_module, "SETTINGS", (*SETTINGS, ghost))
    problems = validate()
    assert len(problems) == 1
    assert "no_such_flag" in problems[0] and "add_to(parser)" in problems[0]


def test_a_default_outside_the_drag_range_is_reported(monkeypatch):
    """The range is the panel's own, so it is the one thing that can still
    contradict the shared default — silently clamping it on first touch."""
    bad = Setting("iters", "iterations", "schedule", int, 3000,
                  minimum=5000, maximum=9000, speed=25)
    monkeypatch.setattr(settings_module, "SETTINGS", (bad,))
    assert ["iters: default 3000 is outside the panel range [5000, 9000]"] == validate()


def test_setup_round_trips_through_the_real_cli_args():
    """``from_args``/``apply_to`` are the only coupling to argparse, so they are
    checked against the REAL parser: a renamed dest must fail here."""
    args = build_parser().parse_args(["--iters", "7000", "--sh-degree", "2"])
    setup = TrainingSetup.from_args(args)
    assert setup.get("iters") == 7000
    assert setup.get("sh_degree") == 2
    assert setup.apply_to(args) == []          # seeded from args: nothing changed

    setup.set("iters", 9000)
    setup.set("lambda_ssim", 0.5)
    changes = setup.apply_to(args)
    assert (args.iters, args.lambda_ssim) == (9000, pytest.approx(0.5))
    assert changes == ["iters: 7,000 -> 9,000", "lambda_ssim: 0.200 -> 0.500"]


def test_apply_to_never_writes_a_launch_only_setting():
    args = build_parser().parse_args(["--backward-pixel-stride", "4"])
    setup = TrainingSetup.from_args(args)
    setup.set("backward_pixel_stride", 16)     # e.g. a stale value in the store
    assert setup.apply_to(args) == []
    assert args.backward_pixel_stride == 4     # the value compiled into the shader


def test_setting_coerce_clamps_instead_of_rejecting():
    """The panel's drag fields let you type a number; out of range must clamp,
    never raise on the render thread."""
    iters = SETTINGS_MAP["iters"]
    assert iters.coerce(-5) == iters.minimum
    assert iters.coerce(10 ** 9) == iters.maximum
    mode = SETTINGS_MAP["densify_mode"]
    assert mode.coerce("not-a-mode") == mode.choices[0]
    assert SETTINGS_MAP["mip_splatting"].coerce(0) is False


def test_choice_settings_round_trip_through_the_combo_index():
    setup = TrainingSetup()
    choices = SETTINGS_MAP["sh_degree"].choices
    setup.set_index("sh_degree", 3)
    assert setup.get("sh_degree") == 3
    assert setup.index_of("sh_degree") == 3
    setup.set_index("sh_degree", len(choices))  # out of range wraps, never raises
    assert setup.get("sh_degree") == choices[0]


def test_viewer_capacity_can_only_be_raised_above_the_trainers():
    """A snapshot is a raw GPU->GPU copy of the trainer's buffers AND its splat
    counter, so a viewer smaller than the trainer would read past its own
    storage: --stream-max-splats can only ever raise the capacity."""
    assert viewer_stream_capacity(10_000_000, 7_000_000) == 10_000_000
    assert viewer_stream_capacity(1_000_000, 7_000_000) == 7_000_000


class _Declared:
    """One widget a section asked for, plus the getters bound to it."""

    def __init__(self, kind: str, **info) -> None:
        self.kind = kind
        self.binds = info.pop("binds", {})
        self.info = info

    def click(self) -> None:
        self.info["on_click"]()

    def change(self, value) -> None:
        self.info["on_change"](value)


class _RecordingPanel:
    """Stand-in for :class:`Panel` that records a section's declarations, so the
    wiring is testable without a GPU device."""

    def __init__(self) -> None:
        self.widgets: list[_Declared] = []

    def only(self, kind: str, **match) -> _Declared:
        found = [w for w in self.widgets if w.kind == kind
                 and all(w.info.get(k) == v for k, v in match.items())]
        assert len(found) == 1, f"expected one {kind} {match}, got {len(found)}"
        return found[0]

    @contextlib.contextmanager
    def section(self, title: str, **binds):
        self.widgets.append(_Declared("section", title=title, binds=binds))
        yield

    def text(self, value="", *, lines=1, mode="wrap", visible=None) -> None:
        self.widgets.append(_Declared("text", value=value, lines=lines, mode=mode,
                                      binds={} if visible is None
                                      else {"visible": visible}))

    def spacer(self) -> None:
        self.widgets.append(_Declared("spacer"))

    def log(self, lines_getter, *, lines=8) -> None:
        self.widgets.append(_Declared("log", getter=lines_getter, lines=lines))

    def progress(self, fraction, *, caption=None, **binds) -> None:
        if caption is not None:
            self.text(caption)
        self.widgets.append(_Declared("progress", binds={"fraction": fraction, **binds}))

    def button(self, label, on_click, **binds) -> None:
        # A dynamic label is bound like any other getter (see Panel.button), so it
        # stays live; ``label`` keeps the build-time text for matching in only().
        if not isinstance(label, str):
            binds.setdefault("label", label)
        self.widgets.append(_Declared(
            "button", label=label if isinstance(label, str) else label(),
            on_click=on_click, binds=binds))

    def combo(self, caption, items, value, on_change, **binds) -> None:
        if caption:
            self.text(caption)
        self.widgets.append(_Declared("combo", items=items, value=value,
                                      on_change=on_change, binds=binds))

    def checkbox(self, caption, value, on_change, **binds) -> None:
        self.widgets.append(_Declared("checkbox", caption=caption, value=value,
                                      on_change=on_change, binds=binds))

    def slider_float(self, caption, value, minimum, maximum, on_change,
                     fmt="%.3f", **binds) -> None:
        self.widgets.append(_Declared("slider", value=value, on_change=on_change,
                                      binds=binds))

    slider_int = slider_float

    def drag_int(self, caption, value, on_change, *, minimum=0, maximum=0,
                 speed=1.0, fmt="%d", **binds) -> None:
        if caption:
            self.text(caption)
        self.widgets.append(_Declared("drag_int", value=value, on_change=on_change,
                                      minimum=minimum, maximum=maximum, binds=binds))

    def drag_float(self, caption, value, on_change, *, minimum=0.0, maximum=0.0,
                   speed=0.01, fmt="%.3f", **binds) -> None:
        if caption:
            self.text(caption)
        self.widgets.append(_Declared("drag_float", value=value, on_change=on_change,
                                      minimum=minimum, maximum=maximum, binds=binds))


class _KeyEvent:
    """Minimal stand-in for ``spy.KeyboardEvent`` (no window/device needed)."""

    def __init__(self, key: str, kind: str) -> None:
        self.key = key
        self._kind = kind

    def is_key_press(self) -> bool:
        return self._kind == "press"

    def is_key_repeat(self) -> bool:
        return self._kind == "repeat"

    def is_key_release(self) -> bool:
        return self._kind == "release"
