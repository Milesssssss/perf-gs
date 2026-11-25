# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Viewer UI logic: docking/resize math, text fitting, log tail, state policy.

Everything here is deliberately DEVICE-FREE — the parts of the UI that used to be
tangled into the render loop (and therefore only testable by eye) are now pure
functions and small state objects:

  * :mod:`trainer.frontend.viewer.ui.dock`  — the anchored-panel resize fix
  * :mod:`trainer.frontend.viewer.ui.text`  — wrap/elide (nothing gets clipped)
  * :mod:`trainer.frontend.viewer.session`  — the log tail
  * :mod:`trainer.frontend.settings`        — the shared CLI/panel settings table
  * :mod:`trainer.frontend.reconstruction.progress` — determinate progress

The application's own state machine is covered by ``test_session_state.py`` and what
the panel shows in each state by ``test_panel_sections.py``; this file stays on the
toolkit underneath them.
"""
from __future__ import annotations

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
    CHOICE,
    FLOAT,
    INT,
    SETTINGS,
    SETTINGS_MAP,
    Setting,
    TrainingSetup,
    validate,
)
from trainer.frontend.viewer.session import LogBuffer
from trainer.frontend.viewer.ui import scale as sc
from trainer.frontend.viewer.ui import text as tx
from trainer.frontend.viewer.ui.dock import PADDING, SCROLLBAR, Dock, Rect
from trainer.frontend.viewer.ui.input import InputMap
from trainer.frontend.viewer.ui.panel import bullets

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
    log = LogBuffer()
    log.append("downloading [10/360 MB]\rdownloading [90/360 MB]")
    assert log.last() == "downloading [90/360 MB]"


def test_log_buffer_drops_blank_lines_and_collapses_repeats():
    log = LogBuffer()
    assert log.append("   ") is None
    log.append("same")
    log.append("same")
    log.append("same")
    assert len(log) == 1
    assert log.last() == "same  (x3)"
    log.append("different")
    assert len(log) == 2


def test_log_buffer_is_bounded_and_tails_newest_last():
    log = LogBuffer(capacity=5)
    for i in range(20):
        log.append(f"line {i}")
    assert len(log) == 5
    assert log.tail(3) == ("line 17", "line 18", "line 19")
    assert log.tail(0) == ()


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


# --------------------------------------------------------------------------
# TRAINING SETUP: the hub's editable hyperparameters (spec table + section)
# --------------------------------------------------------------------------

CONTROLS = ("checkbox", "combo", "drag_int", "drag_float")


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


def test_choosing_a_preset_applies_it_and_balanced_restores_the_defaults():
    setup = TrainingSetup()
    preset = SETTINGS_MAP["preset"]
    assert preset.choices == ("ultrafast", "balanced", "quality", "quality+")
    setup.set("splat_kernel", "opaque_core")
    setup.set("lr_recipe", "default")
    for name in ("ultrafast", "quality", "quality+", "balanced"):
        setup.set_index("preset", preset.choice_index(name))
        cli = build_parser().parse_args(["--preset", name])
        assert setup.values() == {s.setting_name: getattr(cli, s.setting_name) for s in SETTINGS}
    assert setup.values() == TrainingSetup().values()


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
    assert changes == ["iters: 7,000 -> 9,000", "lambda_ssim: 0.300 -> 0.500"]


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
