# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The panel's sections: one per concept, in lifecycle order.

    header            what state am I in, what just happened, what next
    DATASET           what am I working on          (drop target when empty)
    TRAINING          what is happening right now   (run status + controls)
    RESULT            what exists now               (export / train again)
    CONFIGURATION     what the NEXT run will use
    VIEWER            camera
    CONTROLS          keys
    DIAGNOSTICS       the log, off by default

Status, configuration and result are deliberately three sections rather than one
mixed list: "327 it/s" and "iterations: 30,000" answer different questions, and the
old panel put them a line apart. Every readout comes from
:class:`~trainer.frontend.viewer.session.Snapshot`, so no section derives policy of
its own — a button's ``enabled`` is a ``can_*`` property, never a local conjunction
of booleans.

Sections are built once and PULL from getters each frame (see the panel module), so
the render loop contains no UI code and a state transition needs no rebuild.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from trainer.frontend.settings import (
    BOOL,
    CHOICE,
    FLOAT,
    GROUPS,
    INT,
    SETTINGS_MAP,
    Setting,
    TrainingSetup,
    group_settings,
)
from trainer.frontend.viewer.session import (
    KIND_IMAGES,
    KIND_NONE,
    MATCHERS,
    DatasetState,
    ResultState,
    ToolsState,
    TrainingState,
    View,
    format_count,
    format_duration,
    format_rate,
)
from trainer.frontend.viewer.ui.components import (
    GLYPHS,
    activity_block,
    badge,
    bar_scale,
    chart,
    confirm_block,
    inline_error,
    metric,
    name_field,
    note,
    path_row,
    row,
    section_summary,
    status_line,
    titled,
    toasts,
)
from trainer.frontend.viewer.ui.components import (
    actions as action_rows,
)
from trainer.frontend.viewer.ui.panel import ELIDE_START, Panel, bullets


@dataclass(frozen=True)
class Actions:
    """Everything the panel can ask the orchestrator to do.

    One dataclass rather than one per section: this IS the application's command
    surface, and keeping it in a single place makes it obvious when a state has no
    action attached to it. Every callback is one-shot — it posts intent and returns
    immediately, so the render thread never waits on the orchestrator.
    """

    # dataset
    browse: Callable[[], None] | None = None
    install_tools: Callable[[], None] = lambda: None
    reconstruct: Callable[[], None] = lambda: None
    select_matcher: Callable[[int], None] = lambda _i: None
    set_recon_cuda: Callable[[bool], None] = lambda _v: None
    new_dataset: Callable[[], None] = lambda: None
    clear_session: Callable[[], None] = lambda: None
    # training
    start_training: Callable[[], None] = lambda: None
    restart_training: Callable[[], None] = lambda: None
    pause: Callable[[], None] = lambda: None
    resume: Callable[[], None] = lambda: None
    stop: Callable[[], None] = lambda: None
    # result
    export: Callable[[], None] = lambda: None
    set_export_name: Callable[[str], None] = lambda _n: None
    reveal_export: Callable[[], None] | None = None
    # pending confirmations (one at a time, resolved by the section that owns it)
    confirm: Callable[[], None] = lambda: None
    confirm_alt: Callable[[], None] = lambda: None
    cancel: Callable[[], None] = lambda: None
    # configuration: an edit invalidates the match between the on-screen result
    # and what the next run would produce (drives the "parameters changed" hint).
    on_config_change: Callable[[], None] = lambda: None
    # diagnostics
    toggle_log: Callable[[bool], None] = lambda _v: None
    clear_log: Callable[[], None] = lambda: None


@dataclass(frozen=True)
class CameraControls:
    """Camera readouts + the snap-camera selector."""

    names: Callable[[], Sequence[str]]
    selection: Callable[[], int]          # 0 == free-fly, k == camera k-1
    select: Callable[[int], None]
    label: Callable[[], str]
    speed: Callable[[], float]
    fovy: Callable[[], float]


# --------------------------------------------------------------------------
# header: the answer to "what state am I in" and "what did my last action do"
# --------------------------------------------------------------------------

def header(panel: Panel, view: View) -> None:
    """Always-visible state header (deliberately NOT collapsible).

    Three things, in priority order: the state badge, the one-line state, and what
    to do next. Then the current long operation and any transient acknowledgements
    — which is where "did my click do anything" gets answered for every action that
    does not have a section of its own.
    """
    panel.text(lambda: f"{badge(view.status_badge)}  {view.state_label}", lines=2)
    note(panel, lambda: view.next_step(), lines=3)
    activity_block(panel, lambda: view.activity)
    toasts(panel, view)
    panel.spacer()


# --------------------------------------------------------------------------
# DATASET
# --------------------------------------------------------------------------

def dataset_section(panel: Panel, view: View, actions: Actions) -> None:
    """What am I working on — and, when nothing is loaded, the way to begin."""
    def summary() -> str:
        if not view.has_dataset:
            return "empty"
        info = view.dataset
        if view.dataset_state == DatasetState.ERROR:
            return "failed"
        if view.dataset_state != DatasetState.READY:
            return str(view.dataset_state)
        return section_summary(info.name, info.summary())

    with panel.section("DATASET", label=titled("DATASET", summary)):
        _dataset_empty(panel, view, actions)
        _dataset_working(panel, view)
        _dataset_ready(panel, view, actions)
        _dataset_error(panel, view, actions)
        confirm_block(panel, view, "dataset", on_confirm=actions.confirm,
                      on_alt=actions.confirm_alt, on_cancel=actions.cancel)


def _dataset_empty(panel: Panel, view: View, actions: Actions) -> None:
    def empty() -> bool:
        return view.dataset_state == DatasetState.EMPTY

    panel.text("      DROP A DATASET HERE", visible=empty)
    panel.text("   image folder or COLMAP data", visible=empty)
    if actions.browse is not None:
        panel.button("Browse...", actions.browse,
                     enabled=lambda: view.can_drop, visible=empty)
    note(panel, "A folder of images, or one with sparse/ + images/.",
         lines=2, visible=empty)


def _dataset_working(panel: Panel, view: View) -> None:
    """INSPECTING / LOADING / RECONSTRUCTING — the drop is acknowledged here, on the
    frame it lands, before anything has been read from disk."""
    def working() -> bool:
        return view.dataset_state in (DatasetState.INSPECTING, DatasetState.LOADING,
                                      DatasetState.RECONSTRUCTING)

    status_line(panel, lambda: str(view.dataset_state).upper(),
                lambda: view.dataset.name or "...", visible=working)
    panel.text(lambda: f"  detected: {view.dataset.kind_label}",
               visible=lambda: working() and view.dataset.kind != KIND_NONE)


def _dataset_ready(panel: Panel, view: View, actions: Actions) -> None:
    def ready() -> bool:
        return view.dataset_state == DatasetState.READY

    def needs_recon() -> bool:
        # Images not yet reconstructed: the only case where the COLMAP controls
        # mean anything, so the only case that shows them.
        return (ready() and view.dataset.kind == KIND_IMAGES
                and not view.dataset.scene_dir)

    panel.text(lambda: f"{GLYPHS['ok']} {view.dataset.name}", visible=ready)
    note(panel, lambda: view.dataset.kind_label, lines=1, visible=ready)
    note(panel, lambda: view.dataset.summary(), lines=2,
         visible=lambda: ready() and bool(view.dataset.summary()))
    path_row(panel, "  folder", lambda: view.dataset.path, visible=ready)

    # -- images that still need a reconstruction -------------------------
    metric(panel, "  tools", lambda: str(view.tools_state), visible=needs_recon)
    panel.combo("  feature matcher", list(MATCHERS),
                lambda: MATCHERS.index(view.matcher) if view.matcher in MATCHERS else 0,
                actions.select_matcher,
                enabled=lambda: not view.busy, visible=needs_recon)
    panel.checkbox("  CUDA reconstruction builds", lambda: view.recon_cuda,
                   actions.set_recon_cuda,
                   enabled=lambda: view.can_install_tools, visible=needs_recon)
    panel.button("Install reconstruction tools", actions.install_tools,
                 enabled=lambda: view.can_install_tools,
                 visible=lambda: needs_recon()
                 and view.tools_state != ToolsState.READY)
    panel.button("Reconstruct to COLMAP", actions.reconstruct,
                 enabled=lambda: view.can_reconstruct, visible=needs_recon)

    # -- a derived scene: say so, so "148 images" and the trained scene agree --
    path_row(panel, "  reconstructed scene", lambda: view.dataset.scene_dir,
             visible=lambda: ready() and bool(view.dataset.scene_dir)
             and view.dataset.kind == KIND_IMAGES)


def _dataset_error(panel: Panel, view: View, actions: Actions) -> None:
    def failed() -> bool:
        return view.dataset_state == DatasetState.ERROR

    inline_error(panel, lambda: view.dataset_error if failed() else "",
                 title="Could not load dataset")
    path_row(panel, "  folder", lambda: view.dataset.path, visible=failed)
    if actions.browse is not None:
        panel.button("Try another folder", actions.browse,
                     enabled=lambda: view.can_drop, visible=failed)


# --------------------------------------------------------------------------
# TRAINING
# --------------------------------------------------------------------------

def training_section(panel: Panel, view: View, actions: Actions) -> None:
    """What is happening right now: one coherent run status, then its controls."""
    def summary() -> str:
        state = view.training_state
        if state == TrainingState.RUNNING:
            return f"running {GLYPHS['bullet']} {view.stats.percent:.1f}%"
        if state == TrainingState.PAUSED:
            return f"paused at {view.stats.step:,}"
        if state == TrainingState.COMPLETED:
            return "complete"
        if state == TrainingState.FAILED:
            return "failed"
        return str(state)

    with panel.section("TRAINING", label=titled("TRAINING", summary)):
        _training_idle(panel, view, actions)
        _training_live(panel, view, actions)
        inline_error(panel, lambda: view.training_error, title="Training failed")
        confirm_block(panel, view, "training", on_confirm=actions.confirm,
                      on_alt=actions.confirm_alt, on_cancel=actions.cancel)


def _training_idle(panel: Panel, view: View, actions: Actions) -> None:
    def idle() -> bool:
        return view.training_state == TrainingState.IDLE

    panel.text(lambda: "Ready to train." if view.trainable
               else "Load a trainable dataset first.",
               visible=lambda: idle() and not view.has_result)
    metric(panel, "  iterations", lambda: f"{view.stats.total:,}",
           visible=lambda: idle() and view.trainable and view.stats.total > 0)
    panel.button("Start Training", actions.start_training,
                 enabled=lambda: view.can_start_training,
                 visible=lambda: idle() and not view.has_result)


def _training_live(panel: Panel, view: View, actions: Actions) -> None:
    """INITIALIZING through FINALIZING: the complete run status in one block.

    Iteration, progress, rate, elapsed, ETA, splats and quality all live here —
    the numbers a run is judged by should not be spread across the panel.
    """
    def live() -> bool:
        return view.training_live

    # An activity outranks the iteration counter: the eval sweep lands on the last
    # iteration whenever --eval-every divides --iters, so the iteration bar would
    # otherwise sit pinned at 100% for as long as the sweep and the finalize take.
    def stepping() -> bool:
        return live() and not view.activity.active

    def working() -> bool:
        return live() and view.activity.active

    status_line(panel, lambda: str(view.training_state).upper(),
                lambda: _iteration_text(view), visible=stepping)
    panel.progress(lambda: view.stats.fraction,
                   visible=lambda: stepping() and view.stats.total > 0)

    status_line(panel, lambda: str(view.training_state).upper(),
                lambda: view.activity.label, visible=working)
    # fraction None == indeterminate; park at zero rather than invent a position.
    panel.progress(lambda: view.activity.fraction or 0.0, visible=working)
    panel.text(lambda: row("  iteration", _iteration_text(view)), visible=working)
    panel.text(lambda: row("  throughput", _timing_text(view)), visible=live)
    panel.text(lambda: row("  model", _model_text(view)), visible=live)
    panel.text(lambda: row("  sampled cam", _sampled_cam_text(view)),
               visible=lambda: live() and bool(_sampled_cam_text(view)))
    panel.text(lambda: row("  appearance", _appearance_text(view)),
               visible=lambda: live() and bool(_appearance_text(view)))
    note(panel, lambda: view.activity.detail, lines=2,
         visible=lambda: live() and bool(view.activity.detail))

    panel.button("Pause", actions.pause, enabled=lambda: view.can_pause,
                 visible=lambda: live() and not view.can_resume)
    panel.button("Resume", actions.resume, enabled=lambda: view.can_resume,
                 visible=lambda: view.can_resume)
    panel.button("Stop", actions.stop, enabled=lambda: view.can_stop, visible=live)


def _iteration_text(view: View) -> str:
    stats = view.stats
    if stats.total <= 0:
        return "starting"
    return f"{stats.step:,} / {stats.total:,}      {stats.percent:.1f}%"


def _timing_text(view: View) -> str:
    stats = view.stats
    parts = [format_rate(stats.it_per_s), f"{format_duration(stats.elapsed_s)} elapsed"]
    if stats.eta_s is not None:
        parts.append(f"ETA {format_duration(stats.eta_s)}")
    return f"   {GLYPHS['bullet']} ".join(parts)


def _model_text(view: View) -> str:
    """What the panel knows EXACTLY: the live splat count."""
    stats = view.stats
    return f"{format_count(stats.gaussians)} splats" if stats.gaussians else "--"


def _sampled_cam_text(view: View) -> str:
    """PSNR/loss for the ONE camera this iteration drew, labelled as such.

    The reporter gets these by re-rendering that single camera, so they are a live
    signal and not the run's score. Shown under their own label rather than beside
    the splat count, where they read as a dataset-wide metric — an honest average
    needs a sweep over every held-out view (``--evaluation``).
    """
    stats = view.stats
    parts = []
    if stats.psnr is not None:
        parts.append(f"{stats.psnr:.2f} dB")
    if stats.loss is not None:
        parts.append(f"loss {stats.loss:.4f}")
    return f"   {GLYPHS['bullet']} ".join(parts)


def _appearance_text(view: View) -> str:
    stats = view.stats
    parts = []
    if stats.sh_degree:
        parts.append(f"SH {stats.sh_degree}")
    if stats.densify_step:
        label = f"densify @ {stats.densify_step:,}"
        parts.append(f"{label} ({stats.densify_kind})" if stats.densify_kind else label)
    return f"   {GLYPHS['bullet']} ".join(parts)


# --------------------------------------------------------------------------
# RESULT
# --------------------------------------------------------------------------

def result_section(panel: Panel, view: View, actions: Actions) -> None:
    """What exists now. After a run this is the important object, so its actions
    (export, train again, move on) live here rather than under DATASET."""
    def summary() -> str:
        if view.result_state == ResultState.NONE:
            return "none"
        if view.result_state == ResultState.INITIALIZED:
            return "initial points"
        return section_summary(view.result.quality(),
                               "exported" if view.result_state == ResultState.EXPORTED
                               else "not exported")

    with panel.section("RESULT", label=titled("RESULT", summary)):
        panel.text("No reconstruction yet.",
                   visible=lambda: view.result_state == ResultState.NONE)
        note(panel, "Initial points are on screen — nothing has been trained yet.",
             lines=2, visible=lambda: view.result_state == ResultState.INITIALIZED)
        _result_body(panel, view, actions)
        inline_error(panel, lambda: view.result_error, title="Export failed")
        confirm_block(panel, view, "result", on_confirm=actions.confirm,
                      on_alt=actions.confirm_alt, on_cancel=actions.cancel)


def _result_body(panel: Panel, view: View, actions: Actions) -> None:
    def got() -> bool:
        return view.has_result

    panel.text(lambda: f"{GLYPHS['ok']} " + ("Exported"
               if view.result_state == ResultState.EXPORTED else "Training complete"),
               visible=got)
    note(panel, lambda: view.result.summary(), lines=1, visible=got)
    note(panel, lambda: view.result.quality(), lines=1, visible=got)
    note(panel, lambda: f"{GLYPHS['warn']} Parameters changed since this run — "
         f"Train Again will not reproduce it.", lines=3,
         visible=lambda: got() and view.result.stale_config)

    name_field(panel, "  export as", lambda: view.export_name,
               actions.set_export_name, enabled=lambda: view.can_export,
               visible=got)
    panel.button("Export PLY", actions.export, enabled=lambda: view.can_export,
                 visible=got)
    path_row(panel, f"  {GLYPHS['ok']} written", lambda: view.result.exported_path,
             visible=lambda: bool(view.result.exported_path))
    metric(panel, "  size", lambda: f"{view.result.exported_mb:,.1f} MB",
           visible=lambda: view.result.exported_mb > 0)
    if actions.reveal_export is not None:
        panel.button("Show in folder", actions.reveal_export,
                     enabled=lambda: bool(view.result.exported_path) and not view.busy,
                     visible=lambda: bool(view.result.exported_path))

    panel.spacer()
    action_rows(
        panel,
        primary=[("Train Again (fresh run)", actions.restart_training,
                  lambda: view.can_restart_training, got),
                 ("Load New Dataset", actions.new_dataset,
                  lambda: view.can_drop, got)],
        secondary=[("Clear Session", actions.clear_session,
                    lambda: view.can_clear, got)],
    )
    note(panel, "Train Again discards this reconstruction and optimizes the same "
                "dataset from scratch with the current parameters.", lines=3,
         visible=got)


# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

# The knobs shown as the section's own summary, in the order a user reads them.
SUMMARY_KEYS = ("iters", "densify_mode", "sh_degree", "lambda_ssim")


def configuration_section(panel: Panel, view: View, setup: TrainingSetup,
                          actions: Actions, *,
                          show_all: Callable[[], bool],
                          toggle_all: Callable[[bool], None]) -> None:
    """What the NEXT run will use — never mixed with what the current one is doing.

    Progressive disclosure: the four knobs most runs change are always visible; the
    full grouped table is behind one toggle. (This toolkit cannot open or close a
    Group programmatically, so the toggle gates VISIBILITY rather than relying on a
    default-collapsed header.)
    """
    def summary() -> str:
        return section_summary(*(_setting_text(setup, key) for key in SUMMARY_KEYS[:3]))

    with panel.section("CONFIGURATION", label=titled("CONFIGURATION", summary)):
        note(panel, lambda: ("Applies to the next run." if view.can_configure
                             else "Locked while a run is training."), lines=2)
        for key in SUMMARY_KEYS:
            _summary_row(panel, setup, key)
        panel.checkbox("  show all parameters", show_all, toggle_all)
        for group_key, title in GROUPS:
            settings = group_settings(group_key)
            if not settings:
                continue
            with panel.section(title, label=titled(
                    title, lambda s=settings: _group_summary(setup, s)),
                    visible=show_all):
                for setting in settings:
                    _setting_widget(panel, view, setup, setting, actions)


def _summary_row(panel: Panel, setup: TrainingSetup, key: str) -> None:
    setting = SETTINGS_MAP.get(key)
    if setting is None:
        return
    panel.text(lambda s=setting: row(f"  {s.caption}",
                                     s.display(setup.get(s.setting_name))))


def _setting_text(setup: TrainingSetup, key: str) -> str:
    setting = SETTINGS_MAP.get(key)
    return "" if setting is None else setting.display(setup.get(setting.setting_name))


def _group_summary(setup: TrainingSetup, settings: Sequence[Setting]) -> str:
    """A collapsed group still says what it holds (its first two values)."""
    return section_summary(*(s.display(setup.get(s.setting_name))
                             for s in settings[:2]))


def _setting_widget(panel: Panel, view: View, setup: TrainingSetup,
                    setting: Setting, actions: Actions) -> None:
    """One parameter row, chosen by the setting's kind.

    Note the ``s=setting`` default arguments: these lambdas outlive the loop that
    builds them, so the loop variable must be captured by value.
    """
    def editable() -> bool:
        return False if setting.launch_only else view.can_configure

    def write(value, s=setting) -> None:
        setup.set(s.setting_name, value)
        actions.on_config_change()

    def write_index(index, s=setting) -> None:
        setup.set_index(s.setting_name, index)
        actions.on_config_change()

    if setting.launch_only:
        # Fixed when the device was created: show the value, do not pretend to
        # offer it. Saying WHY beats a mysteriously greyed-out control.
        panel.text(lambda s=setting: row(f"  {s.caption}",
                                         s.display(setup.get(s.setting_name))))
        note(panel, f"launch-only: {setting.note}", lines=2)
        return
    if setting.kind == BOOL:
        panel.checkbox(f"  {setting.caption}",
                       lambda s=setting: bool(setup.get(s.setting_name)),
                       write, enabled=editable)
        note(panel, setting.note, lines=2, visible=lambda: bool(setting.note))
        return
    _caption(panel, setting)
    if setting.kind == CHOICE:
        panel.combo("", setting.choice_labels(),
                    lambda s=setting: setup.index_of(s.setting_name),
                    write_index, enabled=editable)
    elif setting.kind == INT:
        panel.drag_int("", lambda s=setting: int(setup.get(s.setting_name)),
                       write, minimum=int(setting.minimum), maximum=int(setting.maximum),
                       speed=setting.speed, enabled=editable)
    elif setting.kind == FLOAT:
        panel.drag_float("", lambda s=setting: float(setup.get(s.setting_name)),
                         write, minimum=float(setting.minimum),
                         maximum=float(setting.maximum), speed=setting.speed,
                         fmt=setting.fmt or "%.3f", enabled=editable)


def _caption(panel: Panel, setting: Setting) -> None:
    """The caption row above a control (sgl's hard-coded item width pushes a real
    widget label off the panel — see the panel module docstring)."""
    label = (f"  {setting.caption}   ({setting.note})" if setting.note
             else f"  {setting.caption}")
    panel.text(label, lines=2)


# --------------------------------------------------------------------------
# VIEWER / CONTROLS / DIAGNOSTICS
# --------------------------------------------------------------------------

def loss_section(panel: Panel, view: View, *, rows: int = 10) -> None:
    """The loss curve, as far as the panel can honestly draw one.

    Two things this deliberately does NOT do. It does not sample the loss itself:
    the only publisher is the reporter's ``report_iteration``, which re-renders the
    camera and reads the whole image back to the host, so its cadence is
    ``--log-every`` and adding a second one would put a GPU sync in the hot loop for
    a decoration. And it does not pretend a zero-based axis: bars are scaled to the
    window (see :func:`~.components.bars`), with the range printed underneath.

    What is plotted is the reporter's ROLLING MSE of a re-render of the ONE camera
    that iteration sampled — neither the L1 + SSIM objective the optimizer descends
    nor an average over the dataset. Good for reading the SHAPE of convergence,
    which is what a curve is for; the caption says so rather than calling it "loss".
    """
    def values() -> list[float]:
        return [value for _step, value in view.loss_history]

    def summary() -> str:
        history = view.loss_history
        if not history:
            return "no samples"
        return section_summary(f"{history[-1][1]:.5f}", _loss_trend(history))

    with panel.section("LOSS", label=titled("LOSS", summary)):
        note(panel, "Rolling MSE of a re-render of the SAMPLED camera, every "
                    "--log-every iters. Not a dataset metric.",
             lines=3, visible=lambda: bool(view.loss_history))
        chart(panel, values, rows=rows)
        note(panel, lambda: bar_scale(values()), lines=1,
             visible=lambda: bool(view.loss_history))
        metric(panel, "  window",
               lambda: f"iters {view.loss_history[0][0]:,}"
                       f"-{view.loss_history[-1][0]:,}"
                       f"  ({len(view.loss_history)} samples)",
               visible=lambda: bool(view.loss_history))
        # The honest empty state: the graph is not broken, nothing is sampling.
        note(panel, "No samples yet. The loss is only read back on a log tick, so "
                    "this stays empty with --log-every 0 (the default). Pass "
                    "--log-every 50 to plot it.", lines=4,
             visible=lambda: not view.loss_history)


def _loss_trend(history: Sequence[tuple[int, float]]) -> str:
    """Change across the window, as a percentage — the number a run is judged by."""
    if len(history) < 2:
        return ""
    first, last = history[0][1], history[-1][1]
    if first <= 0:
        return ""
    delta = 100.0 * (last - first) / first
    return f"{delta:+.0f}% over the window"


def viewer_section(panel: Panel, controls: CameraControls) -> None:
    """Camera: what the viewport is looking through."""
    with panel.section("VIEWER", label=titled(
            "VIEWER", lambda: controls.label().split("/")[-1].strip())):
        panel.combo("  view", controls.names, controls.selection, controls.select)
        panel.text(lambda: f"  {controls.label()}", mode=ELIDE_START)
        panel.text(lambda: row("  move speed", f"{controls.speed():.2f}"))
        panel.text(lambda: row("  field of view", f"{controls.fovy():.0f} deg"))


def controls_section(panel: Panel, entries: Sequence[tuple[str, str]]) -> None:
    """Key bindings, taken from the InputMap itself (see ``InputMap.describe``)."""
    with panel.section("CONTROLS"):
        panel.text(bullets(entries), lines=max(1, len(entries)) + 2)


def diagnostics_section(panel: Panel, view: View, actions: Actions, *,
                        show: Callable[[], bool], lines: int = 12) -> None:
    """The log — secondary by default.

    It stopped being the feedback channel: every state, acknowledgement and failure
    now surfaces in the section it belongs to. What is left here is a developer
    trail, so it is off until asked for.
    """
    with panel.section("DIAGNOSTICS", label=titled(
            "DIAGNOSTICS", lambda: f"{len(view.log)} lines")):
        panel.checkbox("  show log", show, actions.toggle_log)
        panel.log(lambda: view.log[-lines:] if show() else (), lines=lines)
        panel.button("Clear log", actions.clear_log, visible=show)
