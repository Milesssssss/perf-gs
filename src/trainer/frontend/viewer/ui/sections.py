# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The panel's sections: one per concept, in lifecycle order, built once and pulling state each frame.

No section derives policy: a button's ``enabled`` is a ``can_*`` property, never a local conjunction of booleans.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
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
    note,
    path_row,
    row,
    section_summary,
    status_line,
    titled,
    toasts,
)
from trainer.frontend.viewer.ui.panel import ELIDE_START, Panel, bullets


@dataclass(frozen=True)
class Actions:
    """Everything the panel can ask the orchestrator to do.

    Callbacks only post intent and return immediately, so the render thread never waits on the orchestrator.
    """

    # dataset
    browse: Callable[[], None] | None = None
    install_tools: Callable[[], None] = lambda: None
    reconstruct: Callable[[], None] = lambda: None
    select_matcher: Callable[[int], None] = lambda _i: None
    set_recon_cuda: Callable[[bool], None] = lambda _v: None
    new_dataset: Callable[[], None] = lambda: None
    configure_next_run: Callable[[], None] = lambda: None
    back_to_result: Callable[[], None] = lambda: None
    # training
    start_training: Callable[[], None] = lambda: None
    restart_training: Callable[[], None] = lambda: None
    pause: Callable[[], None] = lambda: None
    resume: Callable[[], None] = lambda: None
    stop: Callable[[], None] = lambda: None
    # result
    export: Callable[[], None] = lambda: None
    reveal_export: Callable[[], None] | None = None
    # pending confirmations (one at a time, resolved by the section that owns it)
    confirm: Callable[[], None] = lambda: None
    confirm_alt: Callable[[], None] = lambda: None
    cancel: Callable[[], None] = lambda: None
    # configuration: an edit marks the on-screen result stale (drives the "parameters changed" hint).
    on_config_change: Callable[[], None] = lambda: None
    # diagnostics
    toggle_log: Callable[[bool], None] = lambda _v: None
    clear_log: Callable[[], None] = lambda: None


@dataclass(frozen=True)
class CameraControls:
    """Camera readouts + the camera-set and snap-camera selectors."""

    sets: Callable[[], Sequence[str]]
    set_selection: Callable[[], int]
    select_set: Callable[[int], None]
    names: Callable[[], Sequence[str]]
    selection: Callable[[], int]          # 0 == free-fly, k == camera k-1
    select: Callable[[int], None]
    label: Callable[[], str]
    speed: Callable[[], float]
    fovy: Callable[[], float]


def header(panel: Panel, view: View) -> None:
    """Always-visible, non-collapsible state header: badge, state, next step, activity, acknowledgements."""
    panel.text(lambda: section_summary(badge(view.status_badge),
               view.dataset.name if view.hub and view.has_result and not view.next_run_setup
               else view.state_label), lines=2)
    note(panel, lambda: view.next_step(), lines=2,
         visible=lambda: bool(view.next_step()) and (not view.hub or
                         (not view.has_result and view.primary_action not in ("drop", "train"))))
    activity_block(panel, lambda: view.activity)
    toasts(panel, view)
    panel.spacer()


def dataset_section(panel: Panel, view: View, actions: Actions) -> None:
    """What am I working on — and, when nothing is loaded, the way to begin."""
    _dataset_empty(panel, view, actions)
    _dataset_working(panel, view)
    _dataset_error(panel, view, actions)
    _reconstruction_controls(panel, view, actions)
    confirm_block(panel, view, "dataset", on_confirm=actions.confirm,
                  on_alt=actions.confirm_alt, on_cancel=actions.cancel)
    with panel.section("DATASET", label=titled("DATASET", lambda: view.dataset.name),
                       visible=lambda: view.dataset_state == DatasetState.READY
                       and not view.training_live and not view.has_result):
        _dataset_details(panel, view)


def _dataset_empty(panel: Panel, view: View, actions: Actions) -> None:
    def empty() -> bool:
        return view.dataset_state == DatasetState.EMPTY

    if actions.browse is not None:
        panel.button("Open dataset folder...", actions.browse, enabled=lambda: view.can_drop,
                     visible=lambda: empty() or (view.dataset_state == DatasetState.READY
                                                 and not view.has_result and not view.training_live))
    note(panel, "You can also drop a COLMAP scene or a folder of photos here.", lines=2, visible=empty)


def _dataset_working(panel: Panel, view: View) -> None:
    """INSPECTING / LOADING / RECONSTRUCTING: acknowledges a drop on the frame it lands, before any disk read."""
    def working() -> bool:
        return view.dataset_state in (DatasetState.INSPECTING, DatasetState.LOADING,
                                      DatasetState.RECONSTRUCTING)

    status_line(panel, lambda: str(view.dataset_state).upper(),
                lambda: view.dataset.name or "...", visible=working)
    panel.text(lambda: f"  detected: {view.dataset.kind_label}",
               visible=lambda: working() and view.dataset.kind != KIND_NONE)


def _dataset_details(panel: Panel, view: View) -> None:
    panel.text(lambda: section_summary(view.dataset.kind_label, view.dataset.summary()), lines=2)
    panel.text(lambda: view.dataset.path, mode=ELIDE_START)

    # Name the derived scene, so "148 images" and the trained scene agree.
    path_row(panel, "  reconstructed scene", lambda: view.dataset.scene_dir,
             visible=lambda: bool(view.dataset.scene_dir)
             and view.dataset.kind == KIND_IMAGES)


def _reconstruction_controls(panel: Panel, view: View, actions: Actions) -> None:
    def needs_recon() -> bool:
        return (view.dataset_state == DatasetState.READY and view.dataset.kind == KIND_IMAGES
                and not view.dataset.scene_dir)

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


def _dataset_error(panel: Panel, view: View, actions: Actions) -> None:
    def failed() -> bool:
        return view.dataset_state == DatasetState.ERROR

    inline_error(panel, lambda: view.dataset_error if failed() else "",
                 title="Could not load dataset")
    path_row(panel, "  folder", lambda: view.dataset.path, visible=failed)
    if actions.browse is not None:
        panel.button("Try another folder", actions.browse,
                     enabled=lambda: view.can_drop, visible=failed)


def training_section(panel: Panel, view: View, actions: Actions, setup: TrainingSetup | None = None) -> None:
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

    if setup is not None:
        for key in RUN_SETTINGS:
            _setting_widget(panel, view, setup, SETTINGS_MAP[key], actions, visible=lambda: view.show_run_settings)
        panel.button("Start Training", actions.start_training, enabled=lambda: view.can_start_training,
                     visible=lambda: view.show_run_settings and not view.has_result)
        panel.button(lambda: "Start with new settings" if view.result.stale_config else "Start from point cloud",
                     actions.restart_training, enabled=lambda: view.can_restart_training,
                     visible=lambda: view.next_run_setup and view.confirm is None)
        panel.button("Back to trained model", actions.back_to_result,
                     visible=lambda: view.next_run_setup and view.confirm is None)
    with (panel.section("TRAINING", label=titled("TRAINING", summary)) if setup is None else nullcontext()):
        if setup is None:
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
    """INITIALIZING through FINALIZING: the complete run status in one block."""
    def live() -> bool:
        return view.training_live

    # An activity outranks the iteration bar, which would otherwise sit at 100% through the final eval sweep.
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
    """What the panel knows exactly: the live splat count."""
    stats = view.stats
    return f"{format_count(stats.gaussians)} splats" if stats.gaussians else "--"


def _sampled_cam_text(view: View) -> str:
    """PSNR/loss for the one camera this iteration drew: a live signal, not the run's score."""
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


def result_section(panel: Panel, view: View, actions: Actions) -> None:
    """Keep saving the finished model separate from configuring its replacement."""
    def finished() -> bool:
        return view.has_result and not view.next_run_setup and view.confirm is None

    panel.text(lambda: section_summary(view.result.summary(), view.result.quality()),
               lines=2, visible=finished)
    panel.button("Export PLY...", actions.export, enabled=lambda: view.can_export, visible=finished)
    path_row(panel, "Saved", lambda: view.result.exported_path,
             visible=lambda: finished() and bool(view.result.exported_path))
    if actions.reveal_export is not None:
        panel.button("Show in folder", actions.reveal_export,
                     visible=lambda: finished() and bool(view.result.exported_path))
    panel.button("Set up another run", actions.configure_next_run,
                 enabled=lambda: view.can_restart_training, visible=finished)
    panel.button("Open another dataset...", actions.new_dataset, enabled=lambda: view.can_drop,
                 visible=finished)
    inline_error(panel, lambda: view.result_error, title="Export failed")
    confirm_block(panel, view, "result", on_confirm=actions.confirm,
                  on_alt=actions.confirm_alt, on_cancel=actions.cancel)


RUN_SETTINGS = ("preset", "iters")


def configuration_section(panel: Panel, view: View, setup: TrainingSetup,
                          actions: Actions, *,
                          show_all: Callable[[], bool],
                          toggle_all: Callable[[bool], None]) -> None:
    """Detailed settings for the next run; the main controls sit beside Start Training."""
    def available() -> bool:
        return view.show_run_settings

    panel.checkbox("Advanced training settings", show_all, toggle_all, visible=available)
    for group_key, title in GROUPS:
        settings = [s for s in group_settings(group_key) if s.setting_name not in RUN_SETTINGS]
        if settings:
            with panel.section(title, label=titled(title, lambda s=settings: _group_summary(setup, s)),
                               visible=lambda: available() and show_all()):
                for setting in settings:
                    _setting_widget(panel, view, setup, setting, actions)


def _group_summary(setup: TrainingSetup, settings: Sequence[Setting]) -> str:
    """A collapsed group still says what it holds (its first two values)."""
    return section_summary(*(s.display(setup.get(s.setting_name))
                             for s in settings[:2]))


def _setting_widget(panel: Panel, view: View, setup: TrainingSetup,
                    setting: Setting, actions: Actions, *, visible: Callable[[], bool] | None = None) -> None:
    """One parameter row, chosen by the setting's kind.

    The ``s=setting`` defaults capture by value, since these lambdas outlive the loop that builds them.
    """
    def editable() -> bool:
        return view.can_configure

    def write(value, s=setting) -> None:
        setup.set(s.setting_name, value)
        actions.on_config_change()

    def write_index(index, s=setting) -> None:
        setup.set_index(s.setting_name, index)
        actions.on_config_change()

    if setting.kind == BOOL:
        panel.checkbox(f"  {setting.caption}",
                       lambda s=setting: bool(setup.get(s.setting_name)),
                       write, enabled=editable)
        note(panel, setting.about, lines=3)
        note(panel, setting.note, lines=2, visible=lambda: bool(setting.note))
        return
    binds = {"enabled": editable}
    if visible is None:
        _caption(panel, setting)
        note(panel, setting.about, lines=3)
        caption = ""
    else:
        caption = setting.caption.capitalize()
        binds["visible"] = visible
    if setting.kind == CHOICE:
        panel.combo(caption, setting.choice_labels(),
                    lambda s=setting: setup.index_of(s.setting_name),
                    write_index, **binds)
    elif setting.kind == INT:
        panel.drag_int(caption, lambda s=setting: int(setup.get(s.setting_name)),
                       write, minimum=int(setting.minimum), maximum=int(setting.maximum),
                       speed=setting.speed, **binds)
    elif setting.kind == FLOAT:
        panel.drag_float("", lambda s=setting: float(setup.get(s.setting_name)),
                         write, minimum=float(setting.minimum),
                         maximum=float(setting.maximum), speed=setting.speed,
                         fmt=setting.fmt or "%.3f", enabled=editable)


def _caption(panel: Panel, setting: Setting) -> None:
    """The caption row above a control, since sgl's fixed item width pushes real labels off the panel."""
    label = (f"  {setting.caption}   ({setting.note})" if setting.note
             else f"  {setting.caption}")
    panel.text(label, lines=2)


def loss_section(panel: Panel, view: View, *, rows: int = 10) -> None:
    """The reporter's rolling MSE of the sampled camera as window-scaled bars: a convergence shape, not a metric.

    Never samples the loss itself (a GPU sync in the hot loop), so its cadence is the reporter's ``--log-every``.
    """
    def values() -> list[float]:
        return [value for _step, value in view.loss_history]

    def summary() -> str:
        history = view.loss_history
        if not history:
            return "no samples"
        return section_summary(f"{history[-1][1]:.5f}", _loss_trend(history))

    with panel.section("LOSS", label=titled("LOSS", summary),
                       visible=lambda: not view.hub or bool(view.loss_history)):
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


def viewer_section(panel: Panel, controls: CameraControls, *, visible=lambda: True) -> None:
    """Camera: what the viewport is looking through."""
    with panel.section("VIEWER", label=titled(
            "VIEWER", lambda: controls.label().split("/")[-1].strip()), visible=visible):
        panel.combo("  cameras", controls.sets, controls.set_selection, controls.select_set,
                    visible=lambda: len(controls.sets()) > 1)
        panel.combo("  view", controls.names, controls.selection, controls.select)
        panel.text(lambda: f"  {controls.label()}", mode=ELIDE_START)
        panel.text(lambda: row("  move speed", f"{controls.speed():.2f}"))
        panel.text(lambda: row("  field of view", f"{controls.fovy():.0f} deg"))


def controls_section(panel: Panel, entries: Sequence[tuple[str, str]], *, visible=lambda: True) -> None:
    """Key bindings, taken from the InputMap itself (see ``InputMap.describe``)."""
    with panel.section("CONTROLS", visible=visible):
        panel.text(bullets(entries), lines=max(1, len(entries)) + 2)


def diagnostics_section(panel: Panel, view: View, actions: Actions, *,
                        show: Callable[[], bool], lines: int = 12) -> None:
    """The log: a developer trail, off until asked for, since user feedback surfaces in each section."""
    with panel.section("DIAGNOSTICS", label=titled(
            "DIAGNOSTICS", lambda: f"{len(view.log)} lines")):
        panel.checkbox("  show log", show, actions.toggle_log)
        panel.log(lambda: view.log[-lines:] if show() else (), lines=lines)
        panel.button("Clear log", actions.clear_log, visible=show)
