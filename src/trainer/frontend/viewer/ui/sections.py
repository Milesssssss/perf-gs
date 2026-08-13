"""The viewer panel's sections, as independent builders.

Each function adds one collapsible section to a :class:`~.panel.Panel` and takes
exactly the state it needs — a :class:`~trainer.frontend.viewer.state.StateView`
for shared session state, and a small controls dataclass for whatever it drives.
The render loop therefore contains NO panel code: it samples the state view once
per frame and calls ``panel.refresh()``.

Adding a readout is one ``panel.text(lambda: ...)`` line inside the relevant
section; adding a control is one ``panel.button``/``combo``/``slider`` line with
an ``enabled=`` getter. Nothing else changes.
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
    Setting,
    TrainingSetup,
    group_settings,
)
from trainer.frontend.viewer.state import (
    MATCHERS,
    PHASE_ERROR,
    StateView,
    TrainingStats,
    format_duration,
)
from trainer.frontend.viewer.ui.panel import ELIDE_START, Panel, bullets


@dataclass(frozen=True)
class SessionActions:
    """What the DATASET section can ask the orchestrator to do."""

    setup: Callable[[], None]
    reconstruct: Callable[[], None]
    train: Callable[[], None]
    select_matcher: Callable[[int], None]
    set_recon_cuda: Callable[[bool], None] | None = None
    # Post-training: write the trained gaussians out, or forget the whole session
    # and start on something else. Optional so a caller that offers neither (a
    # test, or a future read-only panel) simply gets no buttons.
    export_ply: Callable[[], None] | None = None
    clear: Callable[[], None] | None = None


@dataclass(frozen=True)
class CameraControls:
    """Camera readouts + the snap-camera selector."""

    names: Callable[[], Sequence[str]]
    selection: Callable[[], int]          # 0 == free-fly, k == camera k-1
    select: Callable[[int], None]
    label: Callable[[], str]
    speed: Callable[[], float]
    fovy: Callable[[], float]


@dataclass(frozen=True)
class TrainingControls:
    """Training readouts + the pause/start buttons."""

    stats: Callable[[], TrainingStats]
    active: Callable[[], bool]
    paused: Callable[[], bool]
    toggle_pause: Callable[[], None]
    start: Callable[[], None] | None = None


def status_section(panel: Panel, view: StateView, *, log_lines: int = 7) -> None:
    """Headline hint + determinate progress bar + wrapped log tail.

    This replaces the old single ``progress`` string that any long line (a runner
    command, a pip URL) overwrote and then got clipped: the hint says what to do
    next, the bar says how far along we are, and the log keeps the last lines
    intact across as many rows as they need.
    """
    with panel.section("STATUS"):
        # Note the lambdas: a StateView attribute must be re-read every frame
        # (it resolves against the CURRENT snapshot), never captured at build.
        panel.text(lambda: view.hint(), lines=2)
        # The bar appears only while something is running — an idle 0% bar reads
        # as "stuck at zero". The caption stays: it doubles as the phase readout.
        panel.progress(lambda: view.progress.fraction,
                       caption=lambda: _progress_caption(view),
                       visible=lambda: view.progress.active)
        panel.text(lambda: f"log ({len(view.log)} lines, newest last)",
                   visible=lambda: bool(view.log))
        panel.log(lambda: view.log[-log_lines:], lines=log_lines)


def _progress_caption(view: StateView) -> str:
    progress = view.progress
    if not progress.active:
        return view.phase_label
    label = progress.label or view.phase_label
    return f"{label}  —  {progress.percent()}%"


def dataset_section(panel: Panel, view: StateView, actions: SessionActions) -> None:
    """Dataset selection and the reconstruct/train pipeline (hub mode)."""
    with panel.section("DATASET"):
        panel.text(lambda: view.dataset_path or "(none — drag a folder here)",
                   mode=ELIDE_START)
        panel.text(lambda: f"type: {view.dataset_kind}"
                           + ("   tools: ready" if view.tools_ready
                              else "   tools: not installed"))
        panel.combo("feature matcher", list(MATCHERS),
                    lambda: view.matcher_index, actions.select_matcher,
                    enabled=lambda: not view.busy)
        if actions.set_recon_cuda is not None:
            # INSTALL-TIME choice, hence its place next to the Set up button: the
            # CUDA and CPU builds of colmap/glomap are separate downloads, so this
            # decides which ones the next setup fetches — and therefore whether
            # feature matching and bundle adjustment run on the GPU at all. It
            # follows that button's own gate: with the tools installed there is no
            # next setup for it to steer.
            panel.checkbox("CUDA recon builds", lambda: view.recon_cuda,
                           actions.set_recon_cuda,
                           enabled=lambda: view.can_setup)
        panel.button("Set up recon tools", actions.setup,
                     enabled=lambda: view.can_setup)
        panel.button("Reconstruct", actions.reconstruct,
                     enabled=lambda: view.can_reconstruct)
        # "Train again" once a run has finished: the same button re-launches on the
        # same scene with whatever TRAINING SETUP now holds (which is why that
        # section unlocks again — see ``can_configure``).
        panel.button(lambda: "Train again" if view.has_trained_model else "Train",
                     actions.train, enabled=lambda: view.can_train)
        if actions.export_ply is not None:
            panel.button("Export .ply", actions.export_ply,
                         enabled=lambda: view.can_export)
        if actions.clear is not None:
            panel.button("Clear session", actions.clear,
                         enabled=lambda: view.can_clear)
        panel.text("reconstruction needs a folder of images (or images/)",
                   lines=2, visible=lambda: not view.tools_ready)


def training_setup_section(panel: Panel, view: StateView,
                           setup: TrainingSetup) -> None:
    """Editable training hyperparameters, grouped by concept (hub mode).

    Rendered GENERICALLY from :data:`trainer.frontend.settings.SETTINGS` —
    the spec table owns which options exist, their ranges and their grouping, so a
    new knob never needs a line here. Each widget writes straight into the shared
    :class:`TrainingSetup`; the orchestrator applies the whole store to the run's
    args once, when Train is pressed.

    Everything greys out via ``can_configure`` while a run is being built or
    trained, because that is when the values are consumed, and comes back once it
    finishes — the store is what the next run is built from (see the settings
    module docstring).
    """
    with panel.section("TRAINING SETUP"):
        panel.text(lambda: ("applied when you press Train"
                            if view.can_configure
                            else "locked — these are read when the run is built"),
                   lines=2)
        for key, title in GROUPS:
            settings = group_settings(key)
            if not settings:
                continue
            with panel.section(title):
                for setting in settings:
                    _setting_widget(panel, view, setup, setting)


def _setting_widget(panel: Panel, view: StateView, setup: TrainingSetup,
                    setting: Setting) -> None:
    """One row of the setup section, chosen by the setting's kind.

    Note the ``setting=setting`` default arguments: these lambdas outlive the loop
    that builds them, so the loop variable must be captured by value.
    """
    editable = (lambda: False) if setting.launch_only else (lambda: view.can_configure)
    if setting.launch_only:
        # Compile-time (device-creation-time) value: show it, don't pretend to
        # offer it. See the settings module docstring.
        panel.text(lambda s=setting: f"{s.caption}: {s.display(setup.get(s.setting_name))}"
                                     f"  ({s.note})", lines=3)
        return
    if setting.kind == BOOL:
        panel.checkbox(setting.caption, lambda s=setting: bool(setup.get(s.setting_name)),
                       lambda value, s=setting: setup.set(s.setting_name, value),
                       enabled=editable)
        return
    _caption(panel, setting)
    if setting.kind == CHOICE:
        panel.combo("", setting.choice_labels(),
                    lambda s=setting: setup.index_of(s.setting_name),
                    lambda index, s=setting: setup.set_index(s.setting_name, index),
                    enabled=editable)
    elif setting.kind == INT:
        panel.drag_int("", lambda s=setting: int(setup.get(s.setting_name)),
                       lambda value, s=setting: setup.set(s.setting_name, value),
                       minimum=int(setting.minimum), maximum=int(setting.maximum),
                       speed=setting.speed, enabled=editable)
    elif setting.kind == FLOAT:
        panel.drag_float("", lambda s=setting: float(setup.get(s.setting_name)),
                         lambda value, s=setting: setup.set(s.setting_name, value),
                         minimum=float(setting.minimum),
                         maximum=float(setting.maximum), speed=setting.speed,
                         fmt=setting.fmt or "%.3f", enabled=editable)


def _caption(panel: Panel, setting: Setting) -> None:
    """The caption row above a control (sgl's hard-coded item width pushes a real
    widget label off the panel — see the panel module docstring). Two rows of room
    so a caption plus its note can wrap instead of being clipped."""
    label = (f"{setting.caption}   ({setting.note})" if setting.note
             else setting.caption)
    panel.text(label, lines=2)


def training_section(panel: Panel, view: StateView,
                     controls: TrainingControls) -> None:
    """Live training numbers + pause/start."""
    with panel.section("TRAINING"):
        panel.text(lambda: _training_state(view, controls))
        panel.text(lambda: _training_metrics(controls.stats()))
        panel.text(lambda: _training_timing(controls.stats()))
        if controls.start is not None:
            panel.button("Start training", controls.start,
                         enabled=lambda: not controls.active())
        panel.button(lambda: "Resume training" if controls.paused()
                     else "Pause training", controls.toggle_pause,
                     enabled=controls.active)


def _training_state(view: StateView, controls: TrainingControls) -> str:
    if view.phase == PHASE_ERROR:
        return "stopped"
    if not controls.active():
        return "not started"
    return "PAUSED" if controls.paused() else "running"


def _training_metrics(stats: TrainingStats) -> str:
    parts = []
    if stats.loss is not None:
        parts.append(f"loss {stats.loss:.4f}")
    if stats.psnr is not None:
        parts.append(f"psnr {stats.psnr:.2f} dB")
    if stats.gaussians:
        parts.append(f"{stats.gaussians:,} splats")
    return "   ".join(parts) if parts else "no metrics yet"


def _training_timing(stats: TrainingStats) -> str:
    rate = f"{stats.it_per_s:.1f} it/s" if stats.it_per_s else "-- it/s"
    return (f"{rate}   elapsed {format_duration(stats.elapsed_s)}"
            f"   eta {format_duration(stats.eta_s)}")


def camera_section(panel: Panel, controls: CameraControls) -> None:
    """Snap-camera selector + camera readouts."""
    with panel.section("CAMERA"):
        panel.combo("view", controls.names, controls.selection, controls.select)
        panel.text(controls.label, mode=ELIDE_START)
        panel.text(lambda: f"move speed {controls.speed():.2f}"
                           f"   fov {controls.fovy():.0f}°")


def help_section(panel: Panel, entries: Sequence[tuple[str, str]]) -> None:
    """Key bindings, taken from the InputMap itself (see ``InputMap.describe``)."""
    with panel.section("CONTROLS"):
        # +2 rows of slack: a help row that wraps must not push the last one out.
        panel.text(bullets(entries), lines=max(1, len(entries)) + 2)
