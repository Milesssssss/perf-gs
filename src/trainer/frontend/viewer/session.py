# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The application's state machine — what the UI is a view OF.

The panel used to read one ``phase`` string plus a handful of loose booleans, so
"is a dataset loading", "is a run going" and "does a result exist" were all the
same axis and the UI had to infer meaning from combinations that could not all be
represented. Here they are THREE independent lifecycles that happen to run in one
session:

    DatasetState    EMPTY -> INSPECTING -> [RECONSTRUCTING] -> LOADING -> READY
                                                                       -> ERROR
    TrainingState   IDLE -> INITIALIZING -> RUNNING <-> PAUSED -> STOPPING
                                                     -> FINALIZING -> COMPLETED
                                                                   -> FAILED
    ResultState     NONE -> INITIALIZED -> TRAINED -> EXPORTED

plus :class:`ToolsState` for the optional COLMAP/GLOMAP binaries, because
installing them is a long operation that has nothing to do with either lifecycle.

Everything the UI shows is derived from those states in ONE place
(:class:`Snapshot`), so a widget never re-derives policy and two widgets can never
disagree. The orchestrator drives transitions through :class:`Session`; the render
thread only reads snapshots.

Three UI mechanisms live here rather than in the panel, because they are state and
not decoration:

  * :class:`Activity` — the current long-running operation, with a fraction when
    one is knowable and an indeterminate spinner-equivalent when it is not. Every
    operation that can take noticeable time publishes one, which is what makes
    "did my click do anything" answerable.
  * :class:`Notice` — transient acknowledgements (toasts). Successes expire on
    their own; errors stay until something replaces them.
  * :class:`Confirm` — a pending question with the command it will run if
    accepted. There is no modal-dialog widget in this UI toolkit, so a
    confirmation is a state the relevant section renders inline. That also keeps
    destructive actions next to the thing being destroyed.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import StrEnum

from trainer.frontend.settings import TrainingSetup

LOG_CAPACITY = 400
# How long a success/info toast stays up. Errors and warnings do not expire: an
# unresolved failure must not scroll away before it is read.
NOTICE_TTL_S = 6.0
MAX_NOTICES = 3
# Loss samples kept for the graph. The publisher's cadence is --log-every, so this
# is a window of ticks, not of iterations.
LOSS_HISTORY = 240


class DatasetState(StrEnum):
    EMPTY = "empty"
    INSPECTING = "inspecting"        # classifying a dropped folder
    RECONSTRUCTING = "reconstructing"  # images -> COLMAP (long, has progress)
    LOADING = "loading"              # reading the model / uploading the preview
    READY = "ready"
    ERROR = "error"


class TrainingState(StrEnum):
    IDLE = "idle"
    INITIALIZING = "initializing"    # scene load, pipeline build, textures
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    FINALIZING = "finalizing"        # eval sweep / metrics / artifacts
    COMPLETED = "completed"
    FAILED = "failed"


class ResultState(StrEnum):
    NONE = "none"
    INITIALIZED = "initialized"      # init gaussians on screen, never trained
    TRAINED = "trained"
    EXPORTED = "exported"


class ToolsState(StrEnum):
    MISSING = "missing"
    INSTALLING = "installing"
    READY = "ready"


# Session modes: the hub owns the whole lifecycle from an empty window; a plain
# `--scene` run arrives with its dataset already chosen and only ever observes.
MODE_TRAIN = "train"
MODE_HUB = "hub"

# What a dropped folder turned out to be (trainer.frontend.reconstruction).
KIND_NONE = "none"
KIND_COLMAP = "colmap"
KIND_IMAGES = "images"

# COLMAP matcher choices offered by the panel (order == combo order).
MATCHERS = ("exhaustive", "sequential", "vocab_tree")

# Notice severities.
OK = "ok"
INFO = "info"
WARN = "warn"
ERROR = "error"

_DATASET_BUSY = (DatasetState.INSPECTING, DatasetState.RECONSTRUCTING,
                 DatasetState.LOADING)
# PAUSED is deliberately NOT busy: a paused run is waiting for the user, and the
# things a paused session can still do (export, inspect, configure) are exactly
# the things it should not lock.
_TRAINING_BUSY = (TrainingState.INITIALIZING, TrainingState.RUNNING,
                  TrainingState.STOPPING, TrainingState.FINALIZING)
_TRAINING_LIVE = (*_TRAINING_BUSY, TrainingState.PAUSED)


# --------------------------------------------------------------------------
# value records
# --------------------------------------------------------------------------

def format_duration(seconds: float | None) -> str:
    """``1m 23s`` / ``2h 04m`` — reading time, not raw seconds."""
    if seconds is None:
        return "--"
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_count(value: int) -> str:
    """``7.82M`` / ``148k`` / ``940`` — splat counts at a glance, no fake digits."""
    value = int(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 10_000:
        return f"{value / 1_000:.0f}k"
    return f"{value:,}"


def format_rate(it_per_s: float) -> str:
    return f"{it_per_s:,.0f} it/s" if it_per_s >= 1 else "-- it/s"


@dataclass(frozen=True)
class DatasetInfo:
    """What was loaded, in the terms the user dropped it in."""

    path: str = ""
    kind: str = KIND_NONE
    images: int = 0
    width: int = 0
    height: int = 0
    points: int = 0
    # Where a reconstruction of this dataset was written (images -> COLMAP), so
    # the panel can say the training scene is derived rather than the drop itself.
    scene_dir: str = ""

    @property
    def name(self) -> str:
        """The leaf directory — how a user refers to a dataset ("kitchen")."""
        return self.path.replace("\\", "/").rstrip("/").rpartition("/")[2] or self.path

    @property
    def kind_label(self) -> str:
        return {KIND_COLMAP: "COLMAP reconstruction",
                KIND_IMAGES: "image folder"}.get(self.kind, "unrecognized folder")

    def summary(self) -> str:
        """``148 images · 1920x1080 · 1.7M points`` — only what is known."""
        parts: list[str] = []
        if self.images:
            parts.append(f"{self.images:,} images")
        if self.width and self.height:
            parts.append(f"{self.width}x{self.height}")
        if self.points:
            parts.append(f"{format_count(self.points)} points")
        return " · ".join(parts)


@dataclass(frozen=True)
class TrainingStats:
    """Live numbers published by the trainer's reporter."""

    step: int = 0
    total: int = 0
    gaussians: int = 0
    it_per_s: float = 0.0
    loss: float | None = None
    psnr: float | None = None
    elapsed_s: float = 0.0
    sh_degree: int = 0
    densify_step: int = 0            # iteration of the last densification
    densify_kind: str = ""           # what that densification did

    @property
    def fraction(self) -> float:
        return 0.0 if self.total <= 0 else min(1.0, self.step / self.total)

    @property
    def percent(self) -> float:
        return 100.0 * self.fraction

    @property
    def eta_s(self) -> float | None:
        """Seconds left at the current rate, or None when not yet meaningful."""
        if self.total <= 0 or self.it_per_s <= 1e-6 or self.step >= self.total:
            return None
        return (self.total - self.step) / self.it_per_s


@dataclass(frozen=True)
class ResultInfo:
    """The reconstruction that currently exists — the object the user acts on.

    Deliberately carries NO quality metric. The only PSNR the panel has access to
    live is the reporter's re-render of the ONE camera that iteration sampled, and
    showing that next to "30,000 iterations" reads as the dataset-wide test score
    it is not. An honest number costs a sweep over every held-out view — that is
    what ``--evaluation`` produces, into ``final_psnr.json`` / ``eval_curve.jsonl``.
    """

    iterations: int = 0
    target_iterations: int = 0
    wall_s: float = 0.0
    splats: int = 0
    run_dir: str = ""
    exported_path: str = ""
    exported_mb: float = 0.0
    # True once the configuration has been edited since this result was produced:
    # what is on screen no longer matches what the panel would train next.
    stale_config: bool = False

    def summary(self) -> str:
        parts = [f"{self.iterations:,} iterations", format_duration(self.wall_s)]
        return " · ".join(parts)

    def quality(self) -> str:
        """The size of the model, which the panel CAN know exactly."""
        return f"{format_count(self.splats)} splats"


@dataclass(frozen=True)
class Activity:
    """One long-running operation, as the UI needs to show it.

    ``fraction`` is None when no honest number exists — the UI then shows an
    indeterminate readout instead of a bar that would lie about being at 0%.
    """

    label: str = ""
    fraction: float | None = None
    detail: str = ""

    @property
    def active(self) -> bool:
        return bool(self.label)

    @property
    def determinate(self) -> bool:
        return self.fraction is not None

    def percent(self) -> int:
        return int(round(100.0 * min(1.0, max(0.0, self.fraction or 0.0))))

    def caption(self) -> str:
        text = self.label
        if self.detail:
            text = f"{text} · {self.detail}"
        return f"{text}  {self.percent()}%" if self.determinate else text


@dataclass(frozen=True)
class Notice:
    """A transient acknowledgement. Successes expire; failures do not."""

    text: str
    kind: str = OK
    at: float = 0.0

    @property
    def sticky(self) -> bool:
        return self.kind in (WARN, ERROR)

    def marker(self) -> str:
        return {OK: "+", INFO: "-", WARN: "!", ERROR: "x"}.get(self.kind, "-")

    def line(self) -> str:
        return f"{self.marker()} {self.text}"


@dataclass(frozen=True)
class Confirm:
    """A pending question, rendered inline by the section it belongs to.

    ``command`` is what the orchestrator runs if the user accepts, so the UI never
    has to remember what it asked about.
    """

    key: str                      # which section renders it
    title: str
    body: str
    command: str
    confirm_label: str = "Continue"
    alt_label: str = ""           # e.g. "Export first"
    alt_command: str = ""


# --------------------------------------------------------------------------
# log
# --------------------------------------------------------------------------

class LogBuffer:
    """Bounded, de-noised log tail.

    Subprocess output (COLMAP, GLOMAP) arrives as raw lines, sometimes carrying
    carriage-return progress redraws and long runs of identical lines. Only the
    last segment of a ``\\r`` redraw is kept, blank lines are dropped, and an
    immediately repeated line is counted instead of repeated.
    """

    def __init__(self, capacity: int = LOG_CAPACITY) -> None:
        self._lines: deque[str] = deque(maxlen=capacity)
        self._repeats = 0

    def append(self, message: str) -> str | None:
        text = str(message).replace("\t", "  ")
        text = text.split("\r")[-1].strip()
        if not text:
            return None
        if self._lines and self._strip_count(self._lines[-1]) == text:
            self._repeats += 1
            self._lines[-1] = f"{text}  (x{self._repeats + 1})"
            return self._lines[-1]
        self._repeats = 0
        self._lines.append(text)
        return text

    @staticmethod
    def _strip_count(line: str) -> str:
        base, sep, tail = line.rpartition("  (x")
        return base if sep and tail.endswith(")") else line

    def tail(self, count: int) -> tuple[str, ...]:
        return () if count <= 0 else tuple(list(self._lines)[-count:])

    def last(self) -> str:
        return self._lines[-1] if self._lines else ""

    def clear(self) -> None:
        self._lines.clear()
        self._repeats = 0

    def __len__(self) -> int:
        return len(self._lines)


# --------------------------------------------------------------------------
# the snapshot: every question the UI asks, answered once
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Snapshot:
    """Immutable view of the whole session for one UI frame."""

    mode: str = MODE_TRAIN
    dataset_state: DatasetState = DatasetState.EMPTY
    dataset: DatasetInfo = field(default_factory=DatasetInfo)
    dataset_error: str = ""
    training_state: TrainingState = TrainingState.IDLE
    training_error: str = ""
    stats: TrainingStats = field(default_factory=TrainingStats)
    result_state: ResultState = ResultState.NONE
    result: ResultInfo = field(default_factory=ResultInfo)
    result_error: str = ""
    tools_state: ToolsState = ToolsState.MISSING
    activity: Activity = field(default_factory=Activity)
    notices: tuple[Notice, ...] = ()
    confirm: Confirm | None = None
    matcher: str = MATCHERS[0]
    recon_cuda: bool = True
    export_name: str = "output.ply"
    log: tuple[str, ...] = ()
    # Panel disclosure, kept here so it is one testable place rather than mutable
    # closure state in the render thread. This toolkit cannot open or close a
    # collapsing header programmatically, so "collapsed by default" is expressed as
    # visibility the user opts into.
    show_all_config: bool = False
    show_log: bool = False
    # (iteration, loss) samples for the loss graph, oldest first. Bounded: this is
    # a shape-of-the-curve readout, not a record — the run's own jsonl is that.
    loss_history: tuple[tuple[int, float], ...] = ()

    # ---- top-level: what state is the application in? ----

    @property
    def busy(self) -> bool:
        """Whether a long operation owns the session (locks destructive actions)."""
        return (self.dataset_state in _DATASET_BUSY
                or self.training_state in _TRAINING_BUSY
                or self.tools_state == ToolsState.INSTALLING)

    @property
    def training_live(self) -> bool:
        """A run exists and has not finished — includes PAUSED."""
        return self.training_state in _TRAINING_LIVE

    @property
    def has_dataset(self) -> bool:
        return self.dataset_state != DatasetState.EMPTY

    @property
    def has_result(self) -> bool:
        return self.result_state in (ResultState.TRAINED, ResultState.EXPORTED)

    @property
    def unexported(self) -> bool:
        """A finished run whose gaussians only exist in memory."""
        return self.result_state == ResultState.TRAINED

    @property
    def trainable(self) -> bool:
        """A COLMAP scene exists to train on."""
        return bool(self.dataset.scene_dir) and self.dataset_state == DatasetState.READY

    @property
    def state_label(self) -> str:
        """The session in one line — the SUBJECT, not the operation.

        What is happening is the badge plus the activity readout; repeating it here
        (and again in :meth:`next_step`) is how a header ends up saying "Inspecting
        dataset" three times.
        """
        if self.dataset_state in _DATASET_BUSY:
            return self.dataset.name or "working"
        if self.dataset_state == DatasetState.ERROR:
            return "dataset failed to load"
        if self.training_state == TrainingState.FAILED:
            return "training failed"
        if self.training_state == TrainingState.PAUSED:
            return f"paused at iteration {self.stats.step:,}"
        # A live run: the badge already says RUNNING, so the useful complement is
        # WHAT is being trained.
        if self.training_state in _TRAINING_BUSY:
            return self.dataset.name or "training"
        # Export is the more recent news than the completion that preceded it.
        if self.result_state == ResultState.EXPORTED:
            return "result exported"
        if self.training_state == TrainingState.COMPLETED:
            return "training complete"
        if self.trainable:
            return "ready to train"
        if self.dataset_state == DatasetState.READY:
            return "dataset ready — needs reconstruction"
        return "no dataset"

    @property
    def status_badge(self) -> str:
        """A short, scannable state word for the header."""
        if self.tools_state == ToolsState.INSTALLING:
            return "INSTALLING"
        if self.dataset_state in _DATASET_BUSY:
            return self.dataset_state.upper()
        if self.dataset_state == DatasetState.ERROR:
            return "ERROR"
        if self.training_state in (TrainingState.RUNNING, TrainingState.INITIALIZING,
                                  TrainingState.STOPPING, TrainingState.FINALIZING,
                                  TrainingState.PAUSED, TrainingState.FAILED):
            return self.training_state.upper()
        if self.training_state == TrainingState.COMPLETED:
            return "COMPLETE"
        if self.has_dataset:
            return "READY"
        return "EMPTY"

    # ---- what can I do next? (one source for every enabled= binding) ----

    @property
    def hub(self) -> bool:
        return self.mode == MODE_HUB

    @property
    def can_drop(self) -> bool:
        """Drag-and-drop stays live for the whole session — it is how a user moves
        on to the next dataset — but not while an operation owns the GPU."""
        return self.hub and not self.busy and self.confirm is None

    @property
    def can_install_tools(self) -> bool:
        return (self.hub and not self.busy and self.tools_state != ToolsState.READY
                and self.confirm is None)

    @property
    def can_reconstruct(self) -> bool:
        return (self.hub and not self.busy and self.confirm is None
                and self.dataset.kind == KIND_IMAGES
                and self.dataset_state == DatasetState.READY
                and self.tools_state == ToolsState.READY
                and not self.dataset.scene_dir)

    @property
    def can_start_training(self) -> bool:
        """First run of this dataset: no result yet to protect."""
        return (self.hub and self.trainable and not self.busy
                and self.confirm is None and not self.training_live
                and self.result_state in (ResultState.NONE, ResultState.INITIALIZED))

    @property
    def can_restart_training(self) -> bool:
        """A finished result exists; this discards it and trains from scratch."""
        return (self.hub and self.trainable and not self.busy
                and self.confirm is None and not self.training_live
                and self.has_result)

    @property
    def can_pause(self) -> bool:
        return self.training_state == TrainingState.RUNNING

    @property
    def can_resume(self) -> bool:
        return self.training_state == TrainingState.PAUSED

    @property
    def can_stop(self) -> bool:
        """Stopping keeps what has been optimized so far — it is not a cancel."""
        return self.training_state in (TrainingState.RUNNING, TrainingState.PAUSED)

    @property
    def can_export(self) -> bool:
        return (self.hub and self.has_result and not self.busy
                and self.confirm is None)

    @property
    def can_clear(self) -> bool:
        return (self.hub and self.has_dataset and not self.busy
                and self.confirm is None)

    @property
    def can_configure(self) -> bool:
        """Parameters are read when a run is BUILT, so the only time an edit would
        be silently ignored is while one is training. Everything else — installing
        tools, reconstructing, a finished run — stays editable, because those are
        exactly the moments a user sets the NEXT run up."""
        return self.hub and self.training_state not in (
            TrainingState.INITIALIZING, TrainingState.RUNNING,
            TrainingState.STOPPING, TrainingState.FINALIZING)

    @property
    def primary_action(self) -> str:
        """The one action the panel should present most prominently.

        Naming it here rather than in the panel is what keeps "what can I do next"
        answerable from state alone.
        """
        if self.confirm is not None:
            return "confirm"
        if self.busy:
            return "busy"
        if not self.has_dataset:
            return "drop"
        if self.dataset_state == DatasetState.ERROR:
            return "drop"
        if self.can_reconstruct:
            return "reconstruct"
        if self.dataset.kind == KIND_IMAGES and not self.dataset.scene_dir:
            return "install_tools"
        if self.can_resume:
            return "resume"
        if self.training_state == TrainingState.RUNNING:
            return "pause"
        if self.has_result:
            return "export" if self.unexported else "restart"
        if self.can_start_training:
            return "train"
        return "none"

    def next_step(self) -> str:
        """One sentence telling the user what to do next.

        Empty while an operation is running: there is nothing to do but wait, and
        the activity readout already says what for.
        """
        if self.confirm is not None:
            return self.confirm.title
        if self.activity.active:
            return ""
        return {
            "drop": "Drop a dataset folder onto the window to begin.",
            "install_tools": "Images need COLMAP — install the reconstruction tools.",
            "reconstruct": "Images found. Reconstruct them into a COLMAP scene.",
            "train": "Dataset is ready. Start training.",
            "pause": "Training. Pause or stop it at any time.",
            "resume": f"Paused at iteration {self.stats.step:,}. Resume or stop.",
            "export": "Training complete. Export the .ply, or train again.",
            "restart": "Result exported. Train again, or load another dataset.",
        }.get(self.primary_action, "")


# --------------------------------------------------------------------------
# the session: the only mutable thing, and the only lock
# --------------------------------------------------------------------------

class Session:
    """Locked state shared by the render thread and the orchestrator.

    The render thread READS snapshots and POSTS commands; the orchestrator takes
    commands and publishes transitions. Neither touches the other's objects.
    """

    def __init__(self, mode: str = MODE_TRAIN, log_lines: int = 40,
                 setup: TrainingSetup | None = None,
                 clock=time.monotonic) -> None:
        self._lock = threading.RLock()
        self._log = LogBuffer()
        self._log_lines = log_lines
        self._clock = clock
        self._dropped: str | None = None
        self._command: str | None = None
        self._snap = Snapshot(mode=mode)
        # Editable parameters: user INPUT, not published state, hence its own store
        # (see trainer.frontend.settings).
        self.setup = setup if setup is not None else TrainingSetup()

    # ---- reading ----

    def snapshot(self) -> Snapshot:
        """The frame's consistent view, with expired notices dropped."""
        with self._lock:
            snap = self._snap
        keep = tuple(n for n in snap.notices
                     if n.sticky or (self._clock() - n.at) < NOTICE_TTL_S)
        return snap if keep == snap.notices else replace(snap, notices=keep)

    def view(self) -> View:
        return View(self)

    @property
    def mode(self) -> str:
        return self._snap.mode

    @property
    def matcher(self) -> str:
        with self._lock:
            return self._snap.matcher

    @property
    def recon_cuda(self) -> bool:
        with self._lock:
            return self._snap.recon_cuda

    @property
    def scene_dir(self) -> str:
        with self._lock:
            return self._snap.dataset.scene_dir

    @property
    def dataset_path(self) -> str:
        with self._lock:
            return self._snap.dataset.path

    @property
    def export_name(self) -> str:
        with self._lock:
            return self._snap.export_name

    # ---- render thread: post intent ----

    def drop(self, path: str) -> None:
        """A folder was dropped. Kept separate from :meth:`request` so the
        orchestrator can decide whether it needs confirmation first."""
        with self._lock:
            self._dropped = str(path)

    def request(self, command: str) -> None:
        with self._lock:
            self._command = str(command)

    def take_dropped(self) -> str | None:
        with self._lock:
            path, self._dropped = self._dropped, None
            return path

    def take_command(self) -> str | None:
        with self._lock:
            command, self._command = self._command, None
            return command

    # ---- render thread: settings the orchestrator reads later ----

    def set_matcher_index(self, index: int) -> None:
        with self._lock:
            self._snap = replace(self._snap,
                                 matcher=MATCHERS[int(index) % len(MATCHERS)])

    def set_recon_cuda(self, enabled: bool) -> None:
        with self._lock:
            self._snap = replace(self._snap, recon_cuda=bool(enabled))

    def set_export_name(self, name: str) -> None:
        with self._lock:
            self._snap = replace(self._snap, export_name=str(name).strip()
                                 or "output.ply")

    def toggle_config_detail(self, shown: bool) -> None:
        with self._lock:
            self._snap = replace(self._snap, show_all_config=bool(shown))

    def toggle_log(self, shown: bool) -> None:
        with self._lock:
            self._snap = replace(self._snap, show_log=bool(shown))

    def mark_config_edited(self) -> None:
        """A parameter changed: any existing result no longer matches what the
        next run would produce. Drives the "modified" hint on the result."""
        with self._lock:
            if self._snap.has_result and not self._snap.result.stale_config:
                self._snap = replace(self._snap, result=replace(
                    self._snap.result, stale_config=True))

    # ---- notices, confirmations ----

    def notify(self, text: str, kind: str = OK) -> None:
        """Post a transient acknowledgement (see :class:`Notice`)."""
        with self._lock:
            notice = Notice(text=str(text), kind=kind, at=self._clock())
            self._snap = replace(
                self._snap, notices=(*self._snap.notices, notice)[-MAX_NOTICES:])

    def clear_notices(self) -> None:
        with self._lock:
            self._snap = replace(self._snap, notices=())

    def ask(self, confirm: Confirm) -> None:
        with self._lock:
            self._snap = replace(self._snap, confirm=confirm)

    def take_confirm(self, *, accept: bool, alt: bool = False) -> str | None:
        """Resolve a pending question; returns the command to run (None to drop)."""
        with self._lock:
            pending, self._snap = self._snap.confirm, replace(self._snap,
                                                              confirm=None)
        if pending is None or not accept:
            return None
        return pending.alt_command if alt and pending.alt_command else pending.command

    @property
    def pending_confirm(self) -> Confirm | None:
        with self._lock:
            return self._snap.confirm

    # ---- activity (long-running work) ----

    def begin_activity(self, label: str, fraction: float | None = None) -> None:
        with self._lock:
            self._snap = replace(self._snap,
                                 activity=Activity(label=label, fraction=fraction))

    def set_progress(self, label: str = "", fraction: float | None = None,
                     active: bool = True, detail: str | None = None) -> None:
        """Update the current activity. Shaped to match the progress parsers, which
        emit ``(label, fraction)`` per output line.

        ``detail`` is optional so a per-item callback ("camera 12/194") can publish
        the bar and its caption in ONE locked update instead of a set_progress +
        set_detail pair; omitted, the existing detail is preserved.
        """
        with self._lock:
            if not active:
                self._snap = replace(self._snap, activity=Activity())
                return
            current = self._snap.activity
            self._snap = replace(self._snap, activity=Activity(
                label=label or current.label,
                fraction=current.fraction if fraction is None else fraction,
                detail=current.detail if detail is None else str(detail)))

    def set_detail(self, detail: str = "") -> None:
        """Refine the current activity ("evaluating camera 12/194") without
        touching its label or fraction."""
        with self._lock:
            self._snap = replace(self._snap,
                                 activity=replace(self._snap.activity,
                                                  detail=str(detail)))

    def end_activity(self) -> None:
        with self._lock:
            self._snap = replace(self._snap, activity=Activity())

    # ---- dataset lifecycle ----

    def dataset_inspecting(self, path: str) -> None:
        """A drop was accepted: acknowledge it before anything is read."""
        with self._lock:
            self._snap = replace(
                self._snap, dataset_state=DatasetState.INSPECTING,
                dataset=DatasetInfo(path=str(path)), dataset_error="",
                activity=Activity(label="Inspecting dataset"))

    def dataset_detected(self, kind: str, *, scene_dir: str = "") -> None:
        """Say WHAT was detected as soon as it is known, before loading it."""
        with self._lock:
            self._snap = replace(self._snap, dataset=replace(
                self._snap.dataset, kind=str(kind), scene_dir=str(scene_dir)))

    def dataset_loading(self, label: str = "Loading dataset") -> None:
        with self._lock:
            self._snap = replace(self._snap, dataset_state=DatasetState.LOADING,
                                 activity=Activity(label=label))

    def dataset_reconstructing(self) -> None:
        with self._lock:
            self._snap = replace(self._snap,
                                 dataset_state=DatasetState.RECONSTRUCTING,
                                 activity=Activity(label="Reconstructing",
                                                   fraction=0.0))

    def dataset_ready(self, info: DatasetInfo) -> None:
        with self._lock:
            self._snap = replace(self._snap, dataset_state=DatasetState.READY,
                                 dataset=info, dataset_error="",
                                 activity=Activity())

    def dataset_failed(self, reason: str) -> None:
        with self._lock:
            self._snap = replace(self._snap, dataset_state=DatasetState.ERROR,
                                 dataset_error=str(reason), activity=Activity())

    def update_dataset(self, **fields) -> None:
        """Merge into the current dataset record (image count, resolution, ...)."""
        with self._lock:
            self._snap = replace(self._snap,
                                 dataset=replace(self._snap.dataset, **fields))

    # ---- tools lifecycle ----

    def tools(self, state: ToolsState) -> None:
        with self._lock:
            activity = (Activity(label="Installing reconstruction tools", fraction=0.0)
                        if state == ToolsState.INSTALLING else Activity())
            self._snap = replace(self._snap, tools_state=state, activity=activity)

    # ---- training lifecycle ----

    def training_initializing(self, target_iterations: int = 0) -> None:
        """Acknowledge Start/Restart immediately: the scene load and pipeline build
        come BEFORE the first iteration and are the longest part of a short run."""
        with self._lock:
            stats = TrainingStats(total=int(target_iterations))
            self._snap = replace(self._snap,
                                 training_state=TrainingState.INITIALIZING,
                                 training_error="", stats=stats, loss_history=(),
                                 activity=Activity(label="Initializing run"))

    def training_running(self) -> None:
        with self._lock:
            self._snap = replace(self._snap, training_state=TrainingState.RUNNING,
                                 activity=Activity())

    def training_paused(self) -> None:
        with self._lock:
            self._snap = replace(self._snap, training_state=TrainingState.PAUSED,
                                 activity=Activity())

    def training_stopping(self) -> None:
        with self._lock:
            self._snap = replace(self._snap, training_state=TrainingState.STOPPING,
                                 activity=Activity(label="Stopping training"))

    def training_finalizing(self, label: str = "Finalizing run") -> None:
        with self._lock:
            self._snap = replace(self._snap,
                                 training_state=TrainingState.FINALIZING,
                                 activity=Activity(label=label, fraction=0.0))

    def training_completed(self, result: ResultInfo) -> None:
        """The run is over and its output is now the session's important object."""
        with self._lock:
            self._snap = replace(self._snap,
                                 training_state=TrainingState.COMPLETED,
                                 result_state=ResultState.TRAINED, result=result,
                                 result_error="", activity=Activity())

    def training_failed(self, reason: str) -> None:
        with self._lock:
            self._snap = replace(self._snap, training_state=TrainingState.FAILED,
                                 training_error=str(reason), activity=Activity())

    def publish_training(self, **fields) -> None:
        """Merge live numbers; fields not passed keep their previous value, so a
        throughput tick and a loss/psnr tick publish independently.

        A loss value is ALSO appended to :attr:`Snapshot.loss_history` for the
        graph. Recording it here rather than in a separate call means the graph is
        fed by whatever cadence the reporter already runs on and costs nothing
        extra — the loss readback is the expensive part and it is not ours.
        """
        with self._lock:
            stats = replace(self._snap.stats, **fields)
            history = self._snap.loss_history
            loss = fields.get("loss")
            if loss is not None:
                history = (*history, (int(stats.step), float(loss)))[-LOSS_HISTORY:]
            self._snap = replace(self._snap, stats=stats, loss_history=history)

    # ---- result lifecycle ----

    def result_initialized(self, splats: int = 0) -> None:
        """Init gaussians are on screen but nothing has been trained yet."""
        with self._lock:
            if self._snap.result_state in (ResultState.NONE,
                                           ResultState.INITIALIZED):
                self._snap = replace(self._snap,
                                     result_state=ResultState.INITIALIZED,
                                     result=ResultInfo(splats=int(splats)))

    def result_exported(self, path: str, size_mb: float) -> None:
        with self._lock:
            self._snap = replace(
                self._snap, result_state=ResultState.EXPORTED, result_error="",
                result=replace(self._snap.result, exported_path=str(path),
                               exported_mb=float(size_mb)),
                activity=Activity())

    def result_failed(self, reason: str) -> None:
        with self._lock:
            self._snap = replace(self._snap, result_error=str(reason),
                                 activity=Activity())

    def drop_result(self) -> None:
        """Forget the reconstruction (a restart is about to replace it)."""
        with self._lock:
            self._snap = replace(self._snap, result_state=ResultState.NONE,
                                 result=ResultInfo(), result_error="")

    # ---- whole-session reset ----

    def clear_session(self) -> None:
        """Back to the empty drop target.

        Keeps what is not this dataset's story: the installed tools, the matcher
        and build choices, the parameter values (the user's own), and the log.
        Pending intent is dropped so a click that arrived just before the reset
        cannot act on the session that was cleared.
        """
        with self._lock:
            self._dropped = None
            self._command = None
            self._snap = replace(
                self._snap, dataset_state=DatasetState.EMPTY,
                dataset=DatasetInfo(), dataset_error="",
                training_state=TrainingState.IDLE, training_error="",
                stats=TrainingStats(), loss_history=(),
                result_state=ResultState.NONE, result=ResultInfo(),
                result_error="", activity=Activity(), confirm=None)

    def reset_for_new_run(self) -> None:
        """Between runs on the SAME dataset: the previous run's numbers and result
        go, the dataset stays (it does not need re-reading)."""
        with self._lock:
            self._snap = replace(
                self._snap, training_state=TrainingState.IDLE, training_error="",
                stats=TrainingStats(), loss_history=(),
                result_state=ResultState.NONE, result=ResultInfo(),
                result_error="", confirm=None)

    # ---- log (secondary: diagnostics, not user feedback) ----

    def log(self, message: str) -> None:
        with self._lock:
            stored = self._log.append(message)
            if stored is not None:
                self._snap = replace(self._snap,
                                     log=self._log.tail(self._log_lines))

    def clear_log(self) -> None:
        with self._lock:
            self._log.clear()
            self._snap = replace(self._snap, log=())


class View:
    """Per-frame snapshot cache for the render thread.

    The loop calls :meth:`sample` once per frame; every binding then reads the SAME
    frozen snapshot, so one lock is taken per frame and no two widgets can show
    state from different instants. Attribute access forwards to the snapshot, so
    bindings read as ``lambda: view.can_export``.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.snap = session.snapshot()

    def sample(self) -> Snapshot:
        self.snap = self.session.snapshot()
        return self.snap

    def __getattr__(self, name: str):
        return getattr(self.snap, name)
