# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The application's state machine: independent dataset, training and result lifecycles the UI is a view of.

The orchestrator drives transitions through :class:`Session`; the render thread only reads snapshots.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from enum import StrEnum

from trainer.frontend.settings import TrainingSetup

LOG_CAPACITY = 400
# Success/info toast lifetime; errors and warnings never expire, so a failure can't scroll away unread.
NOTICE_TTL_S = 6.0
MAX_NOTICES = 3
# Loss samples kept for the graph: a window of --log-every ticks, not of iterations.
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


# Session modes: the hub owns the lifecycle from an empty window; a `--scene` run only observes.
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
# PAUSED is not busy: a paused session must stay free to export, inspect and configure.
_TRAINING_BUSY = (TrainingState.INITIALIZING, TrainingState.RUNNING,
                  TrainingState.STOPPING, TrainingState.FINALIZING)
_TRAINING_LIVE = (*_TRAINING_BUSY, TrainingState.PAUSED)


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
    # Where an images -> COLMAP reconstruction was written, i.e. the derived training scene.
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

    Deliberately has no quality metric: the only live PSNR is one sampled camera, not a test score.
    """

    iterations: int = 0
    target_iterations: int = 0
    wall_s: float = 0.0
    splats: int = 0
    run_dir: str = ""
    exported_path: str = ""
    exported_mb: float = 0.0
    # True once the configuration is edited after this result, so the next run would not reproduce it.
    stale_config: bool = False

    def summary(self) -> str:
        parts = [f"{self.iterations:,} iterations", format_duration(self.wall_s)]
        return " · ".join(parts)

    def quality(self) -> str:
        """The size of the model, which the panel can know exactly."""
        return f"{format_count(self.splats)} splats"


@dataclass(frozen=True)
class Activity:
    """One long-running operation; ``fraction`` is None when no honest number exists (indeterminate)."""

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
    """A pending question, rendered inline by its section; ``command`` runs if the user accepts."""

    key: str                      # which section renders it
    title: str
    body: str
    command: str
    confirm_label: str = "Continue"
    alt_label: str = ""           # e.g. "Export first"
    alt_command: str = ""


class LogBuffer:
    """Bounded, de-noised log tail: keeps the last ``\\r`` redraw segment, drops blanks, counts repeats."""

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
    # Panel disclosure as opt-in visibility: the toolkit cannot collapse a header programmatically.
    show_all_config: bool = False
    show_log: bool = False
    # (iteration, loss) samples for the graph, oldest first; bounded, since the run's jsonl is the record.
    loss_history: tuple[tuple[int, float], ...] = ()

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
        """The session in one line: the subject, not the operation (the badge and activity say that)."""
        if self.dataset_state in _DATASET_BUSY:
            return self.dataset.name or "working"
        if self.dataset_state == DatasetState.ERROR:
            return "dataset failed to load"
        if self.training_state == TrainingState.FAILED:
            return "training failed"
        if self.training_state == TrainingState.PAUSED:
            return f"paused at iteration {self.stats.step:,}"
        # A live run: the badge already says RUNNING, so name what is being trained.
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

    @property
    def hub(self) -> bool:
        return self.mode == MODE_HUB

    @property
    def can_drop(self) -> bool:
        """Drop stays live all session (it is how a user moves on), except while an operation owns the GPU."""
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
        return self.training_state in (TrainingState.RUNNING,
                                        TrainingState.FINALIZING)

    @property
    def can_resume(self) -> bool:
        return self.training_state == TrainingState.PAUSED

    @property
    def can_stop(self) -> bool:
        """Stopping keeps what has been optimized so far — it is not a cancel."""
        return self.training_state in (TrainingState.RUNNING, TrainingState.PAUSED,
                                        TrainingState.FINALIZING)

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
        """Locked only while a run is training, since parameters are read when a run is built."""
        return self.hub and self.training_state not in (
            TrainingState.INITIALIZING, TrainingState.RUNNING,
            TrainingState.STOPPING, TrainingState.FINALIZING)

    @property
    def primary_action(self) -> str:
        """The one action the panel should present most prominently."""
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
        """One sentence telling the user what to do next (empty while an activity runs)."""
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


class Session:
    """Locked state shared by the render thread and the orchestrator.

    Render thread: reads snapshots, posts commands. Orchestrator: takes commands, publishes transitions.
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
        self._paused_run: tuple[TrainingState, Activity] | None = None
        # Editable parameters are user input, not published state, hence their own store.
        self.setup = setup if setup is not None else TrainingSetup()

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

    # Render thread: post intent.

    def drop(self, path: str) -> None:
        """Post a dropped folder, separately from :meth:`request` so the orchestrator can confirm first."""
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

    # Render thread: settings the orchestrator reads later.

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
        """A parameter changed: mark any existing result stale (drives the result's "modified" hint)."""
        with self._lock:
            if self._snap.has_result and not self._snap.result.stale_config:
                self._snap = replace(self._snap, result=replace(
                    self._snap.result, stale_config=True))

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

    def begin_activity(self, label: str, fraction: float | None = None) -> None:
        with self._lock:
            self._snap = replace(self._snap,
                                 activity=Activity(label=label, fraction=fraction))

    def set_progress(self, label: str = "", fraction: float | None = None,
                     active: bool = True, detail: str | None = None) -> None:
        """Update the current activity, matching the progress parsers' ``(label, fraction)`` per line.

        ``detail`` lets a per-item callback publish bar and caption in one locked update; None keeps it.
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
        """Refine the current activity ("evaluating camera 12/194") without touching its label or fraction."""
        with self._lock:
            self._snap = replace(self._snap,
                                 activity=replace(self._snap.activity,
                                                  detail=str(detail)))

    def end_activity(self) -> None:
        with self._lock:
            self._snap = replace(self._snap, activity=Activity())

    def dataset_inspecting(self, path: str) -> None:
        """A drop was accepted: acknowledge it before anything is read."""
        with self._lock:
            self._snap = replace(
                self._snap, dataset_state=DatasetState.INSPECTING,
                dataset=DatasetInfo(path=str(path)), dataset_error="",
                activity=Activity(label="Inspecting dataset"))

    def dataset_detected(self, kind: str, *, scene_dir: str = "") -> None:
        """Say what was detected as soon as it is known, before loading it."""
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

    def tools(self, state: ToolsState) -> None:
        with self._lock:
            activity = (Activity(label="Installing reconstruction tools", fraction=0.0)
                        if state == ToolsState.INSTALLING else Activity())
            self._snap = replace(self._snap, tools_state=state, activity=activity)

    def training_initializing(self, target_iterations: int = 0) -> None:
        """Acknowledge Start/Restart at once: scene load and pipeline build precede the first iteration."""
        with self._lock:
            self._paused_run = None
            stats = TrainingStats(total=int(target_iterations))
            self._snap = replace(self._snap,
                                 training_state=TrainingState.INITIALIZING,
                                 training_error="", stats=stats, loss_history=(),
                                 activity=Activity(label="Initializing run"))

    def training_running(self) -> None:
        with self._lock:
            self._paused_run = None
            self._snap = replace(self._snap, training_state=TrainingState.RUNNING,
                                 activity=Activity())

    def training_paused(self) -> None:
        with self._lock:
            if not self._snap.can_pause:
                return
            self._paused_run = (self._snap.training_state, self._snap.activity)
            self._snap = replace(self._snap, training_state=TrainingState.PAUSED)

    def training_resumed(self) -> None:
        """Resume the operation that was paused, including finalization progress."""
        with self._lock:
            if self._snap.training_state != TrainingState.PAUSED:
                return
            assert self._paused_run is not None
            state, activity = self._paused_run
            self._paused_run = None
            self._snap = replace(self._snap, training_state=state, activity=activity)

    def training_stopping(self) -> bool:
        with self._lock:
            if not self._snap.can_stop:
                return False
            self._paused_run = None
            self._snap = replace(self._snap, training_state=TrainingState.STOPPING,
                                 activity=Activity(label="Stopping training"))
            return True

    def training_finalizing(self, label: str = "Finalizing run") -> None:
        with self._lock:
            if self._snap.training_state in (TrainingState.PAUSED,
                                             TrainingState.STOPPING):
                return
            self._snap = replace(self._snap,
                                 training_state=TrainingState.FINALIZING,
                                 activity=Activity(label=label, fraction=0.0))

    def training_completed(self, result: ResultInfo) -> None:
        """The run is over and its output is now the session's important object."""
        with self._lock:
            self._paused_run = None
            self._snap = replace(self._snap,
                                 training_state=TrainingState.COMPLETED,
                                 result_state=ResultState.TRAINED, result=result,
                                 result_error="", activity=Activity())

    def training_failed(self, reason: str) -> None:
        with self._lock:
            self._paused_run = None
            self._snap = replace(self._snap, training_state=TrainingState.FAILED,
                                 training_error=str(reason), activity=Activity())

    def publish_training(self, **fields) -> None:
        """Merge live numbers; fields not passed keep their previous value.

        A loss is also appended to :attr:`Snapshot.loss_history`, so the graph costs no extra readback.
        """
        with self._lock:
            stats = replace(self._snap.stats, **fields)
            history = self._snap.loss_history
            loss = fields.get("loss")
            if loss is not None:
                history = (*history, (int(stats.step), float(loss)))[-LOSS_HISTORY:]
            self._snap = replace(self._snap, stats=stats, loss_history=history)

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

    def clear_session(self) -> None:
        """Back to the empty drop target, keeping tools, matcher/build choices, parameters and the log.

        Pending intent is dropped so a click just before the reset cannot act on the cleared session.
        """
        with self._lock:
            self._dropped = None
            self._command = None
            self._paused_run = None
            self._snap = replace(
                self._snap, dataset_state=DatasetState.EMPTY,
                dataset=DatasetInfo(), dataset_error="",
                training_state=TrainingState.IDLE, training_error="",
                stats=TrainingStats(), loss_history=(),
                result_state=ResultState.NONE, result=ResultInfo(),
                result_error="", activity=Activity(), confirm=None)

    def reset_for_new_run(self) -> None:
        """Between runs on the same dataset: drop the previous run's numbers and result, keep the dataset."""
        with self._lock:
            self._paused_run = None
            self._snap = replace(
                self._snap, training_state=TrainingState.IDLE, training_error="",
                stats=TrainingStats(), loss_history=(),
                result_state=ResultState.NONE, result=ResultInfo(),
                result_error="", confirm=None)

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
    """Per-frame snapshot cache for the render thread; attribute access forwards to the snapshot.

    Call :meth:`sample` once per frame so every binding reads the same frozen state (one lock per frame).
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.snap = session.snapshot()

    def sample(self) -> Snapshot:
        self.snap = self.session.snapshot()
        return self.snap

    def __getattr__(self, name: str):
        return getattr(self.snap, name)
