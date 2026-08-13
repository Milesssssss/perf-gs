"""Shared session state between the viewer (UI thread) and the orchestrator.

The viewer PRODUCES commands (a dataset was dropped, the matcher was chosen,
setup/reconstruct/train/export/clear were requested) and CONSUMES display state
(dataset kind, tools-ready, phase, progress, training stats, log tail). The
orchestrator does the reverse. Neither thread ever touches the other's widgets: everything crosses
through one lock here.

This is also where UI POLICY lives, so it exists exactly once instead of being
re-derived in the render loop: :meth:`SessionState.snapshot` returns an immutable
:class:`SessionSnapshot` that already carries ``can_setup`` / ``can_reconstruct``
/ ``can_train`` / ``can_export`` / ``can_clear``, the current
:class:`Progress` (phase-aware — training progress
comes from the iteration counter, reconstruction from its stage parser) and a
wrapped-ready log tail. The panel binds straight to those fields.

The predecessor of this module kept progress as a SINGLE string, which is why UI
messages "broke down": a 200-character runner command line overwrote the status
and was then clipped at the panel edge. Lines now land in a bounded
:class:`LogBuffer` that the panel renders as a wrapped, multi-line tail.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field, replace

from trainer.frontend.settings import TrainingSetup

# Orchestrator phases (also drive the panel's busy/enabled logic).
PHASE_IDLE = "idle"
PHASE_SETUP = "setup"
PHASE_RECONSTRUCT = "reconstruct"
PHASE_TRAINING = "training"
# The last iteration is NOT the end of the run: the final eval sweep, the metrics
# pass and the .ply export render every camera, which is minutes of work the
# orchestrator spends outside the loop. Without a phase of its own that time read
# as a finished run parked at "iter N/N" while no panel command could be served.
PHASE_FINALIZE = "finalize"
PHASE_DONE = "done"
PHASE_ERROR = "error"

_BUSY = (PHASE_SETUP, PHASE_RECONSTRUCT, PHASE_TRAINING, PHASE_FINALIZE)

# Session modes. The hub picks a dataset and drives reconstruct/train from the
# panel; a plain training run already has its scene from the CLI. Only the hints
# and the DATASET section differ.
MODE_TRAIN = "train"
MODE_HUB = "hub"

# Human-readable phase captions for the panel's status line.
PHASE_LABELS = {
    PHASE_IDLE: "idle",
    PHASE_SETUP: "installing recon tools",
    PHASE_RECONSTRUCT: "reconstructing",
    PHASE_TRAINING: "training",
    PHASE_FINALIZE: "finalizing the run",
    PHASE_DONE: "done",
    PHASE_ERROR: "error",
}

# COLMAP matcher choices offered by the panel (order == combo order).
MATCHERS = ("exhaustive", "sequential", "vocab_tree")

# Dataset classifications produced by reconstruction.classify_dataset.
KIND_NONE = "none"
KIND_COLMAP = "colmap"
KIND_IMAGES = "images"

LOG_CAPACITY = 400


@dataclass(frozen=True)
class Progress:
    """A determinate progress readout: ``fraction`` in [0, 1] plus a caption."""

    label: str = ""
    fraction: float = 0.0
    active: bool = False

    def percent(self) -> int:
        return int(round(100.0 * min(1.0, max(0.0, self.fraction))))


@dataclass(frozen=True)
class TrainingStats:
    """Live training numbers, published by the trainer's reporter."""

    step: int = 0
    total: int = 0
    gaussians: int = 0
    it_per_s: float = 0.0
    loss: float | None = None
    psnr: float | None = None
    elapsed_s: float = 0.0

    @property
    def fraction(self) -> float:
        return 0.0 if self.total <= 0 else min(1.0, self.step / self.total)

    @property
    def eta_s(self) -> float | None:
        """Seconds left at the current rate, or None when not yet meaningful."""
        if self.total <= 0 or self.it_per_s <= 1e-6 or self.step >= self.total:
            return None
        return (self.total - self.step) / self.it_per_s


def format_duration(seconds: float | None) -> str:
    """``h:mm:ss`` / ``m:ss`` for the panel (never a bare float of seconds)."""
    if seconds is None:
        return "--"
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class LogBuffer:
    """Bounded, de-noised log tail.

    Subprocess output (pip, COLMAP) arrives as raw lines, sometimes carrying
    carriage-return progress redraws and long runs of identical lines. Only the
    last segment of a ``\\r`` redraw is kept, blank lines are dropped, and an
    immediately repeated line is counted instead of repeated.
    """

    def __init__(self, capacity: int = LOG_CAPACITY) -> None:
        self._lines: deque[str] = deque(maxlen=capacity)
        self._repeats = 0

    def append(self, message: str) -> str | None:
        """Add a line; returns the stored text (None when it was dropped)."""
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
        if count <= 0:
            return ()
        lines = list(self._lines)
        return tuple(lines[-count:])

    def last(self) -> str:
        return self._lines[-1] if self._lines else ""

    def clear(self) -> None:
        self._lines.clear()
        self._repeats = 0

    def __len__(self) -> int:
        return len(self._lines)


@dataclass(frozen=True)
class SessionSnapshot:
    """Immutable view of the session for one UI frame."""

    mode: str = MODE_TRAIN
    phase: str = PHASE_IDLE
    dataset_path: str = ""
    dataset_kind: str = KIND_NONE
    scene_dir: str = ""
    tools_ready: bool = False
    # A finished run whose gaussians are still in memory — what makes Export
    # possible, and what tells the panel a further Train is a RE-run. Cleared when
    # the next run starts (its pipeline replaces this one's) and by clear_session.
    has_trained_model: bool = False
    matcher: str = MATCHERS[0]
    # INSTALL-TIME choice of build for the COLMAP/GLOMAP binaries, read by the
    # orchestrator when the panel asks for setup (CUDA by default — it is what puts
    # feature matching and bundle adjustment on the GPU at all).
    recon_cuda: bool = True
    # What the current phase is doing right now, when its own progress number
    # cannot say it (see :meth:`SessionState.set_activity`).
    activity: str = ""
    progress: Progress = field(default_factory=Progress)
    stats: TrainingStats = field(default_factory=TrainingStats)
    log: tuple[str, ...] = ()

    # ---- derived UI policy (computed once, in SessionState.snapshot) ----
    @property
    def busy(self) -> bool:
        return self.phase in _BUSY

    @property
    def phase_label(self) -> str:
        return PHASE_LABELS.get(self.phase, self.phase)

    @property
    def matcher_index(self) -> int:
        return MATCHERS.index(self.matcher) if self.matcher in MATCHERS else 0

    @property
    def has_dataset(self) -> bool:
        return bool(self.dataset_path)

    @property
    def can_setup(self) -> bool:
        """Setup is a ONE-SHOT: once the binaries are there nothing is left for the
        button to do, so it greys out rather than inviting a repeat of a
        multi-hundred-MB download. ``tools_ready`` is probed from disk when the hub
        starts (:meth:`ReconTools.is_ready`), so it holds across sessions and not
        just after a setup run in THIS one.
        """
        return not self.busy and not self.tools_ready

    @property
    def can_reconstruct(self) -> bool:
        return (self.dataset_kind == KIND_IMAGES and self.tools_ready
                and not self.busy)

    @property
    def can_train(self) -> bool:
        """Also true after a run: pressing Train again re-launches on the same
        scene with whatever the TRAINING SETUP section now holds."""
        return bool(self.scene_dir) and not self.busy

    @property
    def can_export(self) -> bool:
        """Whether there is a trained model to write out as a .ply."""
        return self.has_trained_model and not self.busy

    @property
    def can_clear(self) -> bool:
        """Whether there is anything to forget (dataset and/or finished run)."""
        return not self.busy and (self.has_dataset or self.has_trained_model)

    @property
    def can_configure(self) -> bool:
        """Whether the TRAINING SETUP section is still editable.

        A run reads these values when it is BUILT, so the only time an edit would
        be silently ignored is while a run is actually training — that is the only
        phase that locks. Installing the recon tools and reconstructing a dataset
        both happen BEFORE any run exists, so there is nothing to protect: the
        panel stays editable through them (they take minutes, which is exactly when
        you want to set the run up). Finalizing does not lock either — the values
        it would clobber belong to the run that just ended, and the next one reads
        them fresh, which is how Train again re-launches with new parameters.
        """
        return self.mode == MODE_HUB and self.phase != PHASE_TRAINING

    @property
    def message(self) -> str:
        return self.log[-1] if self.log else ""

    def hint(self) -> str:
        """One line telling the user what to do NEXT — the panel's headline."""
        if self.phase == PHASE_ERROR:
            return "something failed — see the log below"
        if self.phase == PHASE_TRAINING:
            return "training"
        if self.phase == PHASE_SETUP:
            return "installing reconstruction tools..."
        if self.phase == PHASE_RECONSTRUCT:
            return "reconstructing — this takes a while"
        if self.phase == PHASE_FINALIZE:
            return "run finished — writing out its metrics and .ply"
        if self.phase == PHASE_DONE:
            if self.mode == MODE_HUB:
                return ("training finished — export the .ply, press Train again, "
                        "or drop another dataset")
            return "training finished — the viewer stays live"
        if self.mode == MODE_TRAIN:
            return "inspecting the initialized gaussians"
        if not self.has_dataset:
            return "drag a dataset folder onto the window"
        if self.dataset_kind == KIND_COLMAP or self.scene_dir:
            return "COLMAP dataset ready — press Train"
        if self.dataset_kind == KIND_IMAGES:
            return ("images found — press Reconstruct" if self.tools_ready
                    else "images found — press Set up tools first")
        return "unrecognized folder — need images/ or sparse/"


class SessionState:
    """Locked mailbox + display state shared by the viewer and orchestrator."""

    def __init__(self, mode: str = MODE_TRAIN, log_lines: int = 40,
                 setup: TrainingSetup | None = None) -> None:
        self._lock = threading.Lock()
        self._log = LogBuffer()
        self._log_lines = log_lines
        # viewer -> orchestrator (one-shot commands)
        self._dropped: str | None = None
        # "setup" | "reconstruct" | "train" | "export" | "clear"
        self._command: str | None = None
        # shared display state
        self._snapshot = SessionSnapshot(mode=mode)
        # Editable training hyperparameters (hub only in practice). Its own lock —
        # these are user INPUTS, not published state, so they are not part of the
        # per-frame snapshot (see trainer.frontend.settings).
        self.setup = setup if setup is not None else TrainingSetup()

    @property
    def mode(self) -> str:
        return self._snapshot.mode

    # ---- viewer thread: produce commands ----

    def drop(self, path: str) -> None:
        with self._lock:
            self._dropped = str(path)

    def request(self, command: str) -> None:
        with self._lock:
            self._command = str(command)

    def set_matcher(self, matcher: str) -> None:
        with self._lock:
            if matcher in MATCHERS:
                self._snapshot = replace(self._snapshot, matcher=matcher)

    def set_matcher_index(self, index: int) -> None:
        self.set_matcher(MATCHERS[int(index) % len(MATCHERS)])

    def set_recon_cuda(self, enabled: bool) -> None:
        """Pick the builds the next 'Set up recon tools' will install (CUDA or CPU
        COLMAP + GLOMAP)."""
        with self._lock:
            self._snapshot = replace(self._snapshot, recon_cuda=bool(enabled))

    # ---- orchestrator thread: consume commands ----

    def take_dropped(self) -> str | None:
        with self._lock:
            path, self._dropped = self._dropped, None
            return path

    def take_command(self) -> str | None:
        with self._lock:
            command, self._command = self._command, None
            return command

    @property
    def matcher(self) -> str:
        with self._lock:
            return self._snapshot.matcher

    @property
    def recon_cuda(self) -> bool:
        with self._lock:
            return self._snapshot.recon_cuda

    @property
    def scene_dir(self) -> str:
        with self._lock:
            return self._snapshot.scene_dir

    # ---- orchestrator thread: publish display state ----

    def set_dataset(self, path: str, kind: str) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, dataset_path=str(path),
                                     dataset_kind=str(kind))

    def set_scene_dir(self, path: str) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, scene_dir=str(path))

    def set_tools_ready(self, ready: bool) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, tools_ready=bool(ready))

    def set_trained_model(self, present: bool) -> None:
        """Publish whether a finished run's gaussians are still in memory (see
        :attr:`SessionSnapshot.has_trained_model`)."""
        with self._lock:
            self._snapshot = replace(self._snapshot,
                                     has_trained_model=bool(present))

    def clear_session(self) -> None:
        """Forget the dataset and the finished run — back to a fresh hub.

        What is NOT this dataset's story stays: the installed tools, the matcher
        and build choices, the TRAINING SETUP values (the user's parameters are
        theirs to keep) and the log, which is the record of what just happened.
        Pending commands are dropped too, so a click that arrived just before the
        reset cannot act on the session that was cleared.
        """
        with self._lock:
            self._dropped = None
            self._command = None
            self._snapshot = replace(
                self._snapshot, phase=PHASE_IDLE, dataset_path="",
                dataset_kind=KIND_NONE, scene_dir="", has_trained_model=False,
                progress=Progress(), stats=TrainingStats())

    def set_phase(self, phase: str) -> None:
        """Set the phase; leaving a busy phase also retires its progress bar.

        Changing phase always clears the activity suffix: it describes work of the
        phase being left, so carrying it over would caption the new one with it.
        """
        with self._lock:
            progress = self._snapshot.progress
            if phase not in _BUSY:
                progress = Progress()
            self._snapshot = replace(self._snapshot, phase=str(phase),
                                     progress=progress, activity="")

    def set_activity(self, activity: str = "") -> None:
        """Say what the current phase is doing RIGHT NOW, when its own progress
        number cannot say it — the training bar counts iterations, so the final
        eval sweep would otherwise sit at ``iter N/N`` with no explanation. Empty
        clears it."""
        with self._lock:
            self._snapshot = replace(self._snapshot, activity=str(activity))

    def set_progress(self, label: str = "", fraction: float = 0.0,
                     active: bool = True) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, progress=Progress(
                label=str(label), fraction=float(fraction), active=bool(active)))

    def publish_training(self, **fields) -> None:
        """Merge live training numbers (see :class:`TrainingStats`). Fields not
        passed keep their previous value, so a throughput tick and a loss/psnr
        tick can publish independently."""
        with self._lock:
            self._snapshot = replace(
                self._snapshot, stats=replace(self._snapshot.stats, **fields))

    def log(self, message: str) -> None:
        """Append one log line (safe to pass raw subprocess output)."""
        with self._lock:
            stored = self._log.append(message)
            if stored is not None:
                self._snapshot = replace(
                    self._snapshot, log=self._log.tail(self._log_lines))

    def clear_log(self) -> None:
        with self._lock:
            self._log.clear()
            self._snapshot = replace(self._snapshot, log=())

    # ---- viewer thread: read ----

    def snapshot(self) -> SessionSnapshot:
        """Consistent view of the whole session, with training progress folded
        into :attr:`SessionSnapshot.progress`."""
        with self._lock:
            snap = self._snapshot
        if snap.phase == PHASE_TRAINING and snap.stats.total > 0:
            stats = snap.stats
            label = f"iter {stats.step}/{stats.total}"
            snap = replace(snap, progress=Progress(
                label=f"{label} — {snap.activity}" if snap.activity else label,
                fraction=stats.fraction, active=True))
        return snap

    def view(self) -> "StateView":
        return StateView(self)


class StateView:
    """Per-frame snapshot cache for the UI thread.

    The render loop calls :meth:`sample` once per frame; every binding then reads
    the SAME frozen snapshot (one lock acquisition per frame, and no widget can
    show state from a different instant than its neighbour). Attribute access is
    forwarded, so bindings read as ``lambda: view.can_train``.
    """

    def __init__(self, state: SessionState) -> None:
        self.state = state
        self.snap = state.snapshot()

    def sample(self) -> SessionSnapshot:
        self.snap = self.state.snapshot()
        return self.snap

    def __getattr__(self, name: str):
        return getattr(self.snap, name)
