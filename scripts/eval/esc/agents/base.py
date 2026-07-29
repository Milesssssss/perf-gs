"""Agent request/result value objects and the :class:`AgentRunner` protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from esc.model import AgentTelemetry, PatchStats


# --- Change tasks -----------------------------------------------------------

@dataclass(frozen=True)
class ChangeTaskRequest:
    """Ask an agent to make a change inside ``workspace``."""

    task_name: str
    prompt: str
    workspace: Path
    seed: int


@dataclass(frozen=True)
class ChangeTaskResult:
    """What the agent did. ``patch_stats`` may be None, in which case the harness
    derives the structural footprint from the workspace's git diff."""

    telemetry: AgentTelemetry
    patch_stats: PatchStats | None = None
    notes: str = ""


# --- Comprehension tasks ----------------------------------------------------

@dataclass(frozen=True)
class ComprehensionRequest:
    """Ask an agent a read-only question about ``workspace``."""

    task_name: str
    question: str
    workspace: Path
    seed: int


@dataclass(frozen=True)
class ComprehensionResult:
    """The agent's answer plus the files it cited and its self-confidence."""

    answer: str
    cited_files: tuple[str, ...]
    confidence: float
    context_tokens: int


# --- Refactor proposals (loop) ----------------------------------------------

@dataclass(frozen=True)
class RefactorRequest:
    """Ask an agent to improve a hotspot without changing behaviour."""

    workspace: Path
    hotspot: str
    seed: int
    instructions: str = ""


@dataclass(frozen=True)
class RefactorResult:
    """Outcome of a refactor proposal applied in ``workspace``."""

    telemetry: AgentTelemetry
    changed: bool
    summary: str = ""


# --- The boundary -----------------------------------------------------------

@runtime_checkable
class AgentRunner(Protocol):
    """The single interface every evaluated agent (or human proxy) implements."""

    name: str

    def run_change_task(self, request: ChangeTaskRequest) -> ChangeTaskResult: ...

    def run_comprehension_task(self, request: ComprehensionRequest) -> ComprehensionResult: ...

    def propose_refactor(self, request: RefactorRequest) -> RefactorResult: ...
