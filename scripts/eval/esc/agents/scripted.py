"""A deterministic, offline agent runner.

The scripted runner replays pre-programmed behaviour keyed by task name. Because
a scripted change can *actually apply real edits* to the sandbox, it drives the
full pipeline — sandbox, gates, hidden tests, scoring — deterministically and
without a live model. That makes the harness self-testable and gives operators a
zero-cost dry run before spending tokens on a real agent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from esc.agents.base import (
    ChangeTaskRequest,
    ChangeTaskResult,
    ComprehensionRequest,
    ComprehensionResult,
    RefactorRequest,
    RefactorResult,
)
from esc.model import AgentTelemetry, PatchStats

#: An ``apply`` callable mutates the sandbox for one (workspace, seed) and
#: returns the structural footprint of what it changed.
ApplyFn = Callable[[Path, int], PatchStats]


@dataclass(frozen=True)
class ScriptedChange:
    """Programmed behaviour for one change task."""

    telemetry: AgentTelemetry
    apply: ApplyFn | None = None
    patch_stats: PatchStats = field(default_factory=PatchStats)


@dataclass(frozen=True)
class ScriptedComprehension:
    """Programmed answer for one comprehension task."""

    answer: str
    cited_files: tuple[str, ...]
    confidence: float
    context_tokens: int


@dataclass(frozen=True)
class ScriptedRefactor:
    telemetry: AgentTelemetry
    apply: ApplyFn | None = None
    summary: str = ""


class ScriptedAgentRunner:
    """Replays scripted behaviour; satisfies :class:`~esc.agents.base.AgentRunner`."""

    def __init__(
        self,
        changes: Mapping[str, ScriptedChange] | None = None,
        comprehensions: Mapping[str, ScriptedComprehension] | None = None,
        refactor: ScriptedRefactor | None = None,
        name: str = "scripted",
    ) -> None:
        self.name = name
        self._changes = dict(changes or {})
        self._comprehensions = dict(comprehensions or {})
        self._refactor = refactor

    def run_change_task(self, request: ChangeTaskRequest) -> ChangeTaskResult:
        behavior = self._changes.get(request.task_name)
        if behavior is None:
            return ChangeTaskResult(AgentTelemetry(), PatchStats(), notes="no scripted change")
        stats = behavior.apply(request.workspace, request.seed) if behavior.apply else behavior.patch_stats
        return ChangeTaskResult(telemetry=behavior.telemetry, patch_stats=stats)

    def run_comprehension_task(self, request: ComprehensionRequest) -> ComprehensionResult:
        answer = self._comprehensions.get(request.task_name)
        if answer is None:
            return ComprehensionResult("", (), 0.0, 0)
        return ComprehensionResult(
            answer=answer.answer,
            cited_files=answer.cited_files,
            confidence=answer.confidence,
            context_tokens=answer.context_tokens,
        )

    def propose_refactor(self, request: RefactorRequest) -> RefactorResult:
        if self._refactor is None:
            return RefactorResult(AgentTelemetry(), changed=False, summary="no scripted refactor")
        if self._refactor.apply is not None:
            self._refactor.apply(request.workspace, request.seed)
        return RefactorResult(self._refactor.telemetry, changed=True, summary=self._refactor.summary)
