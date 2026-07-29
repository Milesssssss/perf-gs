"""The coding-agent boundary.

Everything the harness needs from an agent is expressed here as small request/
result value objects and a single :class:`AgentRunner` protocol. Two
implementations are provided:

    scripted    deterministic, applies known edits — drives the tests and
                dry-runs the whole pipeline without a live model
    subprocess  invokes a configured agent CLI inside the sandbox and reads the
                resulting git diff — the seam for a real coding agent

Keeping this boundary narrow is what lets the evaluator, tasks, gates and loop
stay agnostic to *which* agent (or human) makes the change.
"""

from __future__ import annotations

from esc.agents.base import (
    AgentRunner,
    ChangeTaskRequest,
    ChangeTaskResult,
    ComprehensionRequest,
    ComprehensionResult,
    RefactorRequest,
    RefactorResult,
)
from esc.agents.scripted import (
    ScriptedAgentRunner,
    ScriptedChange,
    ScriptedComprehension,
)
from esc.agents.subprocess_runner import SubprocessAgentRunner, SubprocessConfig

__all__ = [
    "AgentRunner",
    "ChangeTaskRequest",
    "ChangeTaskResult",
    "ComprehensionRequest",
    "ComprehensionResult",
    "RefactorRequest",
    "RefactorResult",
    "ScriptedAgentRunner",
    "ScriptedChange",
    "ScriptedComprehension",
    "SubprocessAgentRunner",
    "SubprocessConfig",
]
