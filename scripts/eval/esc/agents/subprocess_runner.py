# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""A subprocess-backed agent runner — the seam for a real coding agent CLI.

The runner invokes a configured command inside the sandbox, then reads the
resulting ``git diff`` to measure the structural footprint. The agent may drop a
small JSON telemetry/answer file in the workspace to report tokens, retries and
cited files; when it does not, the runner synthesises what it can from wall time
and the diff.

This is intentionally a template: the exact command belongs to whichever agent
you are evaluating. Wire it via :class:`SubprocessConfig.command`, whose tokens
are ``str.format``-expanded with ``workspace``, ``prompt_file``, ``seed`` and
``task``.
"""

from __future__ import annotations

import json
import subprocess
import time
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


@dataclass(frozen=True)
class SubprocessConfig:
    """How to invoke and read back a real agent."""

    command: tuple[str, ...]
    timeout_s: float = 1800.0
    telemetry_filename: str = ".esc_telemetry.json"
    answer_filename: str = ".esc_answer.json"
    prompt_env: str = "ESC_PROMPT"
    seed_env: str = "ESC_SEED"
    name: str = "subprocess"


@dataclass
class _Run:
    returncode: int
    wall_time_s: float
    stdout: str
    stderr: str
    timed_out: bool = False


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=workspace, capture_output=True, text=True, check=False
    )
    return result.stdout


def git_patch_stats(workspace: Path) -> PatchStats:
    """Derive a structural footprint from the sandbox's working-tree diff."""
    numstat = _git(workspace, "diff", "--numstat")
    porcelain = _git(workspace, "status", "--porcelain")

    files: set[str] = set()
    lines = 0
    for row in numstat.splitlines():
        parts = row.split("\t")
        if len(parts) != 3:
            continue
        added, deleted, path = parts
        files.add(path)
        lines += _to_int(added) + _to_int(deleted)

    for row in porcelain.splitlines():
        status_path = row[3:].strip()
        if status_path.startswith("??"):
            continue
        if status_path:
            files.add(status_path)

    modules = {str(Path(f).parent) for f in files}
    return PatchStats(
        files_changed=len(files),
        modules_changed=len(modules),
        lines_changed=lines,
    )


def _to_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


class SubprocessAgentRunner:
    """Runs an external agent command; satisfies ``AgentRunner``."""

    def __init__(self, config: SubprocessConfig) -> None:
        self.config = config
        self.name = config.name

    # -- public interface ----------------------------------------------------

    def run_change_task(self, request: ChangeTaskRequest) -> ChangeTaskResult:
        run = self._invoke(request.prompt, request.workspace, request.seed, request.task_name)
        telemetry = self._read_telemetry(request.workspace, run)
        stats = git_patch_stats(request.workspace)
        telemetry = _with_patch_lines(telemetry, stats.lines_changed)
        return ChangeTaskResult(telemetry=telemetry, patch_stats=stats, notes=run.stderr[-500:])

    def run_comprehension_task(self, request: ComprehensionRequest) -> ComprehensionResult:
        run = self._invoke(request.question, request.workspace, request.seed, request.task_name)
        return self._read_answer(request.workspace, run)

    def propose_refactor(self, request: RefactorRequest) -> RefactorResult:
        prompt = request.instructions or f"Improve the maintainability of {request.hotspot}."
        run = self._invoke(prompt, request.workspace, request.seed, request.hotspot)
        telemetry = self._read_telemetry(request.workspace, run)
        changed = bool(_git(request.workspace, "status", "--porcelain").strip())
        return RefactorResult(telemetry=telemetry, changed=changed, summary=run.stdout[-500:])

    # -- internals -----------------------------------------------------------

    def _invoke(self, prompt: str, workspace: Path, seed: int, task: str) -> _Run:
        prompt_file = workspace / ".esc_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        command = [
            token.format(workspace=str(workspace), prompt_file=str(prompt_file),
                         seed=seed, task=task)
            for token in self.config.command
        ]
        env = {self.config.prompt_env: prompt, self.config.seed_env: str(seed)}
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                command, cwd=workspace, capture_output=True, text=True,
                timeout=self.config.timeout_s, check=False, env=_merged_env(env),
            )
            return _Run(completed.returncode, time.perf_counter() - start,
                        completed.stdout, completed.stderr)
        except subprocess.TimeoutExpired:
            return _Run(124, self.config.timeout_s, "", "timeout", timed_out=True)
        finally:
            prompt_file.unlink(missing_ok=True)

    def _read_telemetry(self, workspace: Path, run: _Run) -> AgentTelemetry:
        data = _read_json(workspace / self.config.telemetry_filename)
        return AgentTelemetry(
            tokens=int(data.get("tokens", 0)),
            wall_time_s=float(data.get("wall_time_s", run.wall_time_s)),
            retries=int(data.get("retries", 1 if run.timed_out else 0)),
            tool_calls=int(data.get("tool_calls", 0)),
            patch_lines=int(data.get("patch_lines", 0)),
            files_inspected=int(data.get("files_inspected", 0)),
            failed_test_cycles=int(data.get("failed_test_cycles", 0)),
        )

    def _read_answer(self, workspace: Path, run: _Run) -> ComprehensionResult:
        data = _read_json(workspace / self.config.answer_filename)
        if not data:
            return ComprehensionResult(run.stdout.strip(), (), 0.0, 0)
        return ComprehensionResult(
            answer=str(data.get("answer", "")),
            cited_files=tuple(data.get("cited_files", ())),
            confidence=float(data.get("confidence", 0.0)),
            context_tokens=int(data.get("context_tokens", 0)),
        )


def _with_patch_lines(telemetry: AgentTelemetry, lines: int) -> AgentTelemetry:
    if telemetry.patch_lines:
        return telemetry
    return AgentTelemetry(
        tokens=telemetry.tokens, wall_time_s=telemetry.wall_time_s,
        retries=telemetry.retries, tool_calls=telemetry.tool_calls,
        patch_lines=lines, files_inspected=telemetry.files_inspected,
        failed_test_cycles=telemetry.failed_test_cycles,
    )


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _merged_env(extra: dict[str, str]) -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.update(extra)
    return env
