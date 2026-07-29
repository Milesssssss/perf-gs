"""Change-task and comprehension-task framework.

A *change task* hands an agent a prompt, then verifies the result against hidden
tests and hard-gate signals executed inside the sandbox. A *comprehension task*
asks a read-only question and grades the answer. Both are deliberately
data-driven so new task families are declared, not coded.

Verification runs the sandbox's *own* source by prepending ``<workspace>/src`` to
``PYTHONPATH`` — this sidesteps the repo's editable-install gotcha where
``import trainer`` would otherwise resolve to the installed copy instead of the
candidate under evaluation.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from esc.gates import GateInputs
from esc.model import AgentTelemetry


# --- Command execution ------------------------------------------------------

@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class CommandRunner(Protocol):
    def run(self, args: Sequence[str], cwd: Path,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None) -> CommandResult: ...


class LiveCommandRunner:
    """Runs real subprocesses, capturing output and wall time."""

    def run(self, args: Sequence[str], cwd: Path,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None) -> CommandResult:
        merged = dict(os.environ)
        if env:
            merged.update(env)
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                list(args), cwd=cwd, capture_output=True, text=True,
                env=merged, timeout=timeout, check=False,
            )
            return CommandResult(completed.returncode, completed.stdout,
                                 completed.stderr, time.perf_counter() - start)
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", "timeout", time.perf_counter() - start, timed_out=True)


@dataclass
class TaskEnvironment:
    """How to execute verification inside a sandbox."""

    runner: CommandRunner = field(default_factory=LiveCommandRunner)
    python: tuple[str, ...] = (sys.executable,)
    prepend_src_to_pythonpath: bool = True

    def _env(self, workspace: Path) -> dict[str, str]:
        if not self.prepend_src_to_pythonpath:
            return {}
        src = str((workspace / "src").resolve())
        existing = os.environ.get("PYTHONPATH", "")
        return {"PYTHONPATH": src + (os.pathsep + existing if existing else "")}

    def run_python(self, workspace: Path, module_args: Sequence[str],
                   timeout: float | None = None) -> CommandResult:
        return self.runner.run([*self.python, *module_args], cwd=workspace,
                              env=self._env(workspace), timeout=timeout)

    def write_hidden_test(self, workspace: Path, test: HiddenTest) -> Path:
        target = workspace / test.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(test.source, encoding="utf-8")
        return target


# --- Task specification -----------------------------------------------------

@dataclass(frozen=True)
class HiddenTest:
    """A test file injected into the sandbox at verification time (kept out of
    the repo so agents cannot train against it)."""

    filename: str
    source: str


@dataclass(frozen=True)
class Check:
    """One verification command classified by its gate role."""

    args: tuple[str, ...]
    kind: str = "hidden"  # hidden | existing | abi | static | parity
    name: str = ""
    expect_returncode: int = 0
    timeout_s: float = 600.0


@dataclass(frozen=True)
class TaskVerification:
    hidden_tests_pass: bool
    gate_inputs: GateInputs
    detail: str = ""


class ChangeTask:
    """Abstract change task. Subclasses expose a prompt and a verifier."""

    name: str
    family: str
    baseline: AgentTelemetry
    expected_change_files: tuple[str, ...]
    doc: str

    @property
    def prompt(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def verify(self, workspace: Path, env: TaskEnvironment) -> TaskVerification:  # pragma: no cover
        raise NotImplementedError


class CommandChangeTask(ChangeTask):
    """A data-driven change task: hidden tests + classified verification checks."""

    def __init__(
        self,
        name: str,
        family: str,
        prompt: str,
        baseline: AgentTelemetry,
        hidden_tests: Sequence[HiddenTest] = (),
        checks: Sequence[Check] = (),
        expected_change_files: Sequence[str] = (),
        doc: str = "",
    ) -> None:
        self.name = name
        self.family = family
        self._prompt = prompt
        self.baseline = baseline
        self.hidden_tests = tuple(hidden_tests)
        self.checks = tuple(checks)
        self.expected_change_files = tuple(expected_change_files)
        self.doc = doc

    @property
    def prompt(self) -> str:
        return self._prompt

    def verify(self, workspace: Path, env: TaskEnvironment) -> TaskVerification:
        pytest_checks = self._install_hidden_tests(workspace, env)
        outcomes = [(c, self._run_check(workspace, env, c))
                    for c in (*self.checks, *pytest_checks)]

        def passed(kind: str) -> list[bool]:
            return [res.returncode == c.expect_returncode and not res.timed_out
                    for c, res in outcomes if c.kind == kind]

        hidden = passed("hidden")
        existing = passed("existing")
        abi = passed("abi")
        static_failures = sum(1 for ok in passed("static") if not ok)
        parity = passed("parity")

        gate_inputs = GateInputs(
            existing_tests_pass=all(existing) if existing else True,
            max_numeric_deviation=0.0 if all(parity) else 1.0,
            perf_ratio=1.0,
            abi_stable=all(abi) if abi else True,
            deterministic=True,
            static_blockers=static_failures,
            mutation_score=0.0,
        )
        detail = "\n".join(
            f"[{c.kind}] {c.name or ' '.join(c.args)} -> rc={res.returncode}"
            for c, res in outcomes
        )
        return TaskVerification(
            hidden_tests_pass=all(hidden) if hidden else False,
            gate_inputs=gate_inputs,
            detail=detail,
        )

    def _install_hidden_tests(self, workspace: Path, env: TaskEnvironment) -> list[Check]:
        checks: list[Check] = []
        for test in self.hidden_tests:
            path = env.write_hidden_test(workspace, test)
            rel = path.relative_to(workspace).as_posix()
            checks.append(Check(
                args=("-m", "pytest", rel, "-q", "-p", "no:cacheprovider"),
                kind="hidden", name=f"hidden:{test.filename}",
            ))
        return checks

    def _run_check(self, workspace: Path, env: TaskEnvironment, check: Check) -> CommandResult:
        return env.run_python(workspace, check.args, timeout=check.timeout_s)


# --- Comprehension tasks ----------------------------------------------------

@dataclass(frozen=True)
class ComprehensionTask:
    """A read-only understandability probe with a gradable answer key."""

    name: str
    question: str
    expected_files: tuple[str, ...]
    answer_keywords: tuple[str, ...]
    doc: str = ""
