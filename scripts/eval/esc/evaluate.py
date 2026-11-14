# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Paired evaluation: turn a codebase + agent into an :class:`EscResult`.

Unmeasured dimensions (e.g. no mutation evidence) are dropped and renormalised, not counted as zero.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path

from esc.agents.base import (
    AgentRunner,
    ChangeTaskRequest,
    ComprehensionRequest,
)
from esc.agents.subprocess_runner import git_patch_stats
from esc.config import EscConfig
from esc.gates import evaluate_gates
from esc.model import (
    ComprehensionRunResult,
    EscResult,
    MutationResult,
    PatchStats,
    SubScores,
    TaskRunResult,
)
from esc.scoring.aggregate import compute_esc_partial
from esc.scoring.correctness import score_correctness
from esc.scoring.edit_efficiency import score_edit_efficiency
from esc.scoring.locality import score_locality
from esc.scoring.static_quality import score_static_quality
from esc.scoring.testability import score_testability
from esc.scoring.understandability import score_understandability
from esc.tasks.base import ChangeTask, ComprehensionTask, TaskEnvironment
from esc.tasks.grading import grade_comprehension
from esc.tasks.registry import (
    TaskFamily,
    all_change_tasks,
    all_comprehension_tasks,
)
from esc.workspace import create_sandbox


def analyze_static(root: Path, config: EscConfig) -> dict:
    """Compute the agent-free dimensions: ``S`` and the static part of ``L``."""
    s_value, s_diag = score_static_quality(root, config)
    l_value, l_diag = score_locality(root, config, ())
    return {
        "static_quality": s_value,
        "static_quality_diagnostics": s_diag,
        "locality_static": l_value,
        "locality_diagnostics": l_diag,
    }


def _git_changed_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True,
        text=True, check=False,
    )
    files: list[str] = []
    for row in result.stdout.splitlines():
        path = row[3:].strip().replace("\\", "/")
        if path and not path.startswith("??"):
            files.append(path)
    return files


def _derive_patch_stats(root: Path, task: ChangeTask) -> PatchStats:
    """Derive a structural footprint from git when the agent did not report one."""
    base = git_patch_stats(root)
    changed = _git_changed_files(root)
    expected_components = {f.split("/")[0] for f in task.expected_change_files}
    if expected_components:
        unrelated = sum(1 for f in changed if f.split("/")[0] not in expected_components)
        used_extension = any(
            f in task.expected_change_files or "contract_schema" in f for f in changed
        )
    else:
        unrelated = 0
        used_extension = True
    return PatchStats(
        files_changed=base.files_changed,
        modules_changed=base.modules_changed,
        lines_changed=base.lines_changed,
        unrelated_components_touched=unrelated,
        used_designated_extension_point=used_extension,
    )


def run_change_task(
    task: ChangeTask, seed: int, root: Path, agent: AgentRunner,
    env: TaskEnvironment, config: EscConfig,
) -> TaskRunResult:
    """Run one (task, seed) in a fresh sandbox and score its outcome."""
    with create_sandbox(root, config) as sandbox:
        request = ChangeTaskRequest(task.name, task.prompt, sandbox.root, seed)
        outcome = agent.run_change_task(request)
        stats = outcome.patch_stats or _derive_patch_stats(sandbox.root, task)
        verification = task.verify(sandbox.root, env)
        gate_report = evaluate_gates(verification.gate_inputs, config.gates)
        success = gate_report.passed and verification.hidden_tests_pass
        return TaskRunResult(
            task_name=task.name,
            seed=seed,
            success=success,
            gate_report=gate_report,
            telemetry=outcome.telemetry,
            patch_stats=stats,
            detail=verification.detail,
        )


def run_comprehension_task(
    task: ComprehensionTask, seed: int, workspace: Path, agent: AgentRunner,
) -> ComprehensionRunResult:
    request = ComprehensionRequest(task.name, task.question, workspace, seed)
    result = agent.run_comprehension_task(request)
    return grade_comprehension(task, result, seed)


def collect_runs(
    root: Path,
    families: Sequence[TaskFamily],
    seeds: Sequence[int],
    agent: AgentRunner,
    config: EscConfig,
    env: TaskEnvironment | None = None,
    execute_change_tasks: bool = True,
) -> tuple[list[TaskRunResult], list[ComprehensionRunResult]]:
    """Run the agent across all tasks x seeds and return the raw records."""
    env = env or TaskEnvironment()
    families = tuple(families)
    change_tasks = all_change_tasks(families)
    comprehension_tasks = all_comprehension_tasks(families)

    change_runs: list[TaskRunResult] = []
    comprehension_runs: list[ComprehensionRunResult] = []
    if not execute_change_tasks:
        return change_runs, comprehension_runs

    for task in change_tasks:
        for seed in seeds:
            change_runs.append(run_change_task(task, seed, root, agent, env, config))
    if comprehension_tasks:
        for seed in seeds:
            with create_sandbox(root, config) as sandbox:
                for task in comprehension_tasks:
                    comprehension_runs.append(
                        run_comprehension_task(task, seed, sandbox.root, agent)
                    )
    return change_runs, comprehension_runs


def evaluate_codebase_detailed(
    root: Path,
    families: Sequence[TaskFamily],
    seeds: Sequence[int],
    agent: AgentRunner,
    config: EscConfig,
    env: TaskEnvironment | None = None,
    mutation: MutationResult | None = None,
    execute_change_tasks: bool = True,
) -> tuple[EscResult, list[TaskRunResult], list[ComprehensionRunResult]]:
    """Full evaluation that also returns the raw run records (for the loop)."""
    families = tuple(families)
    change_runs, comprehension_runs = collect_runs(
        root, families, seeds, agent, config, env, execute_change_tasks
    )
    baselines = {t.name: t.baseline for t in all_change_tasks(families)}
    result = _reduce(root, config, change_runs, comprehension_runs, baselines, mutation)
    return result, change_runs, comprehension_runs


def evaluate_codebase(
    root: Path,
    families: Sequence[TaskFamily],
    seeds: Sequence[int],
    agent: AgentRunner,
    config: EscConfig,
    env: TaskEnvironment | None = None,
    mutation: MutationResult | None = None,
    execute_change_tasks: bool = True,
) -> EscResult:
    """Evaluate ``root`` into a full :class:`EscResult` with a diagnostic dashboard."""
    result, _change, _comp = evaluate_codebase_detailed(
        root, families, seeds, agent, config, env, mutation, execute_change_tasks
    )
    return result


def _reduce(
    root: Path,
    config: EscConfig,
    change_runs: list[TaskRunResult],
    comprehension_runs: list[ComprehensionRunResult],
    baselines: dict,
    mutation: MutationResult | None,
) -> EscResult:
    s_value, s_diag = score_static_quality(root, config)
    l_value, l_diag = score_locality(root, config, change_runs)

    if change_runs:
        c_value, c_diag = score_correctness(change_runs)
        e_value, e_diag = score_edit_efficiency(change_runs, baselines, config.cost)
    else:
        c_value = e_value = None
        c_diag = e_diag = {"reason": "no change runs (static-only)"}

    if comprehension_runs:
        u_value, u_diag = score_understandability(comprehension_runs, config.understanding)
    else:
        u_value = None
        u_diag = {"reason": "no comprehension runs"}

    if mutation is not None:
        t_value, t_diag = score_testability(mutation, config.testability)
    else:
        t_value = None
        t_diag = {"reason": "no mutation/fault evidence supplied"}

    values = {
        "correctness": c_value,
        "edit_efficiency": e_value,
        "understandability": u_value,
        "locality": l_value,
        "testability": t_value,
        "static_quality": s_value,
    }
    esc, effective_weights = compute_esc_partial(values, config.weights)

    subscores = SubScores(
        correctness=c_value or 0.0,
        edit_efficiency=e_value or 0.0,
        understandability=u_value or 0.0,
        locality=l_value,
        testability=t_value or 0.0,
        static_quality=s_value,
    )
    diagnostics = {
        "measured_dimensions": [k for k, v in values.items() if v is not None],
        "unmeasured_dimensions": [k for k, v in values.items() if v is None],
        "effective_weights": effective_weights,
        "correctness": c_diag,
        "edit_efficiency": e_diag,
        "understandability": u_diag,
        "locality": l_diag,
        "testability": t_diag,
        "static_quality": s_diag,
    }
    return EscResult(esc=esc, subscores=subscores, diagnostics=diagnostics)
