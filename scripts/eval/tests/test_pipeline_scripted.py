"""End-to-end pipeline on a toy fixture.

A scripted agent applies a real edit inside the sandbox; verification runs actual
pytest against the sandbox's own source (via PYTHONPATH), the gates evaluate, and
the six dimensions aggregate. This proves the whole harness wiring without a live
model or the heavy trainer environment.
"""

from __future__ import annotations

from pathlib import Path

from esc.agents.scripted import ScriptedAgentRunner, ScriptedChange
from esc.config import EscConfig
from esc.evaluate import evaluate_codebase
from esc.model import AgentTelemetry, MutationResult, PatchStats
from esc.tasks.base import CommandChangeTask, HiddenTest
from esc.tasks.registry import TaskFamily


def _write_toy_repo(root: Path) -> None:
    (root / "src" / "toy").mkdir(parents=True)
    (root / "src" / "toy" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "toy" / "feature.py").write_text("BASE = 1\n", encoding="utf-8")


def _apply_added(workspace: Path, seed: int) -> PatchStats:
    feature = workspace / "src" / "toy" / "feature.py"
    feature.write_text(feature.read_text(encoding="utf-8") + "ADDED = 42\n", encoding="utf-8")
    return PatchStats(files_changed=1, modules_changed=1, lines_changed=1,
                      used_designated_extension_point=True)


def _toy_task() -> CommandChangeTask:
    hidden = HiddenTest(
        filename="tests/test_toy_hidden.py",
        source="from toy import feature\n\n\ndef test_added():\n    assert feature.ADDED == 42\n",
    )
    return CommandChangeTask(
        name="toy_add_constant",
        family="toy",
        prompt="Add `ADDED = 42` to toy.feature.",
        baseline=AgentTelemetry(tokens=1000, wall_time_s=10.0, retries=1, patch_lines=1),
        hidden_tests=(hidden,),
        expected_change_files=("src/toy/feature.py",),
    )


def test_full_pipeline_scripted(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_toy_repo(repo)

    task = _toy_task()
    family = TaskFamily("toy", (task,), ())
    agent = ScriptedAgentRunner(changes={
        "toy_add_constant": ScriptedChange(telemetry=task.baseline, apply=_apply_added),
    })
    mutation = MutationResult(mutants_total=10, mutants_killed=9, deterministic=True,
                              suite_runtime_s=5.0)

    result = evaluate_codebase(
        repo, [family], seeds=(0, 1), agent=agent, config=EscConfig(), mutation=mutation,
    )

    assert result.subscores.correctness == 1.0
    assert result.subscores.locality > 0.0
    assert result.esc > 0.0
    assert "correctness" in result.diagnostics["measured_dimensions"]
    assert "understandability" in result.diagnostics["unmeasured_dimensions"]


def test_pipeline_reports_failure_when_agent_does_nothing(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_toy_repo(repo)

    task = _toy_task()
    family = TaskFamily("toy", (task,), ())
    # Agent that makes no edit -> hidden test must fail -> correctness 0 -> ESC 0.
    agent = ScriptedAgentRunner(changes={
        "toy_add_constant": ScriptedChange(telemetry=task.baseline, patch_stats=PatchStats()),
    })

    result = evaluate_codebase(repo, [family], seeds=(0,), agent=agent, config=EscConfig())
    assert result.subscores.correctness == 0.0
    assert result.esc == 0.0
