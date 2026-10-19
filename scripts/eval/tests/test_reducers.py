# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Subscore reducers: correctness, edit efficiency, understandability, testability."""

from __future__ import annotations

import math

from esc.config import CostWeights, TestabilityThresholds, UnderstandabilityWeights
from esc.model import (
    AgentTelemetry,
    ComprehensionRunResult,
    GateReport,
    MutationResult,
    PatchStats,
    TaskRunResult,
)
from esc.scoring.correctness import score_correctness
from esc.scoring.edit_efficiency import score_edit_efficiency
from esc.scoring.testability import score_testability
from esc.scoring.understandability import score_understandability


def _run(name: str, seed: int, success: bool, telemetry: AgentTelemetry) -> TaskRunResult:
    return TaskRunResult(name, seed, success, GateReport(success), telemetry, PatchStats())


def test_correctness_is_success_ratio():
    runs = [
        _run("t", 0, True, AgentTelemetry()),
        _run("t", 1, False, AgentTelemetry()),
        _run("t", 2, True, AgentTelemetry()),
        _run("t", 3, False, AgentTelemetry()),
    ]
    c, diag = score_correctness(runs)
    assert c == 0.5
    assert diag["per_task"]["t"]["rate"] == 0.5


def test_edit_efficiency_baseline_maps_to_one():
    baseline = AgentTelemetry(tokens=1000, wall_time_s=100, retries=1, patch_lines=50)
    run = _run("t", 0, True, baseline)
    e, _diag = score_edit_efficiency([run], {"t": baseline}, CostWeights())
    assert math.isclose(e, 1.0, rel_tol=1e-9)  # EXCESS_RATIO: equal to baseline -> r=0


def test_edit_efficiency_penalises_overspend():
    baseline = AgentTelemetry(tokens=1000, wall_time_s=100, retries=1, patch_lines=50)
    pricey = AgentTelemetry(tokens=2000, wall_time_s=100, retries=1, patch_lines=50)
    e, _ = score_edit_efficiency([_run("t", 0, True, pricey)], {"t": baseline}, CostWeights())
    # tokens doubled -> r_tokens=1, weighted 0.35 -> E=exp(-0.35).
    assert math.isclose(e, math.exp(-0.35), rel_tol=1e-9)


def test_understandability_perfect_run():
    run = ComprehensionRunResult("q", 0, answer_score=1.0, file_precision=1.0,
                                 file_recall=1.0, context_tokens=0, confidence=1.0,
                                 was_correct=True)
    u, _ = score_understandability([run], UnderstandabilityWeights())
    assert math.isclose(u, 1.0, rel_tol=1e-9)


def test_testability_uses_mutation_and_hygiene():
    mutation = MutationResult(mutants_total=10, mutants_killed=8, deterministic=True,
                              suite_runtime_s=10.0)
    t, diag = score_testability(mutation, TestabilityThresholds())
    assert math.isclose(t, 0.8, rel_tol=1e-9)
    assert diag["runtime_factor"] == 1.0


def test_testability_weakened_tests_are_punished():
    honest = MutationResult(mutants_total=10, mutants_killed=8, suite_runtime_s=10.0)
    weakened = MutationResult(mutants_total=10, mutants_killed=8, suite_runtime_s=10.0,
                              tests_weakened_by_agent=True)
    t_honest, _ = score_testability(honest)
    t_weak, _ = score_testability(weakened)
    assert t_weak < t_honest
