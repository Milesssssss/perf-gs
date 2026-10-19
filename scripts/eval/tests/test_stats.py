# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Paired statistics for the champion/candidate decision."""

from __future__ import annotations

import math
import random

from esc.config import CostWeights
from esc.model import AgentTelemetry, GateReport, PatchStats, TaskRunResult
from esc.stats import align_pairs, paired_bootstrap, run_quality


def _run(name: str, seed: int, success: bool, telemetry: AgentTelemetry) -> TaskRunResult:
    return TaskRunResult(name, seed, success, GateReport(success), telemetry, PatchStats())


def test_run_quality_success_at_baseline_is_one():
    base = AgentTelemetry(tokens=1000, wall_time_s=100, retries=1, patch_lines=50)
    q = run_quality(_run("t", 0, True, base), base, CostWeights())
    assert math.isclose(q, 1.0, rel_tol=1e-9)


def test_run_quality_failure_is_zero():
    base = AgentTelemetry(tokens=1000)
    assert run_quality(_run("t", 0, False, base), base, CostWeights()) == 0.0


def test_align_pairs_matches_by_task_and_seed():
    base = AgentTelemetry(tokens=1000, wall_time_s=100, retries=1, patch_lines=50)
    champ = [_run("t", 0, False, base)]
    cand = [_run("t", 0, True, base)]
    pairs = align_pairs(champ, cand, {"t": base}, CostWeights())
    assert pairs == [(0.0, 1.0)]


def test_bootstrap_confident_when_candidate_always_wins():
    pairs = [(0.0, 1.0)] * 8
    result = paired_bootstrap(pairs, samples=500, rng=random.Random(0))
    assert result.confidence == 1.0
    assert math.isclose(result.delta_mean, 1.0, rel_tol=1e-9)


def test_bootstrap_unconfident_when_no_difference():
    pairs = [(1.0, 1.0)] * 8
    result = paired_bootstrap(pairs, samples=500, rng=random.Random(0))
    assert result.confidence == 0.0
    assert result.delta_mean == 0.0
