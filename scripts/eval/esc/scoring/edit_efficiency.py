# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Edit efficiency ``E`` — how cheaply successful changes are made.

    E = exp( -median_t [ w_tokens r_tokens + w_time r_time
                          + w_retries r_retries + w_patch r_patch ] )

Each ``r`` is a non-negative cost normalised against a fixed baseline version of
the codebase, so ``E`` answers: *relative to the reference design, how much
effort did the agent spend to succeed?* Only successful runs are measured — the
cost of failure is already captured by ``C``.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

from esc.config import CostNormalization, CostWeights
from esc.model import AgentTelemetry, TaskRunResult


def _normalized_cost(observed: float, baseline: float, mode: CostNormalization) -> float:
    """Turn one raw cost axis into a non-negative ``r`` term."""
    ratio = 1.0 if baseline <= 0.0 else observed / baseline
    if mode is CostNormalization.RATIO:
        return max(0.0, ratio)
    return max(0.0, ratio - 1.0)  # EXCESS_RATIO: baseline -> 0 -> E=1


def _run_cost(telemetry: AgentTelemetry, baseline: Mapping[str, float],
              weights: CostWeights) -> float:
    """Weighted sum of normalised cost axes for a single run."""
    observed = telemetry.cost_map()
    total = 0.0
    for axis, weight in weights.weight_map().items():
        r = _normalized_cost(observed[axis], baseline.get(axis, 0.0),
                             weights.normalization)
        total += weight * r
    return total


def score_edit_efficiency(
    runs: Sequence[TaskRunResult],
    baselines: Mapping[str, AgentTelemetry],
    weights: CostWeights | None = None,
) -> tuple[float, dict]:
    """Return ``(E, diagnostics)`` from the successful change-task runs."""
    weights = weights or CostWeights()
    successful = [r for r in runs if r.success]
    if not successful:
        return 0.0, {"reason": "no successful runs to measure", "measured": 0}

    per_run_costs: list[float] = []
    rows: list[dict] = []
    for run in successful:
        baseline = baselines.get(run.task_name)
        baseline_map = baseline.cost_map() if baseline else {}
        cost = _run_cost(run.telemetry, baseline_map, weights)
        per_run_costs.append(cost)
        rows.append({
            "task": run.task_name,
            "seed": run.seed,
            "weighted_cost": cost,
            "has_baseline": baseline is not None,
        })

    median_cost = statistics.median(per_run_costs)
    e = math.exp(-median_cost)
    diagnostics = {
        "measured": len(successful),
        "median_weighted_cost": median_cost,
        "normalization": weights.normalization.value,
        "rows": rows,
    }
    return e, diagnostics
