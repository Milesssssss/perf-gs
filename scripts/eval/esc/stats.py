"""Paired statistics for the champion/candidate decision.

The loop must decide whether a candidate is *meaningfully* better than the
champion on the *same* tasks and seeds. Each (task, seed) yields a per-run
quality that folds correctness and edit cost together; a paired bootstrap over
those aligned deltas estimates the confidence that the candidate improves on the
champion. Pairing (same tasks + seeds) removes task-difficulty variance from the
comparison.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from esc.config import CostWeights
from esc.model import AgentTelemetry, TaskRunResult
from esc.scoring.edit_efficiency import _run_cost


def run_quality(run: TaskRunResult, baseline: AgentTelemetry | None,
                weights: CostWeights) -> float:
    """A per-run scalar in ``[0, 1]``: correctness gated by edit efficiency."""
    if not run.success:
        return 0.0
    baseline_map = baseline.cost_map() if baseline else {}
    return math.exp(-_run_cost(run.telemetry, baseline_map, weights))


def align_pairs(
    champion_runs: Sequence[TaskRunResult],
    candidate_runs: Sequence[TaskRunResult],
    baselines: Mapping[str, AgentTelemetry],
    weights: CostWeights,
) -> list[tuple[float, float]]:
    """Align champion/candidate runs by (task, seed) into quality pairs."""
    def index(runs: Sequence[TaskRunResult]) -> dict[tuple[str, int], TaskRunResult]:
        return {(r.task_name, r.seed): r for r in runs}

    champ = index(champion_runs)
    cand = index(candidate_runs)
    pairs: list[tuple[float, float]] = []
    for key in sorted(champ.keys() & cand.keys()):
        b = baselines.get(key[0])
        pairs.append((run_quality(champ[key], b, weights),
                      run_quality(cand[key], b, weights)))
    return pairs


@dataclass(frozen=True)
class BootstrapResult:
    delta_mean: float
    confidence: float  # fraction of resamples with candidate > champion
    n: int


def paired_bootstrap(
    pairs: Sequence[tuple[float, float]],
    samples: int,
    rng: random.Random,
) -> BootstrapResult:
    """Bootstrap the paired mean delta (candidate - champion)."""
    deltas = [cand - champ for champ, cand in pairs]
    n = len(deltas)
    if n == 0:
        return BootstrapResult(0.0, 0.0, 0)

    observed = sum(deltas) / n
    wins = 0
    for _ in range(samples):
        resample_mean = sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        if resample_mean > 0.0:
            wins += 1
    return BootstrapResult(delta_mean=observed, confidence=wins / samples, n=n)
