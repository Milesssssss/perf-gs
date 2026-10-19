# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Testability and regression safety ``T``.

Combines regression-detecting power (mutation score, seeded-fault detection)
with test hygiene (determinism, flakiness, runtime) and an explicit penalty for
the failure mode where an agent *weakens tests* to force a patch through. A
refactor that makes behaviour harder to verify must lose points even if it looks
cleaner.
"""

from __future__ import annotations

from esc.config import TestabilityThresholds
from esc.model import MutationResult
from esc.scoring.static_quality import linear_decreasing


def score_testability(
    mutation: MutationResult,
    thresholds: TestabilityThresholds | None = None,
) -> tuple[float, dict]:
    """Return ``(T, diagnostics)`` from regression-safety evidence."""
    t = thresholds or TestabilityThresholds()

    have_mutation = mutation.mutants_total > 0
    have_faults = mutation.seeded_faults_total > 0
    if have_mutation and have_faults:
        coverage = t.mutation_weight * mutation.mutation_score + t.fault_weight * mutation.fault_detection
    elif have_mutation:
        coverage = mutation.mutation_score
    elif have_faults:
        coverage = mutation.fault_detection
    else:
        coverage = 0.0

    determinism_factor = 1.0 if mutation.deterministic else t.nondeterministic_factor
    flaky_factor = max(0.0, 1.0 - t.per_flaky_penalty * mutation.flaky_tests)
    runtime_factor = linear_decreasing(mutation.suite_runtime_s, t.runtime_good_s, t.runtime_bad_s)
    weakened_factor = t.weakened_tests_factor if mutation.tests_weakened_by_agent else 1.0

    value = coverage * determinism_factor * flaky_factor * runtime_factor * weakened_factor
    value = max(0.0, min(1.0, value))

    diagnostics = {
        "coverage": coverage,
        "mutation_score": mutation.mutation_score,
        "fault_detection": mutation.fault_detection,
        "determinism_factor": determinism_factor,
        "flaky_factor": flaky_factor,
        "runtime_factor": runtime_factor,
        "weakened_factor": weakened_factor,
    }
    return value, diagnostics
