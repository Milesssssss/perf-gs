"""Hard behavioural gates — the constraints that precede any scoring.

A candidate refactor is *invalid* (and receives no ESC) if it breaks existing
tests, drifts numerically beyond tolerance, regresses performance past the
ceiling, breaks the serialization/ABI, introduces nondeterminism, trips a static
blocker, or drops the mutation score too far. This encodes the optimisation as

    max ESC   subject to all behavioural constraints passing,

which is far safer than folding correctness into a weighted sum.
"""

from __future__ import annotations

from dataclasses import dataclass

from esc.config import GateThresholds
from esc.model import GateReport


@dataclass(frozen=True)
class GateInputs:
    """Everything the gates need to judge one candidate/run."""

    existing_tests_pass: bool
    max_numeric_deviation: float
    perf_ratio: float               # candidate wall time / champion wall time
    abi_stable: bool
    deterministic: bool
    static_blockers: int
    mutation_score: float
    champion_mutation_score: float | None = None


def evaluate_gates(inputs: GateInputs, thresholds: GateThresholds | None = None) -> GateReport:
    """Return a :class:`GateReport`; ``passed`` is True only if every gate holds."""
    t = thresholds or GateThresholds()
    failures: list[str] = []

    if not inputs.existing_tests_pass:
        failures.append("existing tests failed")

    if inputs.max_numeric_deviation > t.numerical_tolerance:
        failures.append(
            f"numerical deviation {inputs.max_numeric_deviation:.3e} "
            f"> tolerance {t.numerical_tolerance:.3e}"
        )

    if inputs.perf_ratio - 1.0 > t.max_perf_regression:
        failures.append(
            f"performance regressed {100 * (inputs.perf_ratio - 1.0):.1f}% "
            f"> {100 * t.max_perf_regression:.1f}%"
        )

    if t.forbid_abi_break and not inputs.abi_stable:
        failures.append("ABI/serialization break")

    if t.require_determinism and not inputs.deterministic:
        failures.append("new nondeterminism detected")

    if inputs.static_blockers > t.max_static_blockers:
        failures.append(
            f"{inputs.static_blockers} static blockers > {t.max_static_blockers}"
        )

    if inputs.mutation_score < t.min_mutation_score:
        failures.append(
            f"mutation score {inputs.mutation_score:.3f} < floor {t.min_mutation_score:.3f}"
        )

    if inputs.champion_mutation_score is not None:
        drop = inputs.champion_mutation_score - inputs.mutation_score
        if drop > t.max_mutation_drop:
            failures.append(
                f"mutation score dropped {drop:.3f} > {t.max_mutation_drop:.3f}"
            )

    return GateReport(passed=not failures, failures=tuple(failures))
