"""The weighted geometric mean at the heart of ESC.

    ESC = 100 * C^wC * E^wE * U^wU * L^wL * T^wT * S^wS

A geometric mean (not a weighted sum) is used so that a single catastrophic
dimension drags the whole score down: if any subscore is 0, ESC is 0. That is
the property that makes "beautifully formatted but impossible to modify" score
poorly.
"""

from __future__ import annotations

import math

from esc.config import EscWeights
from esc.model import EscResult, SubScores


def clamp01(value: float) -> float:
    """Clamp a raw subscore into the unit interval."""
    if value != value:  # NaN
        return 0.0
    return max(0.0, min(1.0, value))


def compute_esc(subscores: SubScores, weights: EscWeights | None = None) -> float:
    """Weighted geometric mean of the six subscores, scaled to ``[0, 100]``.

    Zeros are honoured exactly: a catastrophic dimension yields ESC = 0 rather
    than being averaged away.
    """
    weights = weights or EscWeights()
    weights.validate()

    pairs = zip(subscores.as_ordered(), weights.as_tuple())
    log_sum = 0.0
    for (_label, raw), weight in pairs:
        value = clamp01(raw)
        if value <= 0.0:
            return 0.0
        log_sum += weight * math.log(value)
    return 100.0 * math.exp(log_sum)


def esc_from_subscores(subscores: SubScores, weights: EscWeights | None = None,
                       diagnostics: dict | None = None) -> EscResult:
    """Bundle the headline value with its subscores and evidence."""
    return EscResult(
        esc=compute_esc(subscores, weights),
        subscores=subscores,
        diagnostics=diagnostics or {},
    )


#: Stable key order matching :meth:`EscWeights.as_tuple`.
WEIGHT_KEYS: tuple[str, ...] = (
    "correctness", "edit_efficiency", "understandability",
    "locality", "testability", "static_quality",
)


def compute_esc_partial(
    values: dict[str, float | None], weights: EscWeights | None = None
) -> tuple[float, dict[str, float]]:
    """Geometric mean over the *measured* dimensions, renormalising their
    weights. A dimension whose value is ``None`` is treated as unmeasured and
    dropped (rather than as a catastrophic zero), so a run that omits e.g.
    testability still yields a meaningful ESC. Returns ``(esc, effective_weights)``.
    """
    weights = weights or EscWeights()
    weights.validate()
    weight_map = dict(zip(WEIGHT_KEYS, weights.as_tuple()))

    present = {k: v for k, v in values.items() if v is not None}
    if not present:
        return 0.0, {}

    total_weight = sum(weight_map[k] for k in present)
    effective: dict[str, float] = {}
    log_sum = 0.0
    for key, raw in present.items():
        w = weight_map[key] / total_weight
        effective[key] = w
        value = clamp01(float(raw))
        if value <= 0.0:
            return 0.0, effective
        log_sum += w * math.log(value)
    return 100.0 * math.exp(log_sum), effective
