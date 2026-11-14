# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Understandability ``U``: can a fresh agent build the right mental model from read-only questions?

Each run blends answer correctness, cited-file F1, calibration (1 - Brier) and context economy.
"""

from __future__ import annotations

from collections.abc import Sequence

from esc.config import UnderstandabilityWeights
from esc.model import ComprehensionRunResult


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _context_term(context_tokens: int, budget: int) -> float:
    if budget <= 0:
        return 1.0
    return max(0.0, 1.0 - context_tokens / budget)


def _calibration_term(confidence: float, was_correct: bool) -> float:
    """1 minus the Brier score of a single confident/correct pair."""
    brier = (confidence - (1.0 if was_correct else 0.0)) ** 2
    return 1.0 - brier


def score_understandability(
    runs: Sequence[ComprehensionRunResult],
    weights: UnderstandabilityWeights | None = None,
) -> tuple[float, dict]:
    """Return ``(U, diagnostics)`` averaged over comprehension runs."""
    weights = weights or UnderstandabilityWeights()
    if not runs:
        return 0.0, {"reason": "no comprehension runs", "measured": 0}

    rows: list[dict] = []
    per_run: list[float] = []
    for run in runs:
        files = _f1(run.file_precision, run.file_recall)
        context = _context_term(run.context_tokens, weights.context_token_budget)
        calibration = _calibration_term(run.confidence, run.was_correct)
        combined = (
            weights.answer * run.answer_score
            + weights.files * files
            + weights.calibration * calibration
            + weights.context * context
        )
        per_run.append(combined)
        rows.append({
            "task": run.task_name,
            "seed": run.seed,
            "answer": run.answer_score,
            "file_f1": files,
            "calibration": calibration,
            "context": context,
            "combined": combined,
        })

    u = sum(per_run) / len(per_run)
    return u, {"measured": len(runs), "rows": rows}
