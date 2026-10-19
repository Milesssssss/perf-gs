# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Grading for comprehension tasks.

Turns an agent's raw :class:`~esc.agents.base.ComprehensionResult` into a
scored :class:`~esc.model.ComprehensionRunResult` by comparing the free-text
answer against a keyword key and the cited files against the ground-truth files.
File matching is suffix-based so an agent may cite either a repo-relative path or
a bare filename.
"""

from __future__ import annotations

from esc.agents.base import ComprehensionResult
from esc.model import ComprehensionRunResult
from esc.tasks.base import ComprehensionTask


def _keyword_coverage(answer: str, keywords: tuple[str, ...]) -> float:
    if not keywords:
        return 0.0
    lowered = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lowered)
    return hits / len(keywords)


def _normalize(path: str) -> str:
    return path.replace("\\", "/").lstrip("./").lower()


def _file_matches(cited: tuple[str, ...], expected: tuple[str, ...]) -> tuple[float, float]:
    if not expected:
        return (1.0, 1.0)
    norm_expected = [_normalize(e) for e in expected]
    norm_cited = [_normalize(c) for c in cited]

    matched_expected = sum(
        1 for e in norm_expected if any(c.endswith(e) or e.endswith(c) for c in norm_cited)
    )
    matched_cited = sum(
        1 for c in norm_cited if any(c.endswith(e) or e.endswith(c) for e in norm_expected)
    )
    recall = matched_expected / len(norm_expected)
    precision = matched_cited / len(norm_cited) if norm_cited else 0.0
    return precision, recall


def grade_comprehension(
    task: ComprehensionTask,
    result: ComprehensionResult,
    seed: int,
) -> ComprehensionRunResult:
    """Score one comprehension run."""
    answer_score = _keyword_coverage(result.answer, task.answer_keywords)
    precision, recall = _file_matches(result.cited_files, task.expected_files)
    was_correct = answer_score >= 0.5 and recall >= 0.5
    return ComprehensionRunResult(
        task_name=task.name,
        seed=seed,
        answer_score=answer_score,
        file_precision=precision,
        file_recall=recall,
        context_tokens=result.context_tokens,
        confidence=result.confidence,
        was_correct=was_correct,
    )
