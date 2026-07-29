"""Subscore reducers and the final ESC aggregation.

Import order mirrors the six dimensions; :func:`aggregate.compute_esc` combines
them into the headline value.
"""

from __future__ import annotations

from esc.scoring.aggregate import compute_esc, compute_esc_partial, esc_from_subscores
from esc.scoring.correctness import score_correctness
from esc.scoring.edit_efficiency import score_edit_efficiency
from esc.scoring.locality import score_locality
from esc.scoring.static_quality import score_static_quality
from esc.scoring.testability import score_testability
from esc.scoring.understandability import score_understandability

__all__ = [
    "compute_esc",
    "compute_esc_partial",
    "esc_from_subscores",
    "score_correctness",
    "score_edit_efficiency",
    "score_locality",
    "score_static_quality",
    "score_testability",
    "score_understandability",
]
