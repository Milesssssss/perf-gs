# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Correctness ``C`` — the 40%-weight dimension.

    C = successful agent runs / total agent runs

A run counts as successful only if it cleared every hard gate *and* the
task-specific hidden tests (both already folded into ``TaskRunResult.success``).
This is the dimension that most directly measures "can unseen change be made
without breaking things", which is why it carries the largest weight.
"""

from __future__ import annotations

from collections.abc import Sequence

from esc.model import TaskRunResult


def score_correctness(runs: Sequence[TaskRunResult]) -> tuple[float, dict]:
    """Return ``(C, diagnostics)`` from all change-task runs."""
    if not runs:
        return 0.0, {"reason": "no change-task runs", "total": 0}

    successes = sum(1 for r in runs if r.success)
    total = len(runs)

    per_task: dict[str, dict[str, int]] = {}
    for run in runs:
        row = per_task.setdefault(run.task_name, {"success": 0, "total": 0})
        row["total"] += 1
        row["success"] += int(run.success)

    diagnostics = {
        "total": total,
        "successes": successes,
        "success_rate": successes / total,
        "per_task": {
            name: {**row, "rate": row["success"] / row["total"]}
            for name, row in per_task.items()
        },
    }
    return successes / total, diagnostics
