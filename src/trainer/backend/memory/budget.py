# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Fit the run's GPU allocations to a VRAM budget."""
from __future__ import annotations

from collections.abc import Callable

from trainer.backend.memory.vram import query_video_memory

GIB = 1024 ** 3


def available_bytes(budget_gib: float, adapter_name: str) -> int | None:
    """Bytes this run may still allocate, or None when no budget is known.

    ``budget_gib`` > 0 overrides the DXGI budget (e.g. to emulate a smaller
    card). A reserve is held back for driver overhead and the allocations the
    estimate does not count (staging heaps, setup temporaries, samplers).
    """
    measured = query_video_memory(adapter_name)
    if budget_gib > 0:
        budget = int(budget_gib * GIB)
    elif measured is not None:
        budget = measured[0]
    else:
        return None
    usage = measured[1] if measured is not None else 0
    # shortcut: fixed 10% (bicycle SH3 run: +0.26 GiB unplanned, DXGI budget fell
    # 1.4 GiB mid-run); re-measure if runs page or the reserve proves wasteful.
    return budget - usage - max(GIB // 2, budget // 10)


def fit_capacity(requested: int, available: int, cost: Callable[[int], int]) -> int:
    """Largest capacity <= ``requested`` with ``cost(capacity) <= available``.

    ``cost`` must not decrease with capacity. Returns 0 when nothing fits.
    """
    low, high = 0, requested
    while low < high:
        middle = (low + high + 1) // 2
        if cost(middle) <= available:
            low = middle
        else:
            high = middle - 1
    return low
