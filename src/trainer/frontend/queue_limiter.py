# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU submit-queue backpressure.

Bounds how far the CPU runs ahead of the GPU: keep a rolling window of the last
``max_depth`` submit IDs and wait on the oldest before recording a new one, so a
later ``wait_for_idle`` never has to drain a huge backlog — without ever draining
the queue in the hot path.

Depth is a throughput knob, not only a safety valve: at depth 1 an iteration
cannot be recorded until the previous one has finished on the GPU, so the host
half of a step (context, schedules, ~50 dispatches) serializes against GPU
execution instead of overlapping it. Measured on bicycle: ~127 it/s at depth 1
vs ~155 at depth 3.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainer.backend.compute.gpu import GpuRuntime


DEFAULT_MAX_DEPTH = 5


class QueueLimiter:
    """Fence-based queue-depth cap over runtime-owned submit IDs."""

    def __init__(self, runtime: "GpuRuntime",
                 max_depth: int = DEFAULT_MAX_DEPTH) -> None:
        self._runtime = runtime
        self._max_depth = max(1, int(max_depth))
        self._ring: deque[int] = deque()

    @property
    def depth(self) -> int:
        """Submits currently tracked as in flight (a test/diagnostic view)."""
        return len(self._ring)

    def before_submit(self) -> None:
        """Wait until fewer than ``max_depth`` submits are in flight.

        A loop, not a single pop: callers record submits this never pairs with (a
        densify tick records two extra), and popping one per call let the window
        grow without bound — which made throughput depend on how many densify
        ticks had gone by.
        """
        while len(self._ring) >= self._max_depth:
            oldest = self._ring.popleft()
            if oldest:  # 0 means "no submit yet"
                self._runtime.wait_for_submit(oldest)

    def record(self, submit_id: int) -> None:
        """Add a submit ID to the window."""
        self._ring.append(submit_id)

    def clear(self) -> None:
        """Drop all tracked submits (e.g. before a capture drains the queue)."""
        self._ring.clear()
