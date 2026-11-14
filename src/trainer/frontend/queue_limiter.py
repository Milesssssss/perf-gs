# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU submit-queue backpressure: bounds how far the CPU runs ahead of the GPU without ever
draining the queue in the hot path; depth also sets how much host and GPU work overlap."""

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
        A loop, not a single pop: callers also record unpaired submits (e.g. densify ticks)."""
        while len(self._ring) >= self._max_depth:
            oldest = self._ring.popleft()
            if oldest:  # 0 means "no submit yet"
                self._runtime.wait_for_submit(oldest)

    def record(self, submit_id: int) -> None:
        self._ring.append(submit_id)

    def clear(self) -> None:
        """Drop all tracked submits (e.g. before a capture drains the queue)."""
        self._ring.clear()
