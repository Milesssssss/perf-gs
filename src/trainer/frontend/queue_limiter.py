"""GPU submit-queue backpressure.

Bounds how far the CPU runs ahead of the GPU: keep a rolling window of the last
``max_depth`` submit IDs and wait on the oldest before recording a new one, so a
later ``wait_for_idle`` never has to drain a huge backlog — without ever draining
the queue in the hot path. Extracted from the training entry's nested
``_bound_queue`` closure + ``submit_id_ring`` list.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainer.backend.compute.gpu import GpuRuntime


class QueueLimiter:
    """Fence-based queue-depth cap over runtime-owned submit IDs."""

    def __init__(self, runtime: "GpuRuntime", max_depth: int = 1) -> None:
        self._runtime = runtime
        self._max_depth = max_depth
        self._ring: deque[int] = deque()

    def before_submit(self) -> None:
        """Wait on the oldest in-flight submit if the rolling window is full."""
        if len(self._ring) >= self._max_depth:
            oldest = self._ring.popleft()
            if oldest:  # 0 means "no submit yet"
                self._runtime.wait_for_submit(oldest)

    def record(self, submit_id: int) -> None:
        """Add a submit ID to the window."""
        self._ring.append(submit_id)

    def clear(self) -> None:
        """Drop all tracked submits (e.g. before a capture drains the queue)."""
        self._ring.clear()
