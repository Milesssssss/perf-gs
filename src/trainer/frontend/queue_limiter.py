"""GPU submit-queue backpressure.

Bounds how far the CPU runs ahead of the GPU: keep a rolling window of the last
``max_depth`` submit IDs and wait on the oldest before recording a new one, so a
later ``wait_for_idle`` never has to drain a huge backlog — without ever draining
the queue in the hot path. Extracted from the training entry's nested
``_bound_queue`` closure + ``submit_id_ring`` list.

DEPTH IS A THROUGHPUT KNOB, not just a safety valve. At depth 1 an iteration
cannot be recorded until the previous one has FINISHED on the GPU, so the host
half of a step — building the IterationContext, resolving schedules, and
marshalling ~50 slangpy dispatches — is serialized against GPU execution instead
of overlapping it. Measured on bicycle (54k splats, sh0, mip): depth 1 gives
~127 it/s, depth 3 gives ~155 it/s. Nothing about the GPU work changed; the host
latency simply stopped being on the critical path.

That difference used to appear MID-RUN and look like magic. The densify and
Morton paths each ``record`` a submit that nothing ever waits for, so the ring
grew by two at every densify tick (1 -> 3 -> 5 -> ...) and the cap silently
stopped applying. With ``--densify-from 100 --densify-every 100`` the first tick
is at iteration 200 (the schedule gate is ``densify_from < it``), which is
exactly where a run appeared to speed up by ~20% for no reason — and a run with
``--densify-mode none`` never got the speedup at all. :meth:`before_submit` now
drains to the cap rather than popping exactly one, so the depth is what it says.
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

        A LOOP, not a single pop: callers record submits this method never gets a
        chance to pair with (a densify tick records two extra), and popping one
        per call let the window grow without bound — which turned the cap into a
        suggestion and made throughput depend on how many densify ticks had gone
        by. Draining to the cap makes the configured depth the actual depth.
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
