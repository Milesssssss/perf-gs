# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU-timestamp per-marker timing for one recorded train_step.

A :class:`StepTimer` owns a Vulkan/D3D12 timestamp query pool and turns the
timestamps a recorded command batch writes at marker-group boundaries into
per-marker millisecond durations. It is OFF by default: ``Pipeline`` only writes
timestamps when a timer is installed (see ``Pipeline.benchmark_frozen``), so the
normal training hot path records no extra GPU work.

Design: the timer is *marker-driven*, not fixed-schema. ``Pipeline._timed`` (a
thin wrapper around ``_marker_group``) claims a fresh (start, end) timestamp pair
per marker scope it enters and remembers the label + index pair. After the submit
completes, :meth:`read` resolves the pool once and reports each label's summed
duration (a label entered N times per step sums its N intervals). Nested markers
are supported: the top-level "forward" interval spans its inner "bin_count",
"radix_sort", ... intervals, so both the coarse phase and the fine kernel show up.

``get_timestamp_result`` returns seconds already (the RHI applies the device
timestamp period), so durations are scaled to milliseconds here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import slangpy as spy

# Generous pool: one train_step records well under this many boundaries even
# with every marker group instrumented (forward ~8, backward ~12, adam ~3, ...).
DEFAULT_POOL_SIZE = 128


@dataclass
class StepTiming:
    """Per-marker millisecond durations for one recorded train_step.

    ``markers_ms`` maps each label to its summed duration across every time it
    was entered in the step. Order is preserved (first-seen order)."""

    markers_ms: dict[str, float] = field(default_factory=dict)

    def get(self, label: str, default: float = 0.0) -> float:
        return self.markers_ms.get(label, default)


class StepTimer:
    """Owns a timestamp query pool and the per-step interval bookkeeping.

    Per step: :meth:`begin_step` (resets the pool + interval list), the recorded
    batch calls :meth:`open`/:meth:`close` at each marker boundary, then after the
    submit completes :meth:`read` resolves the pool into per-marker durations.
    """

    def __init__(self, device: spy.Device, pool_size: int = DEFAULT_POOL_SIZE) -> None:
        self._pool = device.create_query_pool(spy.QueryType.timestamp, pool_size)
        self._pool_size = pool_size
        self._next_index = 0
        # (label, start_index, end_index) for each entered marker scope.
        self._intervals: list[tuple[str, int, int]] = []

    @property
    def pool(self) -> spy.QueryPool:
        return self._pool

    def begin_step(self) -> None:
        """Reset the pool + interval bookkeeping for a fresh step. Call between
        two GPU-completed submits (host-side)."""
        self._pool.reset()
        self._next_index = 0
        self._intervals = []

    def _claim(self) -> int:
        idx = self._next_index
        self._next_index += 1
        return idx

    def open(self, label: str, encoder: spy.CommandEncoder) -> int:
        """Record a start timestamp for ``label`` and return its interval id.
        Returns -1 (no-op) if the pool is exhausted."""
        if self._next_index + 1 >= self._pool_size:
            return -1
        start = self._claim()
        end = self._claim()
        encoder.write_timestamp(self._pool, start)
        self._intervals.append((label, start, end))
        return len(self._intervals) - 1

    def close(self, interval_id: int, encoder: spy.CommandEncoder) -> None:
        """Record the end timestamp for the interval opened as ``interval_id``."""
        if interval_id < 0:
            return
        _, _, end = self._intervals[interval_id]
        encoder.write_timestamp(self._pool, end)

    def read(self) -> StepTiming:
        """Resolve the pool into per-marker summed millisecond durations. Call
        only after the submit that wrote the timestamps has completed."""
        n = self._next_index
        if n == 0:
            return StepTiming()
        ts = self._pool.get_timestamp_results(0, n)
        markers: dict[str, float] = {}
        for label, start, end in self._intervals:
            dur = max(0.0, (ts[end] - ts[start]) * 1e3)
            markers[label] = markers.get(label, 0.0) + dur
        return StepTiming(markers_ms=markers)
