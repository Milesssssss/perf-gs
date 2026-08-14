"""The GPU submit-queue depth cap.

Depth is a THROUGHPUT knob, not just a safety valve: at depth 1 an iteration
cannot be recorded until the previous one has finished on the GPU, so the host
half of a step (context, schedules, ~50 slangpy dispatches) is serialized against
GPU execution instead of overlapping it. Measured on bicycle: ~127 it/s at depth
1 vs ~155 at depth 3, with nothing about the GPU work changed.

The bug these tests pin: ``before_submit`` used to pop exactly ONE id, while the
densify and Morton paths each record a submit that nothing waits for. So the ring
grew by two at every densify tick (1 -> 3 -> 5 -> ...), the cap stopped applying,
and throughput jumped ~20% mid-run at the first tick — iteration 200 under
``--densify-from 100 --densify-every 100``, since the gate is ``densify_from <
it``. A run with densification off never got the speedup at all. Draining to the
cap makes the configured depth the actual depth, at every iteration.
"""

from __future__ import annotations

from trainer.frontend.queue_limiter import DEFAULT_MAX_DEPTH, QueueLimiter


class _Runtime:
    """Records which submits were waited on, in order."""

    def __init__(self) -> None:
        self.waited: list[int] = []

    def wait_for_submit(self, submit_id: int) -> None:
        self.waited.append(submit_id)


def _limiter(max_depth: int | None = None):
    runtime = _Runtime()
    limiter = (QueueLimiter(runtime) if max_depth is None
               else QueueLimiter(runtime, max_depth=max_depth))
    return limiter, runtime


def test_the_default_depth_lets_the_host_run_ahead() -> None:
    """Depth 1 costs ~20% throughput by putting host submission latency on the
    critical path, so the default must be >1."""
    assert DEFAULT_MAX_DEPTH > 1
    limiter, _ = _limiter()
    assert limiter._max_depth == DEFAULT_MAX_DEPTH


def test_nothing_is_waited_on_until_the_window_is_full() -> None:
    limiter, runtime = _limiter(3)
    for submit_id in (1, 2, 3):
        limiter.before_submit()
        limiter.record(submit_id)
    assert runtime.waited == []          # exactly `depth` submits fit
    assert limiter.depth == 3


def test_a_full_window_waits_on_the_oldest_submit() -> None:
    limiter, runtime = _limiter(3)
    for submit_id in (1, 2, 3):
        limiter.before_submit()
        limiter.record(submit_id)

    limiter.before_submit()
    assert runtime.waited == [1]         # the oldest, not the newest
    limiter.record(4)
    limiter.before_submit()
    assert runtime.waited == [1, 2]


def test_unpaired_records_cannot_inflate_the_window() -> None:
    """THE regression. A densify tick records two submits nothing waits for; the
    window must come back to the cap on the next iteration instead of keeping the
    extra slack forever."""
    limiter, runtime = _limiter(3)
    for submit_id in (1, 2, 3):
        limiter.before_submit()
        limiter.record(submit_id)

    limiter.record(4)          # densify pass
    limiter.record(5)          # morton re-sort
    assert limiter.depth == 5

    limiter.before_submit()    # next iteration: drains back to the cap
    assert limiter.depth == 2
    limiter.record(6)
    assert limiter.depth == 3

    # ... and stays there, however many ticks go by.
    for extra in range(7, 13):
        limiter.record(extra)
        limiter.before_submit()
        limiter.record(extra + 100)
        assert limiter.depth <= 3


def test_depth_one_is_still_strict_lockstep_when_asked_for() -> None:
    """The old behaviour remains available (profiling wants a shallow queue), and
    at depth 1 every iteration waits for its predecessor."""
    limiter, runtime = _limiter(1)
    limiter.before_submit()
    limiter.record(1)
    limiter.before_submit()
    assert runtime.waited == [1]
    limiter.record(2)
    limiter.before_submit()
    assert runtime.waited == [1, 2]


def test_a_zero_submit_id_is_never_waited_on() -> None:
    """0 means "no submit yet" — waiting on it would be a fence that never
    signals."""
    limiter, runtime = _limiter(1)
    limiter.record(0)
    limiter.before_submit()
    assert runtime.waited == []


def test_clear_drops_the_window_without_waiting() -> None:
    """A capture drains the queue itself, so the tracked ids are stale rather
    than in flight."""
    limiter, runtime = _limiter(3)
    for submit_id in (1, 2, 3):
        limiter.record(submit_id)
    limiter.clear()
    assert limiter.depth == 0
    limiter.before_submit()
    assert runtime.waited == []


def test_a_nonsense_depth_is_clamped_to_one() -> None:
    limiter, _ = _limiter(0)
    assert limiter._max_depth == 1
