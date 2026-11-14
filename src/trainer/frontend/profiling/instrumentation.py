# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The training loop's profiling hooks: Nsight frames and capture, sampled stage timing."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trainer.frontend.profiling.profiler import Profiler
from trainer.frontend.profiling.stage_timing import StageTimingRecorder

if TYPE_CHECKING:
    from trainer.frontend.schedule import IterationContext


class CaptureFinished(Exception):
    """The GPU-Trace capture presented its frozen frames; the run ends without results."""


class LoopInstrumentation:
    """Wraps each iteration's step and mutations; a no-op without profiler or stage timing."""

    def __init__(self, *, profiler: Profiler | None, stage_timing: StageTimingRecorder) -> None:
        self.profiler = profiler
        self.stage_timing = stage_timing

    @contextlib.contextmanager
    def step(self, context: IterationContext, pipeline: Any,
             queue_limiter: Any) -> Iterator[None]:
        """Around ``train_step``: a due GPU-Trace capture first, then sampled timing."""
        if self.profiler is not None and self.profiler.run_capture_if_due(
                pipeline, context, queue_limiter):
            raise CaptureFinished
        self.stage_timing.before_step(context.step_number, pipeline)
        yield
        self.stage_timing.after_step(context.step_number, pipeline)

    def mutations(self, context: IterationContext,
                  pipeline: Any) -> contextlib.AbstractContextManager:
        """Around the scheduled mutations, which run in their own submits."""
        events = context.events
        kind = "structure" if events.structure else ("filler" if events.densify else "periodic")
        return self.stage_timing.mutations.measure(
            context.step_number, pipeline, kind,
            active=events.densify or events.morton or events.mip_periodic)

    def end_iteration(self, pipeline: Any) -> None:
        """Present the completed iteration as one Nsight frame."""
        if self.profiler is not None:
            self.profiler.present_frame(pipeline.last_submit_id)

    def should_stop(self) -> bool:
        """True once the profile window is closed."""
        return self.profiler is not None and self.profiler.should_stop()

    def finish(self, run_dir: Path, wall_seconds: float) -> StageTimingRecorder | None:
        """Write the stage timing of the run; the recorder, or None when off."""
        timing = self.stage_timing
        if not timing.enabled:
            return None
        timing.write(run_dir)
        summary = timing.summary()
        print(f"[stage-timing] {summary['sampled_iters']} sampled iters "
              f"(1 in {summary['sample_every']}), measured overhead "
              f"{summary['sample_overhead_s']:.2f}s of {wall_seconds:.1f}s wall "
              f"({100.0 * summary['sample_overhead_s'] / max(wall_seconds, 1e-9):.2f}%)")
        return timing
