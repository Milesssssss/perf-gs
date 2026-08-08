"""Densification subsystem controller for the training loop.

The v2 loop's ``if do_densify:`` block was ~200 lines of tightly-ordered GPU work
mixing structure-aware / AbsGS filler / simple branches with the SAD-GS eta-window
carry logic, a post-densify Morton
re-sort, the Mip-Splatting filter recompute, population-trajectory snapshots and
a read-back diagnostics pass. :class:`DensificationController` owns that whole
subsystem: it is constructed once with its dependencies, and :meth:`run` executes
one densify tick and returns the (possibly updated) live gaussian count.

The body of :meth:`run` is a verbatim lift of the original inline block (the
loop-local names are re-bound from instance state at the top of the method), so
the extraction is behavior-preserving; the ordering invariants documented inline
(clears BEFORE Morton, eta carry on filler ticks, mip recompute after count
change) are exactly as they were in ``app.main()``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from trainer.config.config_types import DensificationMode

if TYPE_CHECKING:
    import argparse

    from trainer.backend.render.pipeline import TrainingPipeline
    from trainer.frontend.lr_schedule import ExtinctionHardnessSchedule
    from trainer.frontend.queue_limiter import QueueLimiter
    from trainer.frontend.schedule import IterationEvents


@dataclass
class DensifyOutcome:
    """Result of one densify tick. The controller produces this; the reporting
    layer (:meth:`TrainingReporter.report_densify`) turns it into JSON + console
    lines, so orchestration and reporting stay separate."""

    iteration: int
    count: int
    kind: str
    stats: dict


class DensificationController:
    """Owns simple/structure densification plus bookkeeping."""

    def __init__(
        self,
        *,
        pipeline: "TrainingPipeline",
        args: "argparse.Namespace",
        densify_mode: DensificationMode,
        densify_cfg,
        structure_cfg,
        extent: float,
        image_width: int,
        image_height: int,
        morton_world_min,
        morton_world_max,
        extinction_hardness_schedule: "ExtinctionHardnessSchedule",
        queue_limiter: "QueueLimiter",
    ) -> None:
        self._pipeline = pipeline
        self._args = args
        self._densify_mode = densify_mode
        self._densify_cfg = densify_cfg
        self._structure_cfg = structure_cfg
        self._extent = extent
        self._image_width = image_width
        self._image_height = image_height
        self._morton_world_min = morton_world_min
        self._morton_world_max = morton_world_max
        self._extinction_hardness_schedule = extinction_hardness_schedule
        self._queue_limiter = queue_limiter

    def run(self, it: int, ev: "IterationEvents", cur_n_gaussians: int) -> DensifyOutcome:
        """Execute one densify tick and return its outcome (updated count,
        stats, and any diagnostics). Reporting/IO is the caller's job — see
        :meth:`TrainingReporter.report_densify`."""
        # Re-bind loop-local names from instance state so the body below is a
        # verbatim lift of the original inline ``if do_densify:`` block.
        pipeline = self._pipeline
        args = self._args
        mode = self._densify_mode
        densify_cfg = self._densify_cfg
        structure_cfg = self._structure_cfg
        extent = self._extent
        morton_world_min = self._morton_world_min
        morton_world_max = self._morton_world_max
        extinction_hardness_schedule = self._extinction_hardness_schedule
        queue_limiter = self._queue_limiter
        do_structure_step = ev.structure

        # First densify starts the extinction-hardness ramp (held at 1.0 up to here; see ExtinctionHardnessSchedule).
        extinction_hardness_schedule.notify_first_densification(it)

        stats = {}

        sad_carry_eta = (mode == DensificationMode.STRUCTURE_AWARE
                         and not do_structure_step)

        if do_structure_step:
            stats = pipeline.densify_structure_aware(structure_cfg)
        elif mode == DensificationMode.STRUCTURE_AWARE:
            allow_scale_prune = it > args.reset_every
            stats = pipeline.densify_simple(
                densify_cfg,
                grad_thresh=args.grad_thresh,
                scale_thresh=extent * args.grow_scale3d,
                prune_big_scale=extent * args.prune_scale3d,
                allow_scale_prune=allow_scale_prune,
                carry_eta=True,
            )
        elif mode == DensificationMode.SIMPLE:
            allow_scale_prune = it > args.reset_every
            stats = pipeline.densify_simple(
                densify_cfg,
                grad_thresh=args.grad_thresh,
                scale_thresh=extent * args.grow_scale3d,
                prune_big_scale=extent * args.prune_scale3d,
                allow_scale_prune=allow_scale_prune,
            )
        if pipeline.last_submit_id:
            queue_limiter.record(pipeline.last_submit_id)
        pipeline.clear_tracking()

        if mode == DensificationMode.STRUCTURE_AWARE and not sad_carry_eta:
            pipeline.clear_structure_scores()

        # Morton re-sort after every densify.
        pipeline.morton_sort(morton_world_min, morton_world_max,
                             carry_eta=sad_carry_eta)
        if pipeline.last_submit_id:
            queue_limiter.record(pipeline.last_submit_id)

        if ev.mip_after_densify:
            pipeline.compute_3d_filter()

        if stats.get("final_count") is not None:
            cur_n_gaussians = stats["final_count"]
        kind = "structure" if do_structure_step else (
            "absgs" if mode == DensificationMode.STRUCTURE_AWARE else "densify")

        return DensifyOutcome(
            iteration=it, count=cur_n_gaussians, kind=kind, stats=stats)
