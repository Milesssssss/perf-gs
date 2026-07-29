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

import json
from typing import TYPE_CHECKING, Callable

from trainer.config.config_types import DensificationMode

if TYPE_CHECKING:
    import argparse

    from trainer.backend.render.pipeline import Pipeline
    from trainer.frontend.lr_schedule import ExtinctionHardnessSchedule
    from trainer.frontend.reporting.outputs import RunArtifacts
    from trainer.frontend.queue_limiter import QueueLimiter
    from trainer.frontend.schedule import IterationEvents


class DensificationController:
    """Owns simple/structure densification plus bookkeeping."""

    def __init__(
        self,
        *,
        pipeline: "Pipeline",
        args: "argparse.Namespace",
        densify_mode: DensificationMode,
        densify_cfg,
        structure_cfg,
        extent: float,
        image_width: int,
        image_height: int,
        morton_world_min,
        morton_world_max,
        extinction: "ExtinctionHardnessSchedule",
        queue_limiter: "QueueLimiter",
        artifacts: "RunArtifacts",
        diag_module,
        diag_grad_internal,
        print_opacity_bundle: Callable[[str], None],
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
        self._extinction = extinction
        self._queue_limiter = queue_limiter
        self._artifacts = artifacts
        self._diag = diag_module
        self._diag_grad_internal = diag_grad_internal
        self._print_opacity_bundle = print_opacity_bundle

    def run(self, it: int, ev: "IterationEvents", cur_n_gaussians: int) -> int:
        """Execute one densify tick and return the (possibly updated) count."""
        # Re-bind loop-local names from instance state so the body below is a
        # verbatim lift of the original inline ``if do_densify:`` block.
        pipeline = self._pipeline
        args = self._args
        mode = self._densify_mode
        densify_cfg = self._densify_cfg
        structure_cfg = self._structure_cfg
        extent = self._extent
        IMAGES_WIDTH = self._image_width
        IMAGES_HEIGHT = self._image_height
        morton_world_min = self._morton_world_min
        morton_world_max = self._morton_world_max
        extinction = self._extinction
        queue_limiter = self._queue_limiter
        densify_json = self._artifacts.densify_json
        densify_diag_json = self._artifacts.densify_diag_json
        _diag = self._diag
        _diag_grad_internal = self._diag_grad_internal
        _print_opacity_bundle = self._print_opacity_bundle
        do_structure_step = ev.structure

        # First densify starts the extinction-hardness ramp (held at 1.0 up
        # to here; see ExtinctionHardnessSchedule).
        extinction.notify_first_densification(it)

        stats = {}

        # DIAGNOSTIC pre-capture: read back eta/grad/scale + spatial state
        # BEFORE the densify mutates anything, so gate selectivity and
        # fan-out reflect the decision this tick actually made.
        _diag_rec = None
        if densify_diag_json is not None:
            _diag_rec = _diag.capture_pre(
                pipeline, structure=(mode == DensificationMode.STRUCTURE_AWARE),
                width=IMAGES_WIDTH, height=IMAGES_HEIGHT,
                structure_cfg=structure_cfg,
                grad_thresh_internal=_diag_grad_internal,
                dense_scale=structure_cfg.dense_scale)

        sad_carry_eta = (mode == DensificationMode.STRUCTURE_AWARE
                         and not do_structure_step)

        if do_structure_step:
            _print_opacity_bundle(f"structure pass @ iter {it}")
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

        # Morton re-sort after every densify
        if True:
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
        densify_json.write(json.dumps(
            {"iter": it, "event": "densify", "kind": kind, **stats}) + "\n")

        # DIAGNOSTIC post-capture: realized growth + post-densify spatial concentration, after densify + morton have settled.
        if _diag_rec is not None:
            _diag_rec = _diag.capture_post(pipeline, _diag_rec,
                                           structure=(mode == DensificationMode.STRUCTURE_AWARE))
            _diag_rec["iter"] = it
            _diag_rec["kind"] = kind
            densify_diag_json.write(json.dumps(_diag_rec) + "\n")
            fo = _diag_rec.get("fanout", {})
            gate = _diag_rec.get("eta_gate", {})
            sp = _diag_rec.get("spatial_post", {})
            print(f"  [diag] {kind:9s} {_diag_rec['pre_count']}->{_diag_rec['post_count']} "
                  f"(x{_diag_rec['growth_factor']}) | "
                  f"trig={gate.get('frac_triggered_of_seen')} "
                  f"prune={gate.get('frac_prune_of_seen')} "
                  f"prod_p90={fo.get('prod_p90')} prod_max={fo.get('prod_max')} "
                  f"| gini={sp.get('gini')} top1%={sp.get('top1pct_cell_share')}")
            # Opacity-bundle assertions (diagnostic mode only): verify the
            # shader actually honored the RESOLVED floor/cap on THIS pass.
            # Asserted against the configured values (not the reference
            # 0.1/0.8), so a default run (floor 0.005, cap off) never
            # false-fails; the assert only bites if the bundle was
            # configured but did not apply on the compacted (active) buffer.
            if kind == "structure":
                _amin, _amax, _abmsgs = _diag.assert_opacity_bundle(
                    pipeline,
                    floor=structure_cfg.t_opacity,
                    cap_enabled=bool(structure_cfg.opacity_cap
                                     and structure_cfg.opacity_cap > 0.0),
                    cap=structure_cfg.opacity_cap)
                print(f"  [opacity-bundle assert] iter {it}: min={_amin:.4f} "
                      f"max={_amax:.4f} | " + " | ".join(_abmsgs))
        # Only print a per-densify line when we have real counts — i.e. the
        # synchronous score-ranked densify() path. Wait-free paths (AbsGS
        # + structure) report None and read the counter on log ticks
        # instead, matching densify_simple's discipline.
        pre_s = stats.get("pre_count")
        final_s = stats.get("final_count")
        if pre_s is not None and final_s is not None:
            print(f"iter {it}: {kind:9s}| pre={pre_s} -> post={final_s}")

        return cur_n_gaussians
