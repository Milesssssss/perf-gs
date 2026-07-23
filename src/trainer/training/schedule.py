"""Pure per-iteration training schedule.

Extracts the v2 loop's inline modulo timing into one tested, side-effect-free
object. ``TrainingSchedule.events_at(it)`` returns an :class:`IterationEvents`
describing which events fire at iteration ``it``. Being pure (no GPU, no I/O)
makes the tricky invariants unit-testable and frozen against the characterization
goldens:

  * opacity reset WINS over densify on a colliding iter,
  * structure-densify fires on ABSOLUTE multiples of ``sad_structure_every``
    (only on densify ticks that are not resets),
  * preview is buffered after densify/reset (+50, or +350 on a reset iter),
  * the two distinct Mip-recompute sites (inside the densify block vs. the
    standalone periodic recompute past ``densify_to``) are modelled separately
    so the loop keeps exactly one recompute per site.
"""

from __future__ import annotations

from dataclasses import dataclass

from trainer.config_types import DensificationMode, ScheduleConfig


@dataclass(frozen=True)
class IterationEvents:
    """Which events fire at a given training iteration. ``iteration`` is carried
    for convenience; every other field is a boolean event flag."""
    iteration: int
    densify: bool
    opacity_reset: bool
    structure: bool
    preview: bool
    render_all: bool
    log: bool
    eval: bool
    psnr_curve: bool
    morton: bool
    mip_after_densify: bool   # recompute the 3D filter INSIDE the densify block
    mip_periodic: bool         # standalone periodic recompute past densify_to


class TrainingSchedule:
    """Answers ``events_at(it)`` from an immutable :class:`ScheduleConfig`.

    The logic mirrors the original inline v2_train loop EXACTLY (proven by
    tests/test_training_schedule.py) so wiring it into the loop is a
    behavior-preserving swap.
    """

    def __init__(self, cfg: ScheduleConfig) -> None:
        self._cfg = cfg

    @property
    def config(self) -> ScheduleConfig:
        return self._cfg

    def events_at(self, it: int) -> IterationEvents:
        c = self._cfg

        do_densify = (c.densify_from < it <= c.densify_to
                      and (it - c.densify_from) % c.densify_every == 0)
        do_opacity_reset = (c.reset_every > 0 and it % c.reset_every == 0
                            and c.densify_from <= it <= c.densify_to)
        # Reset WINS: a reset iter never also densifies.
        if do_densify and do_opacity_reset:
            do_densify = False
        do_structure = (c.densify_mode == DensificationMode.STRUCTURE_AWARE
                        and do_densify and (it % c.sad_structure_every == 0))

        do_preview = False
        if c.preview_every > 0:
            buffered = [itr + 350 if (c.reset_every > 0 and itr % c.reset_every == 0)
                        else itr + 50
                        for itr in range(c.preview_every, c.iterations,
                                         c.preview_every)]
            do_preview = (it in buffered) or (it == c.iterations)

        if c.render_all_every > 0:
            do_render_all = (it % c.render_all_every == 0) or (it == c.iterations)
        else:
            do_render_all = do_preview

        do_log = (c.log_every > 0
                  and ((it % c.log_every == 0) or (it == 1) or (it == c.iterations)))
        do_eval = c.eval_every > 0 and (it % c.eval_every == 0)
        do_psnr = (c.psnr_curve_enabled
                   and ((it + 1) % max(1, c.psnr_curve_every) == 0))
        do_morton = (c.morton_every > 0 and it % c.morton_every == 0
                     and not do_densify)
        mip_after_densify = c.mip_enabled and do_densify
        mip_periodic = (c.mip_enabled and it > c.densify_to
                        and it % 100 == 0 and it < c.iterations - 100)

        return IterationEvents(
            iteration=it,
            densify=do_densify,
            opacity_reset=do_opacity_reset,
            structure=do_structure,
            preview=do_preview,
            render_all=do_render_all,
            log=do_log,
            eval=do_eval,
            psnr_curve=do_psnr,
            morton=do_morton,
            mip_after_densify=mip_after_densify,
            mip_periodic=mip_periodic,
        )
