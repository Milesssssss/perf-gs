# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

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

import numpy as np

from trainer.config.config import (
    AdamLRs, AppearanceConfig, IterationContextualVariables, IterationEvents,
)
from trainer.config.config_types import DensificationMode, ScheduleConfig
from trainer.data.colmap import ColmapImageEntry


@dataclass(frozen=True)
class IterationContext:
    step_number: int
    camera_index: int
    camera_entry: ColmapImageEntry
    learning_rates: AdamLRs
    events: IterationEvents
    iteration_contextual_variables: IterationContextualVariables


class TrainingSchedule:

    def __init__(self, cfg: ScheduleConfig) -> None:
        self._cfg = cfg

    @property
    def config(self) -> ScheduleConfig:
        return self._cfg

    def events_at(self, it: int) -> IterationEvents:
        c = self._cfg

        do_densify = (c.densify_mode != DensificationMode.NONE
                  and c.densify_from < it <= c.densify_to
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
        # Union of the periodic cadence and the explicit checkpoint ladder, so
        # --eval-iters can add irregular points without --eval-every having to
        # divide them all (and without firing on every common multiple).
        do_eval = ((c.eval_every > 0 and (it % c.eval_every == 0))
                   or (it in c.eval_iters))
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

class MeanLearningRateSchedule:

    def __init__(self, lr_init: float, lr_final: float, max_steps: int) -> None:
        self._lr_init = lr_init
        self._lr_final = lr_final
        self._max_steps = max_steps

    def at(self, step: int) -> float:
        t = min(max(step / max(self._max_steps, 1), 0.0), 1.0)
        return float(np.exp(np.log(self._lr_init) * (1.0 - t) + np.log(self._lr_final) * t))


class ExtinctionHardnessSchedule:
    
    def __init__(self, target: float, warmup: int) -> None:
        self._target = float(target)
        self._warmup = int(warmup)
        self._first_densify_it: int | None = None

    @property
    def target(self) -> float:
        return self._target

    def notify_first_densification(self, iteration: int) -> None:
        if self._first_densify_it is None:
            self._first_densify_it = iteration

    def at(self, step: int) -> float:
        first = self._first_densify_it
        if self._target <= 1.0 or self._warmup < 0:
            return self._target
        if first is None or step <= first:
            return 1.0
        if self._warmup == 0:
            return self._target
        t = min((step - first) / self._warmup, 1.0)
        # Log-linear 1 -> target (== target**t, since log(1) = 0).
        return float(np.exp(np.log(self._target) * t))

class StbnDensitySchedule:

    def __init__(self, target_fraction: float, warmup: int) -> None:
        self._target = float(target_fraction)
        self._warmup = int(warmup)

    @property
    def target(self) -> float:
        return self._target

    def at(self, step: int) -> float:
        if self._target >= 1.0 or self._warmup <= 0:
            return self._target
        t = min(max((step - 1) / self._warmup, 0.0), 1.0)
        return float(np.exp(np.log(self._target) * t))


class AppearanceSchedule:

    def __init__(self, appearanceConfig: AppearanceConfig) -> None:
        self.appearanceConfig = appearanceConfig

    def at(self, step: int) -> int:
        if self.appearanceConfig.sh_degree <= 0:
            return 0
        return min(self.appearanceConfig.sh_degree, step // self.appearanceConfig.band_interval) # one deg up every band_interval steps, capped at sh_degree
