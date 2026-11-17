# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Pure per-iteration training schedule: ``TrainingSchedule.events_at(it)`` says which
events fire at iteration ``it``, with no GPU or I/O so its invariants are unit-testable."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from trainer.config.config import (
    AdamLRs, AppearanceConfig, IterationContextualVariables, IterationEvents,
)
from trainer.config.config import ScheduleConfig
from trainer.config.config_types import DensificationMode
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
        # Reset wins: a reset iter never also densifies.
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
            do_render_all = do_preview and c.render_all_on_preview

        do_log = (c.log_every > 0
                  and ((it % c.log_every == 0) or (it == 1) or (it == c.iterations)))
        do_eval = ((c.eval_every > 0 and (it % c.eval_every == 0))
                   or (it in c.eval_iters))
        do_psnr = (c.psnr_curve_enabled
                   and ((it + 1) % max(1, c.psnr_curve_every) == 0))
        do_morton = (c.morton_every > 0 and it % c.morton_every == 0
                     and not do_densify)
        mip_after_densify = c.mip_enabled and do_densify
        mip_periodic = (c.mip_enabled and it > c.densify_to
                        and it % 100 == 0 and it < c.iterations - 100)
        blend_prune = (do_densify and c.prune_blend_weight > 0.0
                       and it >= c.blend_prune_start
                       and not self._reset_in_window(it))

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
            blend_prune=blend_prune,
        )

    def _reset_in_window(self, it: int) -> bool:
        """Whether an opacity reset fired during the densify window ending at
        ``it`` (a reset replaces its tick, so that window spans two intervals)."""
        c = self._cfg
        if c.reset_every <= 0:
            return False
        last_reset = (it - 1) // c.reset_every * c.reset_every
        return (last_reset > 0 and last_reset >= it - c.densify_every
                and c.densify_from <= last_reset <= c.densify_to)

class LearningRateSchedule:
    """Each step's Adam learning rates. The mean LR decays log-linearly to 1% over ``mean_steps``
    (in scene-extent units); scale and rotation go log-linearly from ``scale_rotation_decay[0]``
    to ``[1]`` times their base over the run."""

    def __init__(self, base: AdamLRs, extent: float, mean_steps: int, iterations: int,
                 scale_rotation_decay: tuple[float, float]) -> None:
        self._base = base
        self._mean_init = base.mean * extent
        self._mean_final = base.mean * 1e-2 * extent
        self._mean_steps = mean_steps
        self._iterations = iterations
        self._scale_rotation_decay = scale_rotation_decay

    def at(self, step: int) -> AdamLRs:
        first, last = self._scale_rotation_decay
        factor = _log_lerp(first, last, step, self._iterations)
        return replace(self._base,
                       mean=_log_lerp(self._mean_init, self._mean_final, step, self._mean_steps),
                       scale=self._base.scale * factor,
                       rot=self._base.rot * factor)


def _log_lerp(start: float, end: float, step: int, steps: int) -> float:
    """``start`` at step 0 to ``end`` at ``steps``, log-linearly; clamped outside."""
    t = min(max(step / max(steps, 1), 0.0), 1.0)
    return float(np.exp(np.log(start) * (1.0 - t) + np.log(end) * t))


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
        return min(self.appearanceConfig.sh_degree, step // self.appearanceConfig.band_interval)
