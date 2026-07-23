"""Per-step training schedules: mean/position LR + extinction hardness.

Both were nested closures inside the training entry (with a mutable
``first_densify_it`` closure variable). Extracting them into small classes
removes that mutable closure state and makes the schedules independently
unit-testable. Pure — no GPU, no I/O.
"""

from __future__ import annotations

import numpy as np


class MeanLearningRateSchedule:
    """Exponential (log-linear) mean/position-LR decay from ``lr_init`` to
    ``lr_final`` over a FIXED ``max_steps`` horizon, clamped at both ends
    (3DGS/SADGS ``get_expon_lr_func``).

    The horizon is FIXED (``position_lr_max_steps``), NOT the run length, so a
    short run doesn't compress the full decay and freeze the means before any
    geometry forms.
    """

    def __init__(self, lr_init: float, lr_final: float, max_steps: int) -> None:
        self._lr_init = lr_init
        self._lr_final = lr_final
        self._max_steps = max_steps

    def at(self, step: int) -> float:
        t = min(max(step / max(self._max_steps, 1), 0.0), 1.0)
        return float(np.exp(np.log(self._lr_init) * (1.0 - t)
                            + np.log(self._lr_final) * t))


class ExtinctionHardnessSchedule:
    """Compositor extinction-hardness warmup: hold ``k = 1.0`` (standard
    alpha-over) until the first densification, then log-linearly ramp
    ``1 -> target`` over ``warmup`` steps.

    ``target <= 1`` disables hardening; ``warmup < 0`` disables the schedule
    (constant target from step 0). The first-densify iteration that starts the
    ramp is recorded via :meth:`notify_first_densification`, replacing the old
    mutable closure variable.
    """

    def __init__(self, target: float, warmup: int) -> None:
        self._target = float(target)
        self._warmup = int(warmup)
        self._first_densify_it: int | None = None

    @property
    def target(self) -> float:
        return self._target

    def notify_first_densification(self, iteration: int) -> None:
        """Record the first densify iteration (starts the ramp). Idempotent —
        only the first call takes effect."""
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
