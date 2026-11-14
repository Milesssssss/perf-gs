# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Opacity reset (periodic) and opacity cap (after structure densification)."""

from __future__ import annotations

from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import AdamResources, PrimitiveStorage

_CLAMP = 0      # opacity <- min(opacity, logit(value))
_RELATIVE = 1   # opacity <- opacity * value


class OpacityReset:
    KERNELS = {"reset": ("densify/opacity_reset.slang", "resetOpacity")}

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary, *,
                 primitive: PrimitiveStorage, adam: AdamResources, capacity: int) -> None:
        self._commands = commands
        self._k = kernels.load(self.KERNELS)
        self._primitive = primitive
        self._adam = adam
        self._capacity = capacity

    def reset(self, cfg) -> None:
        """Clamp to ``cfg.t_opacity_reset_value``, or decay by ``cfg.reset_decay``
        when ``cfg.reset_relative`` (keeps the strong/weak ordering)."""
        if getattr(cfg, "reset_relative", False):
            self._dispatch("opacity_reset", cfg.reset_decay, _RELATIVE)
        else:
            self._dispatch("opacity_reset", cfg.t_opacity_reset_value, _CLAMP)

    def cap(self, value: float, marker: str = "opacity_cap") -> None:
        self._dispatch(marker, float(value), _CLAMP)

    def _dispatch(self, marker: str, value: float, mode: int) -> None:
        self._commands.dispatch(self._k.reset, marker,
            thread_count=[self._capacity, 1, 1],
            gaussians=self._primitive.values.storage,
            adam_state=self._adam.state.storage,
            num_gaussians_counter=self._primitive.active_count.storage,
            reset_value=value,
            mode=mode)
