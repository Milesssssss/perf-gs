# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Compute-kernel compilation shared by the GPU components."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import slangpy as spy

from trainer.backend.compute.device import SHADERS_DIR

# name -> (shader path relative to shaders/, entry point)
KernelSpecs = Mapping[str, tuple[str, str]]


class KernelLibrary:
    """Compiles each (shader, entry point) pair once per device."""

    def __init__(self, device: spy.Device) -> None:
        self.device = device
        self._kernels: dict[tuple[str, str], spy.ComputeKernel] = {}

    def kernel(self, path: str, entry: str) -> spy.ComputeKernel:
        key = (path, entry)
        if key not in self._kernels:
            program = self.device.load_program(str(SHADERS_DIR / path), [entry])
            self._kernels[key] = self.device.create_compute_kernel(program)
        return self._kernels[key]

    def load(self, specs: KernelSpecs) -> SimpleNamespace:
        return SimpleNamespace(**{name: self.kernel(path, entry)
                                  for name, (path, entry) in specs.items()})
