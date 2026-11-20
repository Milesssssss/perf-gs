# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The enumerated choices shared by the CLI, configs, and shaders."""

from __future__ import annotations

from enum import Enum


class DensificationMode(str, Enum):
    """Which densification strategy the loop runs on a densify tick."""
    NONE = "none"
    SIMPLE = "simple"
    STRUCTURE_AWARE = "structure_aware"


class SplatKernel(str, Enum):
    """How a splat's projected footprint becomes alpha (CLI ``--splat-kernel``).
    Declaration order is the shader's SPLAT_KERNEL index (primitives/gaussian/eval.slang)."""
    GAUSSIAN = "gaussian"
    DENSITY = "density"
    LOG_DENSITY = "log_density"
    OPAQUE_CORE = "opaque_core"


class LearningRateRecipe(str, Enum):
    """Per-parameter learning-rate plan (CLI ``--lr-recipe``)."""
    DEFAULT = "default"
    QUALITY = "quality"
    SHORT = "short"
