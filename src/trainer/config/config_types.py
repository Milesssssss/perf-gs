# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The enumerated choices shared by the CLI, configs, and shaders."""

from __future__ import annotations

from enum import Enum


class SamplingMode(str, Enum):
    """Per-iteration train-camera draw (CLI ``--sample-mode``)."""
    OURS = "ours"
    SADGS = "sadgs"


class DensificationMode(str, Enum):
    """Which densification strategy the loop runs on a densify tick."""
    NONE = "none"
    SIMPLE = "simple"
    STRUCTURE_AWARE = "structure_aware"


class PrimitiveKind(str, Enum):
    """Blendable differentiable primitive family; only GAUSSIAN_3D is implemented."""
    GAUSSIAN_3D = "gaussian_3d"
    DISK_2D = "disk_2d"
    SPHERE_3D = "sphere_3d"


class BlendKind(str, Enum):
    """Compositing law applied in the blend loop."""
    ALPHA_OVER = "alpha_over"
    HARDENED_EXTINCTION = "hardened_extinction"


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


def resolve_blend_kind(*, extinction_hardness: float) -> BlendKind:
    """Hardened extinction when ``k != 1``, else alpha-over; names the law, k stays a uniform."""
    return (BlendKind.HARDENED_EXTINCTION if extinction_hardness != 1.0
            else BlendKind.ALPHA_OVER)
