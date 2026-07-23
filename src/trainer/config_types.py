"""Typed training configuration: enums + config dataclasses for the frontend.

Introduced by the frontend refactor to stop threading an ``argparse.Namespace``
through the trainer. This module is ADDITIVE — the legacy ``build_parser()``
surface is unchanged. These types resolve parsed args into explicit, typed
values that the extracted training components (schedule, resources, loop) will
consume. The ``from_args`` classmethods are the single boundary that reads the
Namespace, so no Namespace escapes into code that depends on these types.

Enums use ``(str, Enum)`` (not 3.11's ``StrEnum``) to keep the declared
``requires-python >= 3.10`` support; ``==`` comparison against the raw CLI
string and ``EnumClass(value)`` construction behave identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SamplingMode(str, Enum):
    """Per-iteration train-camera draw (CLI ``--sample-mode``)."""
    OURS = "ours"
    SADGS = "sadgs"


class DensificationMode(str, Enum):
    """Which densification strategy the loop runs on a densify tick."""
    SIMPLE = "simple"
    IMPORTANCE = "importance_score"
    STRUCTURE_AWARE = "structure_aware"


class PrimitiveKind(str, Enum):
    """Blendable differentiable primitive family. GAUSSIAN_3D is today's only
    implemented specialization; the others are the extensibility target."""
    GAUSSIAN_3D = "gaussian_3d"
    DISK_2D = "disk_2d"
    SPHERE_3D = "sphere_3d"


class BlendKind(str, Enum):
    """Compositing law applied in the blend loop."""
    ALPHA_OVER = "alpha_over"
    HARDENED_EXTINCTION = "hardened_extinction"


def resolve_densification_mode(*, structure_densify: bool, simple_densify: bool,
                               importance_score_densify: bool) -> DensificationMode:
    """Collapse the three overlapping densify booleans into one mode, matching
    the CURRENT loop precedence: structure-aware overrides simple overrides
    importance (see ``--structure-densify`` help "Overrides --simple-densify"
    and the elif chain in the training loop). The CLI default
    ``structure_densify=True`` therefore resolves to STRUCTURE_AWARE.

    When no flag is set the loop runs no densify branch (an empty tick); this is
    unreachable from the defaults, and is surfaced here as SIMPLE for the typed
    API rather than a fourth "none" state.
    """
    if structure_densify:
        return DensificationMode.STRUCTURE_AWARE
    if simple_densify:
        return DensificationMode.SIMPLE
    if importance_score_densify:
        return DensificationMode.IMPORTANCE
    return DensificationMode.SIMPLE


def resolve_blend_kind(*, extinction_hardness: float) -> BlendKind:
    """The blend law named by the current render config: hardened extinction when
    the target ``k != 1``, else standard alpha-over. (k is still applied as a
    shader uniform; this only NAMES the law for the specialization layer.)"""
    return (BlendKind.HARDENED_EXTINCTION if extinction_hardness != 1.0
            else BlendKind.ALPHA_OVER)


@dataclass(frozen=True)
class ScheduleConfig:
    """Immutable per-iteration schedule knobs — the foundation the pure
    ``TrainingSchedule`` (next slice) will consume. Mirrors the CLI schedule
    flags 1:1 and carries NO logic itself."""
    iterations: int
    densify_from: int
    densify_to: int
    densify_every: int
    reset_every: int
    structure_densify: bool
    sad_structure_every: int
    sad_accum_every: int
    preview_every: int
    log_every: int
    render_all_every: int
    eval_every: int
    psnr_curve_enabled: bool
    psnr_curve_every: int
    morton_every: int
    stream_every: int
    mip_enabled: bool

    @classmethod
    def from_args(cls, args) -> "ScheduleConfig":
        return cls(
            iterations=args.iters,
            densify_from=args.densify_from,
            densify_to=args.densify_to,
            densify_every=args.densify_every,
            reset_every=args.reset_every,
            structure_densify=bool(args.structure_densify),
            sad_structure_every=args.sad_structure_every,
            sad_accum_every=args.sad_accum_every,
            preview_every=args.preview_every,
            log_every=args.log_every,
            render_all_every=args.render_all_every,
            eval_every=args.eval_every,
            psnr_curve_enabled=bool(args.psnr_curve),
            psnr_curve_every=args.psnr_curve_every,
            morton_every=args.morton_every,
            stream_every=args.stream_every,
            mip_enabled=bool(args.mip_splatting),
        )
