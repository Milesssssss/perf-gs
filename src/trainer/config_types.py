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
    NONE = "none"
    SIMPLE = "simple"
    STRUCTURE_AWARE = "structure_aware"

    @classmethod
    def from_args(cls, args) -> "DensificationMode":
        """Resolve the CLI ``--densify-mode`` choice into the typed mode. The
        single boundary that reads the raw ``args.densify_mode`` string, so the
        enum (not a scatter of booleans) owns which strategy a run uses."""
        return cls(args.densify_mode)


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
    densify_mode: DensificationMode
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
            densify_mode=DensificationMode.from_args(args),
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


class ConfigError(ValueError):
    """Raised for an invalid CLI/config combination, BEFORE any GPU allocation.

    Surfacing these as one readable error at startup replaces opaque downstream
    failures: a ``ZeroDivisionError`` deep in the schedule from a zero modulo
    interval, an empty train split that silently trains on nothing, or a densify
    window that can never open.
    """


def validate_training_config(args, *, n_cams: int, n_train: int,
                             n_test: int) -> DensificationMode:
    """Validate the resolved configuration and return the densification mode.

    Called once at startup — AFTER the scene and holdout split are known but
    BEFORE the device/pipeline allocate GPU memory — so an invalid run fails fast
    with a clear message instead of crashing or silently misbehaving mid-run.
    Only combinations that are genuinely fatal or silently no-op are rejected;
    the CLI defaults always pass.
    """
    mode = DensificationMode.from_args(args)

    if args.iters < 1:
        raise ConfigError(f"--iters must be >= 1, got {args.iters}.")
    if args.max_splats < 1:
        raise ConfigError(f"--max-splats must be >= 1, got {args.max_splats}.")

    # densify_every divides the loop schedule ((it - from) % every), so 0 is a
    # ZeroDivisionError, not a "disable" value — use --densify-mode none instead.
    if args.densify_every < 1:
        raise ConfigError(
            f"--densify-every must be >= 1 (it divides the densify schedule); "
            f"got {args.densify_every}. Use --densify-mode none to disable "
            f"densification.")
    if args.densify_from < 0:
        raise ConfigError(f"--densify-from must be >= 0, got {args.densify_from}.")
    if args.densify_from > args.densify_to:
        raise ConfigError(
            f"--densify-from ({args.densify_from}) must be <= --densify-to "
            f"({args.densify_to}); otherwise the densify window never opens.")

    # These cadence knobs are modulo divisors only on the structure-aware path.
    if mode == DensificationMode.STRUCTURE_AWARE:
        if args.sad_structure_every < 1:
            raise ConfigError(
                f"--sad-structure-every must be >= 1 for --densify-mode "
                f"structure_aware, got {args.sad_structure_every}.")
        if args.sad_accum_every < 1:
            raise ConfigError(
                f"--sad-accum-every must be >= 1 for --densify-mode "
                f"structure_aware, got {args.sad_accum_every}.")

    # The camera split must leave something to train on (and, when the holdout
    # protocol is on, something to evaluate).
    stride = max(args.eval_holdout_stride, 1)
    if n_train < 1:
        raise ConfigError(
            f"empty training split: {n_cams} cameras with holdout stride "
            f"{stride} left 0 train views.")
    if args.eval_holdout and n_test < 1:
        raise ConfigError(
            f"--eval is on but the holdout split left 0 test views "
            f"({n_cams} cameras, stride {stride}).")

    return mode
