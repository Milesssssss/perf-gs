# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

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
    # Explicit extra eval checkpoints, UNION-ed with the eval_every cadence. A
    # convergence study wants an irregular ladder (250, 500, 750, 1000, 1500,
    # 2000, ...) that no single interval expresses; expressing it as
    # --eval-every 250 would buy four sweeps nobody asked for, and each sweep
    # renders and scores the whole held-out set. Empty tuple = old behavior.
    eval_iters: tuple[int, ...] = ()

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
            eval_iters=tuple(sorted({
                int(v) for v in (getattr(args, "eval_iters", None) or ())})),
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
        if args.sad_max_split_factor < 1:
            raise ConfigError(
                f"--sad-max-split-factor must be >= 1, got "
                f"{args.sad_max_split_factor}.")

    # STBN sparse pixel supervision.
    stbn_fraction = float(getattr(args, "stbn_pixel_fraction", 1.0))
    if stbn_fraction <= 0.0 or stbn_fraction > 1.0:
        raise ConfigError(
            f"--stbn-pixel-fraction must be in (0, 1], got {stbn_fraction}. "
            f"1.0 = supervise every pixel (the feature off).")
    if stbn_fraction < 1.0:
        if int(getattr(args, "backward_pixel_stride", 1)) > 1:
            raise ConfigError(
                f"--stbn-pixel-fraction {stbn_fraction} and "
                f"--backward-pixel-stride {args.backward_pixel_stride} both "
                f"subsample the backward's pixels; use one or the other.")
        source = str(getattr(args, "stbn_source", "nvidia"))
        if source == "file" and not str(getattr(args, "stbn_file", "")):
            raise ConfigError("--stbn-source file requires --stbn-file <path.npy>.")

    # Importance-guided (adaptive) sparse sampling.
    if bool(getattr(args, "stbn_adaptive", False)):
        if stbn_fraction >= 1.0:
            raise ConfigError(
                "--stbn-adaptive needs a sparse budget: pass "
                "--stbn-pixel-fraction < 1. At 1.0 every pixel is supervised, so "
                "there is no allocation to make.")
        alpha = float(getattr(args, "stbn_importance_alpha", 0.75))
        if not 0.0 <= alpha < 1.0:
            raise ConfigError(
                f"--stbn-importance-alpha must be in [0, 1), got {alpha}. "
                f"alpha=1 admits p_i=0 pixels, whose 1/p_i weight is undefined.")
        if float(getattr(args, "stbn_importance_sigma", 8.0)) < 0.0:
            raise ConfigError(
                f"--stbn-importance-sigma must be >= 0, got "
                f"{args.stbn_importance_sigma}.")
        if int(getattr(args, "stbn_importance_downsample", 4)) < 1:
            raise ConfigError(
                f"--stbn-importance-downsample must be >= 1, got "
                f"{args.stbn_importance_downsample}.")
        if int(getattr(args, "stbn_warmup_iters", 0)) > 0:
            # The adaptive path rebuilds its table every iteration from a runtime
            # rho, so a ramp is representable — but the budget the solver enforces
            # would then differ from the one the comparison is matched on, and no
            # experiment here needs it. Rejected rather than left as a silent
            # confound.
            raise ConfigError(
                "--stbn-adaptive is incompatible with --stbn-warmup-iters > 0: "
                "the density comparison against the uniform regimes is matched on "
                "a FLAT rho, and a ramp would make the two arms' realized budgets "
                "differ. Use --stbn-warmup-iters 0.")
        uniform = str(getattr(args, "stbn_adaptive_uniform", "blue_temporal"))
        if uniform != "iid" and str(getattr(args, "stbn_source", "nvidia")) == "white":
            raise ConfigError(
                f"--stbn-adaptive-uniform {uniform} draws its uniform from a "
                f"blue-noise volume, but --stbn-source white would hand it white "
                f"noise. Use --stbn-adaptive-uniform iid for the independent "
                f"control, or --stbn-source nvidia/file for the blue ones.")
        # The slice traversal is driven by --stbn-independent (shared with the
        # uniform regimes), so the two knobs have to agree or the run would claim
        # one temporal semantics and walk the other.
        independent = bool(getattr(args, "stbn_independent", False))
        if uniform == "blue_independent" and not independent:
            raise ConfigError(
                "--stbn-adaptive-uniform blue_independent needs "
                "--stbn-independent (a fresh slice per step); without it the "
                "slice would walk each camera's visit counter, which is the "
                "blue_temporal semantics.")
        if uniform == "blue_temporal" and independent:
            raise ConfigError(
                "--stbn-adaptive-uniform blue_temporal walks each camera's visit "
                "counter, which --stbn-independent overrides. Drop "
                "--stbn-independent, or ask for blue_independent.")
        if float(getattr(args, "stbn_max_fraction", 0.0) or 0.0) > 0.0:
            # STBN_MAX_TILE_PIXELS caps the backward's compacted visit list. A
            # nonuniform marginal lets a hot tile select far more than the nominal
            # k, so a shrunken cap would silently TRUNCATE those tiles — dropping
            # gradient the loss had already weighted in, i.e. a biased estimator
            # with no error anywhere.
            raise ConfigError(
                "--stbn-adaptive is incompatible with --stbn-max-fraction: the "
                "adaptive marginal lets a tile's selected count exceed the "
                "nominal budget, and a shrunken STBN_MAX_TILE_PIXELS would "
                "truncate the backward's visit list for exactly the tiles the "
                "importance map targeted.")

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
