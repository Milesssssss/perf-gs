# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Fail-fast checks of the parsed arguments, before any GPU allocation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trainer.config.config_types import DensificationMode

if TYPE_CHECKING:
    import argparse


class ConfigError(ValueError):
    """An invalid CLI/config combination, reported before anything is allocated."""


def validate_training_config(args: argparse.Namespace, *, n_cams: int, n_train: int,
                             n_test: int) -> None:
    """Reject combinations that would crash mid-run or silently do nothing.

    Runs once the scene and holdout split are known; the CLI defaults always pass.
    """
    mode = DensificationMode(args.densify_mode)

    if args.iters < 1:
        raise ConfigError(f"--iters must be >= 1, got {args.iters}.")
    if args.lattice_stride < 1 or args.lattice_until < 0:
        raise ConfigError("Lattice stride must be positive and its last training step must be nonnegative.")
    if args.max_splats < 1:
        raise ConfigError(f"--max-splats must be >= 1, got {args.max_splats}.")
    # The radix sort dispatches one group per 2,048 keys along a single GPU dimension.
    max_pairs = 65_535 * 2_048
    if args.max_sort_pairs is not None and not 1 <= args.max_sort_pairs <= max_pairs:
        raise ConfigError(f"--max-sort-pairs must be between 1 and {max_pairs:,}.")

    # densify_every is a modulo divisor of the schedule, not a "disable" value.
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
    profile_power = float(args.splat_profile_power)
    if profile_power < 1.0:
        raise ConfigError(
            f"--splat-profile-power must be >= 1, got {profile_power}: below 1 "
            f"the falloff has a cusp at every splat's center.")
    blend_weight = float(args.prune_blend_weight)
    if blend_weight < 0.0:
        raise ConfigError(
            f"--prune-blend-weight must be >= 0 (0 = off), got {blend_weight}.")

    # Modulo divisors only on the structure-aware path.
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

    stbn_fraction = float(args.stbn_pixel_fraction)
    if stbn_fraction <= 0.0 or stbn_fraction > 1.0:
        raise ConfigError(
            f"--stbn-pixel-fraction must be in (0, 1], got {stbn_fraction}. "
            f"1.0 = supervise every pixel (the feature off).")
    if stbn_fraction < 1.0:
        if int(args.backward_pixel_stride) > 1:
            raise ConfigError(
                f"--stbn-pixel-fraction {stbn_fraction} and "
                f"--backward-pixel-stride {args.backward_pixel_stride} both "
                f"subsample the backward's pixels; use one or the other.")
        if args.stbn_source == "file" and not args.stbn_file:
            raise ConfigError("--stbn-source file requires --stbn-file <path.npy>.")

    stride = max(args.eval_holdout_stride, 1)
    if n_train < 1:
        raise ConfigError(
            f"empty training split: {n_cams} cameras with holdout stride "
            f"{stride} left 0 train views.")
    if args.eval_holdout and n_test < 1:
        raise ConfigError(
            f"--eval is on but the holdout split left 0 test views "
            f"({n_cams} cameras, stride {stride}).")
