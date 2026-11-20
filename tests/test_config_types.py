# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The shared enums, and how RunConfig resolves the CLI's mode and schedule defaults."""

from __future__ import annotations

from trainer.config.config_types import DensificationMode
from trainer.config.resolve import RunConfig
from trainer.frontend.cli import build_parser


def test_densification_mode_values() -> None:
    assert DensificationMode.NONE == "none"
    assert DensificationMode.SIMPLE == "simple"
    assert DensificationMode.STRUCTURE_AWARE == "structure_aware"


def test_densify_mode_from_args_reads_cli_choice() -> None:
    for choice in ("none", "simple", "structure_aware"):
        args = build_parser().parse_args(["--densify-mode", choice])
        assert RunConfig.from_args(args, 1.0).densify_mode == DensificationMode(choice)


def test_densify_mode_from_cli_default_is_structure_aware() -> None:
    args = build_parser().parse_args([])
    assert RunConfig.from_args(args, 1.0).densify_mode == DensificationMode.STRUCTURE_AWARE


def test_sampling_mode_from_cli_default_is_sadgs() -> None:
    args = build_parser().parse_args([])
    assert args.sample_mode == "sadgs"


def test_schedule_config_from_args_defaults() -> None:
    args = build_parser().parse_args([])
    sc = RunConfig.from_args(args, 1.0).schedule
    assert sc.iterations == 3000
    assert (sc.densify_from, sc.densify_to, sc.densify_every) == (100, 1900, 100)
    assert sc.reset_every == 1200
    assert sc.densify_mode == DensificationMode.STRUCTURE_AWARE
    assert sc.sad_structure_every == 500
    assert sc.sad_accum_every == 10
    assert sc.preview_every == 0
    assert sc.eval_every == 0
    assert sc.mip_enabled is True
    assert sc.psnr_curve_enabled is False
    assert sc.psnr_curve_every == 0
    assert sc.morton_every == 0
    assert sc.stream_every == 10
