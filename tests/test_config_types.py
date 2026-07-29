"""Characterization + contract tests for trainer.config_types (frontend slice 2).

Freezes the enum values and the resolution of the confusing CLI boolean flags
(the three densify toggles, sampling mode, blend law) into the typed surface,
and that ScheduleConfig.from_args maps the parsed defaults correctly. These pin
the typed layer against build_parser() so later slices can migrate consumers off
the raw Namespace without changing behavior.
"""

from __future__ import annotations

from v2_train import build_parser

from trainer.config.config_types import (
    BlendKind,
    DensificationMode,
    PrimitiveKind,
    SamplingMode,
    ScheduleConfig,
    resolve_blend_kind,
)


def test_enum_values_frozen() -> None:
    assert SamplingMode.OURS == "ours"
    assert SamplingMode.SADGS == "sadgs"
    assert DensificationMode.NONE == "none"
    assert DensificationMode.SIMPLE == "simple"
    assert DensificationMode.STRUCTURE_AWARE == "structure_aware"
    assert PrimitiveKind.GAUSSIAN_3D == "gaussian_3d"
    assert PrimitiveKind.DISK_2D == "disk_2d"
    assert PrimitiveKind.SPHERE_3D == "sphere_3d"
    assert BlendKind.ALPHA_OVER == "alpha_over"
    assert BlendKind.HARDENED_EXTINCTION == "hardened_extinction"


def test_densify_mode_from_args_reads_cli_choice() -> None:
    for choice in ("none", "simple", "structure_aware"):
        args = build_parser().parse_args(["--densify-mode", choice])
        assert DensificationMode.from_args(args) == DensificationMode(choice)


def test_densify_mode_from_cli_default_is_structure_aware() -> None:
    args = build_parser().parse_args([])
    assert DensificationMode.from_args(args) == DensificationMode.STRUCTURE_AWARE


def test_sampling_mode_from_cli_default_is_sadgs() -> None:
    args = build_parser().parse_args([])
    assert SamplingMode(args.sample_mode) == SamplingMode.SADGS


def test_blend_kind_from_extinction_target() -> None:
    assert resolve_blend_kind(extinction_hardness=1.0) == BlendKind.ALPHA_OVER
    assert resolve_blend_kind(extinction_hardness=4.0) == BlendKind.HARDENED_EXTINCTION


def test_schedule_config_from_args_defaults() -> None:
    args = build_parser().parse_args([])
    sc = ScheduleConfig.from_args(args)
    assert sc.iterations == 3000
    assert (sc.densify_from, sc.densify_to, sc.densify_every) == (100, 1900, 100)
    assert sc.reset_every == 1200
    assert sc.densify_mode == DensificationMode.STRUCTURE_AWARE
    assert sc.sad_structure_every == 500
    assert sc.sad_accum_every == 10
    assert sc.preview_every == 3000
    assert sc.eval_every == 3000
    assert sc.mip_enabled is True
    assert sc.psnr_curve_enabled is False
    assert sc.psnr_curve_every == 0
    assert sc.morton_every == 0
    assert sc.stream_every == 10
