"""Focused contracts for local training-application configuration objects."""

from __future__ import annotations

from argparse import Namespace

from trainer.config import (
    CompositorConfig,
    LossConfig,
    RasterConfig,
    RegularizationConfig,
)
from trainer.frontend.app import (
    ShaderSpecialization,
    TrainingLoop,
    create_pipeline,
    create_training_device,
)


def test_default_shader_specialization_uses_shader_default() -> None:
    specialization = ShaderSpecialization(backward_pixel_stride=1)
    assert specialization.compiler_defines() is None


def test_strided_backward_specialization_emits_slang_uint_define() -> None:
    specialization = ShaderSpecialization(backward_pixel_stride=4)
    assert specialization.compiler_defines() == {
        "BACKWARD_PIXEL_STRIDE": "4u",
    }


def test_device_factory_passes_resolved_specialization(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_create_device(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("trainer.frontend.app.create_device", fake_create_device)
    args = Namespace(
        backend="vulkan",
        profile_run=False,
        backward_pixel_stride=2,
    )

    assert create_training_device(args) is sentinel
    assert captured == {
        "backend": "vulkan",
        "shader_debug_info": False,
        "defines": {"BACKWARD_PIXEL_STRIDE": "2u"},
    }


def test_pipeline_factory_separates_configuration_ownership(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    def fake_pipeline(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("trainer.frontend.app.Pipeline", fake_pipeline)
    args = Namespace(
        max_splats=123,
        z_near_cull=0.25,
        cluster_cull=False,
        mip_splatting=True,
        mip_kernel_size=0.1,
        extinction_hardness=2.0,
        w_alpha=0.4,
        opacity_reg=0.03,
        scale_reg=0.07,
    )
    scene = Namespace(width=64, height=32)

    result = create_pipeline(
        args=args,
        device="device",
        gpu_lock="lock",
        scene=scene,
        lambda_l2=1.5,
        lambda_ssim=0.2,
    )

    assert result is sentinel
    assert captured["raster_config"] == RasterConfig(
        z_near_cull=0.25,
        cluster_cull=False,
        mip_enabled=True,
        kernel_size=0.1,
    )
    assert captured["compositor_config"] == CompositorConfig(
        extinction_hardness=2.0)
    assert captured["loss_config"] == LossConfig(
        alpha_weight=0.4, ssim_weight=0.2, l2_weight=1.5)
    assert captured["regularization_config"] == RegularizationConfig(
        opacity_weight=0.03, scale_weight=0.07)


def test_training_loop_presents_after_full_iteration() -> None:
    events = []
    loop = object.__new__(TrainingLoop)
    states = iter((True, False))
    context = object()

    loop._begin = lambda: events.append("begin")
    loop._should_continue = lambda: next(states)
    loop._next_iteration = lambda: events.append("next") or context
    loop._prepare_iteration = lambda ctx: events.append("prepare")
    loop._capture_if_requested = lambda ctx: events.append("capture")
    loop._execute_training_step = lambda ctx: events.append("train")
    loop._apply_mutations = lambda ctx: events.append("mutations")
    loop._produce_observations = lambda ctx: events.append("observations") or True
    loop._present_profile_frame_if_enabled = lambda: events.append("present")
    loop._complete = lambda: events.append("complete") or "result"

    assert TrainingLoop.run(loop) == "result"
    assert events == [
        "begin", "next", "prepare", "capture", "train",
        "mutations", "observations", "present", "complete",
    ]
