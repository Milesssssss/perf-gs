"""Smoke test for Pipeline.train_step after the GpuRuntime/CommandBatch extraction.

The GPU parity subset drives the forward/backward via TestPipeline.forward_and_loss
(immediate-dispatch + transient indirect paths) and densify (the batch-active path via
the _encoder compat property), but nothing else exercises the FULL train_step batch
lifecycle: runtime.command_batch -> marker groups -> per-stage _dispatch -> runtime.submit.
This runs one real train_step and checks it records + submits exactly one batch, clears
the active batch afterward, and updates parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.config import AdamLRs, RenderConfig
from trainer.dataset import simple_camera
from trainer.device import create_device

BACKENDS = ["vulkan", "d3d12"]


def _setup(backend: str) -> TestPipeline:
    try:
        device = create_device(backend=backend)
    except Exception as exc:  # backend unavailable on host
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    p = TestPipeline(device=device, width=16, height=16, max_splats=16,
                     cfg=RenderConfig(lambda_ssim=0.0, w_alpha=0.0))
    means = np.array([[0.0, 0.0, 3.0], [0.2, 0.1, 2.5], [-0.15, -0.1, 2.0]],
                     dtype=np.float32)
    log_scales = np.log(np.full((3, 3), 0.3, dtype=np.float32))
    rot_quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (3, 1))
    opacities = np.array([0.8, 0.6, 0.7], dtype=np.float32)
    color_logits = np.array([[1.5, -1.5, -1.5], [-1.5, 1.5, -1.5], [-1.5, -1.5, 1.5]],
                            dtype=np.float32)
    p.set_gaussians_from_numpy(means=means, log_scales=log_scales, rot_quats=rot_quats,
                               opacities=opacities, color_logits=color_logits)
    target = np.zeros((16, 16, 3), np.float32)
    target[..., 0] = 0.5   # non-matching target => non-zero gradients
    p.set_target_image(target)
    return p


@pytest.mark.parametrize("backend", BACKENDS)
def test_train_step_records_one_batch_and_updates(backend: str) -> None:
    p = _setup(backend)
    cam = simple_camera(width=16, height=16, eye=(0.0, 0.0, 0.0),
                        target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                        fovy_degrees=60.0)
    # Seed the "scalar grads are zero at start-of-iter" invariant train_step assumes.
    p._clear_grads()
    p.device.wait_for_idle()

    before = p.buffers.gaussians.to_numpy().copy()
    prev_submit = p.last_submit_id

    p.train_step(cam, lrs=AdamLRs(mean=1e-2, scale=1e-2, rot=1e-3,
                                  opacity=1e-2, color=1e-2))
    p.device.wait_for_idle()

    # A single batch was recorded + submitted (submit id advanced) and cleared.
    assert p.last_submit_id != prev_submit
    assert p._batch is None
    # Adam moved parameters (non-matching target => non-zero grads => update).
    after = p.buffers.gaussians.to_numpy()
    assert not np.array_equal(before, after)


@pytest.mark.parametrize("backend", BACKENDS)
def test_two_train_steps_advance_submit(backend: str) -> None:
    p = _setup(backend)
    cam = simple_camera(width=16, height=16, eye=(0.0, 0.0, 0.0),
                        target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                        fovy_degrees=60.0)
    p._clear_grads()
    p.train_step(cam, lrs=AdamLRs())
    first = p.last_submit_id
    p.train_step(cam, lrs=AdamLRs())
    p.device.wait_for_idle()
    assert p.last_submit_id != first
    assert p._batch is None
