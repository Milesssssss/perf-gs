# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Cross-backend parity: the same deterministic scene should render to the
same pixels on Vulkan and D3D12 within a small tolerance. Tolerance
accounts for driver-level floating-point ordering differences.
"""

from __future__ import annotations

import numpy as np
import pytest
from _pipeline_testkit import TestPipeline

from trainer.backend.compute.device import create_device
from trainer.data.dataset import simple_camera


def _render_scene(backend: str) -> np.ndarray:
    try:
        device = create_device(backend=backend)
    except RuntimeError as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    W, H = 128, 128
    pipeline = TestPipeline(device=device, width=W, height=H, max_splats=32)
    pipeline.set_primitive_from_numpy(
        means=np.array([
            [0.0, 0.0, 3.0],
            [0.3, 0.0, 2.5],
            [-0.3, 0.1, 2.0],
        ], dtype=np.float32),
        log_scales=np.log(np.array([
            [0.2, 0.2, 0.2],
            [0.15, 0.15, 0.15],
            [0.1, 0.1, 0.1],
        ], dtype=np.float32)),
        rot_quats=np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ], dtype=np.float32),
        opacities=np.array([0.9, 0.7, 0.8], dtype=np.float32),
        color_logits=np.array([
            [5.0, -5.0, -5.0],
            [-5.0, 5.0, -5.0],
            [-5.0, -5.0, 5.0],
        ], dtype=np.float32),
    )
    camera = simple_camera(
        width=W, height=H,
        eye=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        fovy_degrees=60.0,
    )
    pipeline.render(camera)
    return pipeline.read_output_image()


def test_vulkan_d3d12_parity() -> None:
    vk = _render_scene("vulkan")
    dx = _render_scene("d3d12")
    np.testing.assert_allclose(vk, dx, atol=1e-3, rtol=1e-3)


def _render_dense(backend: str) -> np.ndarray:
    """Render a multi-bucket scene (>32 splats per tile)."""
    try:
        device = create_device(backend=backend)
    except RuntimeError as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    W, H = 128, 128
    n = 400
    rng = np.random.default_rng(0)
    # Overlapping splats clustered near the optical axis so many land in the
    # same tiles -> forces deep per-pixel composite chains (multiple buckets).
    means = np.column_stack([
        rng.normal(0.0, 0.25, n),
        rng.normal(0.0, 0.25, n),
        rng.uniform(1.5, 4.0, n),
    ]).astype(np.float32)
    log_scales = np.log(rng.uniform(0.05, 0.2, size=(n, 3)).astype(np.float32))
    rot = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    opacities = rng.uniform(0.2, 0.9, n).astype(np.float32)
    colors = rng.normal(0.0, 3.0, size=(n, 3)).astype(np.float32)

    pipeline = TestPipeline(device=device, width=W, height=H, max_splats=n)
    pipeline.set_primitive_from_numpy(
        means=means, log_scales=log_scales, rot_quats=rot,
        opacities=opacities, color_logits=colors,
    )
    camera = simple_camera(
        width=W, height=H,
        eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
        fovy_degrees=60.0,
    )
    pipeline.render(camera)
    return pipeline.read_output_image()


def test_dense_vulkan_d3d12_parity() -> None:
    vk = _render_dense("vulkan")
    dx = _render_dense("d3d12")
    assert np.any(vk[..., 3] > 0.9)
    np.testing.assert_allclose(vk, dx, atol=2e-3, rtol=0.0)
