"""Cross-backend parity: the same deterministic scene should render to the
same pixels on Vulkan and D3D12 within a small tolerance. Tolerance
accounts for driver-level floating-point ordering differences.
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from trainer.dataset import simple_camera
from trainer.device import create_device
from trainer.pipeline import Pipeline


def _render_scene(backend: str) -> np.ndarray:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    W, H = 128, 128
    pipeline = Pipeline(device=device, width=W, height=H, max_splats=32)
    pipeline.set_gaussians_from_numpy(
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
