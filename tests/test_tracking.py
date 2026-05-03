"""v2 tracking-accumulator tests.

Verify that the backward pass populates the per-gaussian |grad_mean_2d|
magnitude and pixel-coverage accumulators in a sensible way:
- gaussians that contributed to visible pixels get non-zero tracking
- gaussians entirely outside the frustum get zero
- clear_tracking() zeros both buffers

The numerical magnitudes are not asserted — that's what the
finite-difference gradient test already covers for the primary grads.
Here we only assert presence/absence and that ``clear_tracking`` works.
"""

from __future__ import annotations

import numpy as np
import pytest

from trainer.dataset import simple_camera
from trainer.device import create_device
from trainer.pipeline import Pipeline


BACKENDS = ["vulkan", "d3d12"]


def _setup(backend: str, w: int, h: int, max_splats: int) -> Pipeline:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return Pipeline(device=device, width=w, height=h, max_splats=max_splats)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tracking_populates_and_clears(backend: str) -> None:
    W = H = 32
    pipeline = _setup(backend, W, H, max_splats=16)

    # Three gaussians:
    #  0 — at origin, in frustum, will contribute
    #  1 — off to the side but still in frustum, will contribute
    #  2 — way off-screen (view-space z negative, behind camera), will NOT contribute
    pipeline.set_gaussians_from_numpy(
        means=np.array([
            [0.0, 0.0, 2.0],
            [0.3, -0.2, 2.5],
            [0.0, 0.0, -5.0],  # behind the camera
        ], dtype=np.float32),
        log_scales=np.log(np.array([
            [0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2],
        ], dtype=np.float32)),
        rot_quats=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (3, 1)),
        opacities=np.array([0.8, 0.7, 0.7], dtype=np.float32),
        color_logits=np.array([
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ], dtype=np.float32),
    )

    # A non-trivial target so the loss gradient is non-zero.
    target = np.ones((H, W, 3), dtype=np.float32) * np.array([0.0, 1.0, 0.0])
    pipeline.set_target_image(target)

    camera = simple_camera(
        width=W, height=H,
        eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
        fovy_degrees=60.0,
    )

    # Clear + run forward/loss/backward.
    pipeline.clear_tracking()
    pipeline._clear_grads()
    pipeline.forward_and_loss(camera)
    pipeline.backward()

    t = pipeline.read_tracking()
    cov = t["pixel_coverage"]
    mag = t["grad_mean_2d_mag"]

    # The two in-frustum gaussians should have contributed.
    assert cov[0] > 0, f"expected gaussian 0 to contribute pixels, got cov={cov[0]}"
    assert cov[1] > 0, f"expected gaussian 1 to contribute pixels, got cov={cov[1]}"
    # Their grad_mean_2d magnitudes should be non-zero (we wrote a gradient).
    assert mag[0] > 0.0, f"expected gaussian 0 to have grad_mean_2d_mag > 0, got {mag[0]}"
    assert mag[1] > 0.0, f"expected gaussian 1 to have grad_mean_2d_mag > 0, got {mag[1]}"

    # The behind-camera gaussian should have contributed nothing.
    assert cov[2] == 0, f"expected behind-camera gaussian to have cov=0, got {cov[2]}"
    assert mag[2] == 0.0, f"expected behind-camera gaussian to have mag=0, got {mag[2]}"

    # clear_tracking() should reset everything.
    pipeline.clear_tracking()
    t2 = pipeline.read_tracking()
    assert np.all(t2["pixel_coverage"] == 0)
    assert np.all(t2["grad_mean_2d_mag"] == 0.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tracking_accumulates_across_iters(backend: str) -> None:
    """Two forward+backward calls (no clear between them) should double
    roughly (not exactly — gradients can change; coverage is exact)."""
    W = H = 32
    pipeline = _setup(backend, W, H, max_splats=4)

    pipeline.set_gaussians_from_numpy(
        means=np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
        log_scales=np.log(np.array([[0.2, 0.2, 0.2]], dtype=np.float32)),
        rot_quats=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.8], dtype=np.float32),
        color_logits=np.array([[1.0, -1.0, -1.0]], dtype=np.float32),
    )
    target = np.ones((H, W, 3), dtype=np.float32) * np.array([0.0, 1.0, 0.0])
    pipeline.set_target_image(target)
    camera = simple_camera(
        width=W, height=H,
        eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
        fovy_degrees=60.0,
    )

    pipeline.clear_tracking()
    pipeline._clear_grads()
    pipeline.forward_and_loss(camera)
    pipeline.backward()
    t1 = pipeline.read_tracking()
    cov1 = int(t1["pixel_coverage"][0])

    # Second pass WITHOUT clearing tracking.
    pipeline._clear_grads()
    pipeline.forward_and_loss(camera)
    pipeline.backward()
    t2 = pipeline.read_tracking()
    cov2 = int(t2["pixel_coverage"][0])

    # Coverage is deterministic (same scene, same camera): exactly 2x.
    assert cov2 == 2 * cov1, f"expected coverage to double, got {cov1} -> {cov2}"
