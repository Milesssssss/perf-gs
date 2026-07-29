"""Golden test for the SSIM loss gradient (losses/ssim_loss.slang).

The SSIM loss path had NO coverage before this file — test_gradients and
test_backward_persplat both set lambda_ssim=0. This pins the two-pass SSIM
gradient (ssimComputeMaps + ssimGradient) against an independent NumPy
reference that implements the SAME math with a naive 2D Gaussian convolution.

Purpose: lock the gradient so the separable (H-then-V) rewrite of the two
121-tap window loops can be proven to produce the identical result. The
separable form only reassociates the (exact, per-axis-clamped) window sum, so
agreement should be float-tight (~1e-4 rel), not merely approximate.
"""
from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.config.config import LossConfig
from trainer.data.dataset import simple_camera
from trainer.backend.compute.device import create_device


BACKENDS = ["vulkan", "d3d12"]

WINDOW_RADIUS = 5
SIGMA = 1.5
C1 = 0.0001
C2 = 0.0009
LAMBDA_SSIM = 0.2
LAMBDA_L2 = 0.0
W_ALPHA = 0.0


def _gauss_1d() -> np.ndarray:
    d = np.arange(-WINDOW_RADIUS, WINDOW_RADIUS + 1, dtype=np.float64)
    return np.exp(-0.5 * (d * d) / (SIGMA * SIGMA))


def _clamp_conv2d(img: np.ndarray, k2d: np.ndarray) -> np.ndarray:
    """2D convolution (correlation) with per-axis edge CLAMP, matching the
    shader's clamp(q, 0, dim-1). img: (H,W,C). Naive 2D — the reference."""
    H, W, C = img.shape
    R = WINDOW_RADIUS
    out = np.zeros_like(img)
    for di in range(-R, R + 1):
        for dj in range(-R, R + 1):
            w = k2d[di + R, dj + R]
            ys = np.clip(np.arange(H) + di, 0, H - 1)
            xs = np.clip(np.arange(W) + dj, 0, W - 1)
            shifted = img[ys][:, xs]
            out += w * shifted
    return out


def _ssim_grad_reference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NumPy reference for dL_dC of the (1 - SSIM) term, sum-over-pixels
    convention. Mirrors ssim_loss.slang exactly (naive 2D window)."""
    g1 = _gauss_1d()
    k2d = np.outer(g1, g1)                       # unnormalized 2D Gaussian
    w_sum = k2d.sum()
    inv_w = 1.0 / w_sum

    mu_x = _clamp_conv2d(x, k2d) * inv_w
    mu_y = _clamp_conv2d(y, k2d) * inv_w
    Ex2 = _clamp_conv2d(x * x, k2d) * inv_w
    Ey2 = _clamp_conv2d(y * y, k2d) * inv_w
    Exy = _clamp_conv2d(x * y, k2d) * inv_w

    sigma_x2 = Ex2 - mu_x * mu_x
    sigma_y2 = Ey2 - mu_y * mu_y
    sigma_xy = Exy - mu_x * mu_y

    A = 2.0 * mu_x * mu_y + C1
    B = 2.0 * sigma_xy + C2
    Cc = mu_x * mu_x + mu_y * mu_y + C1
    D = sigma_x2 + sigma_y2 + C2

    dS_dmux = 2.0 * B * (mu_y * Cc - mu_x * A) / (Cc * Cc * D)
    dS_dsx2 = -A * B / (Cc * D * D)
    dS_dsxy = 2.0 * A / (Cc * D)

    m1 = (dS_dmux - 2.0 * mu_x * dS_dsx2 - mu_y * dS_dsxy) * inv_w
    m2 = (2.0 * dS_dsx2) * inv_w
    m3 = (dS_dsxy) * inv_w

    cm1 = _clamp_conv2d(m1, k2d)
    cm2 = _clamp_conv2d(m2, k2d)
    cm3 = _clamp_conv2d(m3, k2d)

    # loss = (1 - SSIM), so gradient negates the SSIM gradient.
    grad_ssim = -(cm1 + x * cm2 + y * cm3)
    grad_l1 = np.sign(x - y)
    grad = (1.0 - LAMBDA_SSIM) * grad_l1 + LAMBDA_SSIM * grad_ssim
    if LAMBDA_L2 > 0.0:
        grad += LAMBDA_L2 * 2.0 * (x - y)
    return grad


def _scene():
    means = np.array([
        [0.0, 0.0, 3.0], [0.2, 0.1, 2.5], [-0.15, -0.1, 2.0],
    ], dtype=np.float32)
    log_scales = np.log(np.array([
        [0.3, 0.3, 0.3], [0.2, 0.2, 0.2], [0.25, 0.25, 0.25],
    ], dtype=np.float32))
    rot_quats = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.9659, 0.0, 0.2588, 0.0],
        [0.9239, 0.0, 0.0, 0.3827],
    ], dtype=np.float32)
    opacities = np.array([0.8, 0.6, 0.7], dtype=np.float32)
    color_logits = np.array([
        [1.5, -1.5, -1.5], [-1.5, 1.5, -1.5], [-1.5, -1.5, 1.5],
    ], dtype=np.float32)
    return dict(means=means, log_scales=log_scales, rot_quats=rot_quats,
                opacities=opacities, color_logits=color_logits)


def _target(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= max(h - 1, 1)
    xx /= max(w - 1, 1)
    return np.stack([xx, yy, (1.0 - xx) * yy], axis=-1).astype(np.float32)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ssim_grad_matches_reference(backend: str):
    W = H = 24
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    pipeline = TestPipeline(
        device=device, width=W, height=H, max_splats=16,
        loss_config=LossConfig(
            ssim_weight=LAMBDA_SSIM,
            alpha_weight=W_ALPHA,
            l2_weight=LAMBDA_L2,
        ),
    )
    pipeline.set_gaussians_from_numpy(**_scene())
    target = _target(H, W)
    pipeline.set_target_image(target)
    camera = simple_camera(
        width=W, height=H, eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0), fovy_degrees=60.0,
    )

    pipeline.forward_and_loss(camera)
    device.wait_for_idle()

    # Rendered image the shader saw (sRGB in .rgb).
    x = pipeline.buffers.output_image.to_numpy()[..., :3].astype(np.float64)
    y = target.astype(np.float64)

    dL_dC_gpu = pipeline.buffers.dL_dC_texture.to_numpy()[..., :3].astype(np.float64)
    dL_dC_ref = _ssim_grad_reference(x, y)

    # SSIM stats blow up near flat regions (D -> C2), so use a rel tolerance
    # with an absolute floor. Float32 shader vs float64 reference.
    np.testing.assert_allclose(dL_dC_gpu, dL_dC_ref, rtol=2e-3, atol=2e-4)
