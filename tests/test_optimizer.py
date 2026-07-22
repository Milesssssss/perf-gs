"""Characterization (golden) test for the Adam optimizer step (adam.slang).

No other test gates the Adam PARAMETER UPDATE — test_gradients checks the grad
BUFFERS (backward output), not the optimizer that consumes them. This pins
adamStep's exact numerics against an independent numpy reference so the
optimizer can be generalized over the primitive bit-identically.

One dispatch of k_adam over 3 gaussians with known params, zero initial Adam
moments, and known fixed-point gradients; the updated params + m/v state + the
zeroed grad accumulators are compared to a numpy Adam reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.config import RenderConfig
from trainer.device import create_device
from trainer.buffers import GRAD_SCALAR_COMPONENTS


BACKENDS = ["vulkan", "d3d12"]

# MUST match adam.slang.
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
FIXED_SCALE = 65536.0

# Packed Gaussian3D channel index -> LR group. Gradients and both moments use
# this exact channel-major order.
LR_GROUPS = [
    "mean", "mean", "mean",
    "scale", "scale", "scale",
    "rot", "rot", "rot", "rot",
    "opacity",
    "color", "color", "color",
]
ROT_SLICE = slice(6, 10)   # packed params renormalized after the step


def _adam_scalar(param, grad, lr, bc1, bc2):
    """Scalar Adam from zero moments, mirroring adam.slang::adam_scalar."""
    m = (1.0 - BETA1) * grad
    v = (1.0 - BETA2) * grad * grad
    param = param - lr * (m / bc1) / (np.sqrt(v / bc2) + EPS)
    return param, m, v


@pytest.mark.parametrize("backend", BACKENDS)
def test_adam_step_matches_reference(backend: str) -> None:
    try:
        device = create_device(backend=backend)
    except Exception as exc:  # pragma: no cover - backend availability
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    W = H = 8
    max_splats = 16
    N = 3
    p = TestPipeline(device=device, width=W, height=H, max_splats=max_splats,
                     cfg=RenderConfig(lambda_ssim=0.0, w_alpha=0.0))

    means = np.array([[0.10, 0.20, 2.50],
                      [-0.30, 0.05, 3.10],
                      [0.40, -0.15, 1.90]], dtype=np.float32)
    log_scales = np.log(np.array([[0.20, 0.25, 0.30],
                                  [0.10, 0.12, 0.14],
                                  [0.33, 0.31, 0.29]], dtype=np.float32))
    rot_quats = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.9659, 0.0, 0.2588, 0.0],
                          [0.9239, 0.0, 0.0, 0.3827]], dtype=np.float32)
    opacities = np.array([0.70, 0.55, 0.62], dtype=np.float32)
    color_logits = np.array([[0.30, -0.50, 0.80],
                             [-0.20, 0.40, -0.60],
                             [0.10, 0.10, -0.10]], dtype=np.float32)
    p.set_gaussians_from_numpy(means=means, log_scales=log_scales,
                               rot_quats=rot_quats, opacities=opacities,
                               color_logits=color_logits)

    # Original params in the PACKED layout write_gaussians produced (opacity is
    # stored as a logit, not the raw probability).
    pc = np.clip(opacities, 1e-5, 1 - 1e-5).astype(np.float64)
    op_logit = np.log(pc / (1 - pc))
    orig = np.zeros((N, 14), np.float64)
    orig[:, 0:3] = means
    orig[:, 3:6] = log_scales
    orig[:, 6:10] = rot_quats
    orig[:, 10] = op_logit
    orig[:, 11:14] = color_logits

    # Known fixed-point gradients, distinct per (component, gaussian) so a
    # mis-wiring (swapped m/v, wrong LR, wrong param slot) shows as a mismatch.
    rng = np.random.default_rng(1234)
    gradient_plane = np.zeros((14, max_splats), dtype=np.int64)
    grad_raw = {}
    for ci, name in enumerate(GRAD_SCALAR_COMPONENTS):
        vals = (0.5 + 0.13 * ci) + 0.37 * np.arange(N) + rng.uniform(-0.1, 0.1, N)
        gradient_plane[ci, :N] = np.round(vals * FIXED_SCALE).astype(np.int64)
        grad_raw[name] = gradient_plane[ci, :N].astype(np.float64) / FIXED_SCALE
    p.buffers.gradients.copy_from_numpy(np.ascontiguousarray(gradient_plane.reshape(-1)))

    lr = dict(mean=1e-3, scale=5e-3, rot=2e-3, opacity=5e-2, color=2.5e-3)
    bc1 = 1.0 - BETA1 ** 5
    bc2 = 1.0 - BETA2 ** 5
    pixels_per_step = float(W * H)
    inv_pixels = 1.0 / max(pixels_per_step, 1.0)

    p._dispatch(p.k_adam, "adam",
        thread_count=[max_splats, 1, 1],
        primitives=p.buffers.gaussians.storage,
        gradients=p.buffers.gradients.storage,
        moment1=p.buffers.moment1.storage,
        moment2=p.buffers.moment2.storage,
        num_primitives_counter=p.buffers.num_gaussians_counter.storage,
        cluster_visible=p.buffers.cluster_visible.storage,
        learning_rates={"values": [lr[group] for group in LR_GROUPS]},
        cfg={"bias_correction1": bc1, "bias_correction2": bc2},
        pixels_per_step=pixels_per_step,
        channel_stride=max_splats,
    )
    device.wait_for_idle()

    g_out = np.frombuffer(p.buffers.gaussians.to_numpy().tobytes(),
                          dtype=np.float32).reshape(-1, 14)[:N].astype(np.float64)
    m1_out = np.frombuffer(p.buffers.moment1.to_numpy().tobytes(),
                           dtype=np.float32).reshape(14, max_splats)[:, :N].T.astype(np.float64)
    m2_out = np.frombuffer(p.buffers.moment2.to_numpy().tobytes(),
                           dtype=np.float32).reshape(14, max_splats)[:, :N].T.astype(np.float64)

    # --- numpy reference ---
    new_p = orig.copy()
    new_m = np.zeros((N, 14), np.float64)
    new_v = np.zeros((N, 14), np.float64)
    for pi in range(14):
        name = GRAD_SCALAR_COMPONENTS[pi]
        lr_key = LR_GROUPS[pi]
        for gi in range(N):
            g = grad_raw[name][gi] * inv_pixels
            new_p[gi, pi], new_m[gi, pi], new_v[gi, pi] = _adam_scalar(
                orig[gi, pi], g, lr[lr_key], bc1, bc2)
    # Post-step quaternion renormalize.
    for gi in range(N):
        q = new_p[gi, ROT_SLICE]
        new_p[gi, ROT_SLICE] = q / np.linalg.norm(q)

    np.testing.assert_allclose(g_out, new_p, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(m1_out, new_m, atol=1e-6, rtol=1e-4)
    np.testing.assert_allclose(m2_out, new_v, atol=1e-6, rtol=1e-4)

    # Adam must zero the grad accumulators for the next iteration.
    raw = np.frombuffer(p.buffers.gradients.to_numpy().tobytes(), dtype=np.int64)
    plane = raw.reshape(14, max_splats)
    assert np.all(plane[:, :N] == 0), "gradient plane not zeroed after Adam"
