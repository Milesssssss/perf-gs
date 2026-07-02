"""Finite-difference check of the backward pass WITH alpha supervision on.

The existing tests/test_gradients.py hardcodes w_alpha=0, so the accumulated-
alpha gradient path (dL_dA * dA/dalpha_i in backward.slang) was never validated.
That path was where the missing occlusion term lived.

This script sets w_alpha>0, uses OVERLAPPING gaussians at different depths so
there is genuine occlusion (A_behind != 0), and checks the analytic opacity /
mean / log-scale gradients against central differences of the FULL loss
(RGB L2 + w_alpha * alpha L2). Opacity is the most direct probe of the alpha
term since alpha = sigmoid(opacity) * exp(power).
"""
from __future__ import annotations

import numpy as np

from trainer.dataset import simple_camera
from trainer.device import create_device
from trainer.pipeline import Pipeline

W = H = 16
W_ALPHA = 0.2


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    low = c / 12.92
    high = np.power(np.clip((c + 0.055) / 1.055, 1e-7, None), 2.4)
    return np.where(c <= 0.04045, low, high).astype(np.float32)


def _scene():
    # Three gaussians stacked along the view ray with lateral overlap so the
    # front ones occlude the back ones -> A_behind matters.
    means = np.array([
        [0.0, 0.0, 2.0],
        [0.05, 0.0, 2.5],
        [-0.05, 0.03, 3.0],
    ], dtype=np.float32)
    log_scales = np.log(np.array([
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ], dtype=np.float32))
    rot_quats = np.tile(np.array([1, 0, 0, 0], np.float32), (3, 1))
    opacities = np.array([0.6, 0.6, 0.6], dtype=np.float32)
    color_logits = np.array([
        [1.0, -0.5, -0.5],
        [-0.5, 1.0, -0.5],
        [-0.5, -0.5, 1.0],
    ], dtype=np.float32)
    return dict(means=means, log_scales=log_scales, rot_quats=rot_quats,
                opacities=opacities, color_logits=color_logits)


def _target():
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[..., 0] = np.linspace(0.1, 0.7, W)[None, :]
    rgb[..., 1] = 0.3
    rgb[..., 2] = np.linspace(0.6, 0.1, H)[:, None]
    alpha = np.full((H, W, 1), 0.5, np.float32)  # non-trivial alpha target
    return np.concatenate([rgb, alpha], axis=-1)


def _upload_target(p: Pipeline, rgba: np.ndarray) -> None:
    p.buffers.target_image.copy_from_numpy(np.ascontiguousarray(rgba))
    p._active_target = p.buffers.target_image


def _full_loss(p: Pipeline, rgba_target: np.ndarray) -> float:
    """L = sum_pixels [ |C - srgb2lin(T_rgb)| + w_alpha |A_r - A_t| ]
    (L1, matching l2_linear.slang)."""
    out = p.buffers.output_image.to_numpy()
    C = out[..., :3]
    A_r = out[..., 3]
    T_lin = _srgb_to_linear(rgba_target[..., :3])
    A_t = rgba_target[..., 3]
    l_rgb = np.sum(np.abs(C - T_lin))
    l_alpha = W_ALPHA * np.sum(np.abs(A_r - A_t))
    return float(l_rgb + l_alpha)


def main() -> None:
    device = create_device(backend="vulkan")
    p = Pipeline(device=device, width=W, height=H, max_splats=16)
    p.lambda_ssim = 0.0
    p.w_alpha = W_ALPHA
    p.opacity_reg = 0.0
    p.scale_reg = 0.0

    cam = simple_camera(width=W, height=H, eye=(0, 0, 0), target=(0, 0, 1),
                        up=(0, 1, 0), fovy_degrees=60.0)
    tgt = _target()
    scene = _scene()
    n = scene["means"].shape[0]

    p.set_gaussians_from_numpy(**scene)
    _upload_target(p, tgt)
    p.forward_and_loss(cam)
    p._clear_grads()
    p.backward()
    p.device.wait_for_idle()
    analytic = p.read_gradients()

    eps = 1e-3
    table = [
        ("opacity", "opacities", None),
        ("mean_x", "means", 0),
        ("mean_z", "means", 2),
        ("log_scale_x", "log_scales", 0),
        ("log_scale_y", "log_scales", 1),
    ]

    def loss_for(sc):
        p.set_gaussians_from_numpy(**sc)
        _upload_target(p, tgt)
        p.forward_and_loss(cam)
        p.device.wait_for_idle()
        return _full_loss(p, tgt)

    print(f"FD gradient check with w_alpha={W_ALPHA}, {n} occluding gaussians\n")
    worst = 0.0
    for gname, field, idx in table:
        col = analytic[gname]
        for gi in range(n):
            sp = {k: v.copy() for k, v in scene.items()}
            sm = {k: v.copy() for k, v in scene.items()}
            if gname == "opacity":
                pr = float(np.clip(scene[field][gi], 1e-5, 1 - 1e-5))
                bl = np.log(pr / (1 - pr))
                sp[field][gi] = 1 / (1 + np.exp(-(bl + eps)))
                sm[field][gi] = 1 / (1 + np.exp(-(bl - eps)))
            elif idx is None:
                sp[field][gi] += eps
                sm[field][gi] -= eps
            else:
                sp[field][gi, idx] += eps
                sm[field][gi, idx] -= eps
            num = (loss_for(sp) - loss_for(sm)) / (2 * eps)
            a = float(col[gi])
            denom = max(abs(num), abs(a), 1e-2)
            rel = abs(num - a) / denom
            worst = max(worst, rel if abs(num - a) > 1e-2 else 0.0)
            flag = "  <-- MISMATCH" if (rel > 0.10 and abs(num - a) > 1e-2) else ""
            print(f"g{gi} {gname:12s} analytic={a:+.4e} numeric={num:+.4e} rel={rel:.3f}{flag}")

    print(f"\nworst rel-err (above abs floor): {worst:.3f}")
    print("PASS" if worst <= 0.10 else "FAIL")


if __name__ == "__main__":
    main()
