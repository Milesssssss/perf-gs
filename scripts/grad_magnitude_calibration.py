"""Calibrate our grad_mean_2d_mag signal against a known analytic value.

The signal is now the NORM of the per-iter pixel-summed 2D gradient
vector (matches gsplat's `grads.norm(dim=-1)`). For a single
forward+backward this is simply ‖Σ_p dL/dmean2d_p‖.

Setup: single gaussian, known mean/cov/opacity, known target image of
uniform color. Single forward+backward. Read back
grad_mean_2d_mag_accum and visibility_count_accum.

Compute analytically:
  For pixel p:
    alpha_p  = sigmoid(ol) * exp(-0.5 (p - mu)^T Sig^-1 (p - mu))
    C_p      = c * alpha_p    (single gaussian, T_0 = 1)
    residual = 2 * (C_p - T_p)   in each channel
    dL/dalpha_p = residual . c     (vector dot)
    dC_p / dmean_2d  via chain rule on evalGaussian2DFromCov
    dL/dmean_2d_p = dL/dalpha_p * (dalpha_p / dmean_2d)
                  = dL/dalpha_p * alpha_p * (Sig^-1 (p - mu))

Then analytic mag = ‖Σ_p dL/dmean_2d_p‖. For a target image that's
uniform, the per-pixel grads cancel by symmetry and we'd expect the
sum-norm to be near zero — break the symmetry by placing the target
asymmetrically.
"""
from __future__ import annotations

import math

import numpy as np

from trainer.dataset import simple_camera
from trainer.device import create_device
from trainer.pipeline import Pipeline


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    low = c / 12.92
    high = np.power(np.clip((c + 0.055) / 1.055, 1e-7, None), 2.4)
    return np.where(c <= 0.04045, low, high).astype(np.float32)


def main() -> None:
    # --- Scene: single gaussian at origin, camera at +Z=3 looking at origin ---
    W, H = 64, 64
    device = create_device(backend="vulkan")
    pipeline = Pipeline(device=device, width=W, height=H, max_splats=16, max_sort_pairs=10_000)
    pipeline.z_near_cull = 0.01

    # Known params.
    mean = np.array([[0.0, 0.0, 3.0]], dtype=np.float32)
    log_scale_val = np.log(0.1).astype(np.float32)
    log_scale = np.full((1, 3), log_scale_val, dtype=np.float32)
    rot_quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    opacity = np.array([0.9], dtype=np.float32)     # probability
    # Color: red.
    color_rgb = np.array([0.8, 0.1, 0.1], dtype=np.float32)
    # color_logit = logit(color_rgb). Same conversion as dataset.py.
    def logit(p):
        p = np.clip(p, 1e-5, 1.0 - 1e-5)
        return np.log(p / (1.0 - p))
    color_logit = logit(color_rgb).reshape(1, 3).astype(np.float32)

    pipeline.set_gaussians_from_numpy(
        means=mean, log_scales=log_scale, rot_quats=rot_quat,
        opacities=opacity, color_logits=color_logit,
    )

    # --- Target: uniform green ---
    target_srgb = np.ones((H, W, 3), dtype=np.float32)
    target_srgb[..., 0] = 0.0
    target_srgb[..., 1] = 0.8
    target_srgb[..., 2] = 0.0
    pipeline.set_target_image(target_srgb)
    target_linear = _srgb_to_linear(target_srgb)

    # --- Camera: look from -Z, so gaussian projects to image center ---
    cam = simple_camera(
        width=W, height=H,
        eye=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        fovy_degrees=60.0,
    )

    # --- Run forward + loss + backward ---
    pipeline.clear_tracking()
    pipeline._clear_grads()
    pipeline.forward_and_loss(cam)
    pipeline.backward()

    # --- Read our shader's tracking ---
    tracking = pipeline.read_tracking()
    mag_total_gpu = float(tracking["grad_mean_2d_mag"][0])
    vis_gpu = int(tracking["visibility_count"][0])

    # --- Compute analytical mag_total on CPU ---
    # Render-side knowns:
    #   mean_2d_px = image center = (W/2, H/2) = (32, 32)
    #   Gaussian is isotropic with world scale 0.1, at depth 3, focal pixels
    #   focal_px = cot(fovy/2) * W/2 = cot(30°) * 32 ≈ 1.732 * 32 = 55.4
    #   screen-space sigma = focal_px * 0.1 / 3 ≈ 1.847 px
    # 2D cov is diag(sigma^2 + 0.3, sigma^2 + 0.3) per project.slang's
    # low-pass filter. Sigma^2 ≈ 3.41, so cov_00 ≈ 3.71.
    fovy = math.radians(60.0)
    focal_px = (1.0 / math.tan(fovy / 2.0)) * (W / 2.0)
    sigma_screen = focal_px * 0.1 / 3.0
    cov00 = sigma_screen ** 2 + 0.3
    cov11 = cov00
    cov01 = 0.0

    # Read actual cov from GPU to compare.
    raw_proj = pipeline.buffers.projected_2d.to_numpy()
    proj = np.frombuffer(raw_proj.tobytes(), dtype=np.float32).reshape(-1, 6)[0]
    mean_2d = proj[0:2]
    cov_gpu = proj[2:5]
    depth_gpu = proj[5]
    print(f"projection: mean_2d={mean_2d}, cov={cov_gpu}, depth={depth_gpu:.4f}")
    print(f"analytical: mean_2d=({W/2},{H/2}), cov=[{cov00:.4f},{cov01},{cov11:.4f}]")

    # Use GPU values for analysis.
    mu = mean_2d
    a = cov_gpu[2] / (cov_gpu[0] * cov_gpu[2] - cov_gpu[1]**2)  # inv_det * cov11
    b = -cov_gpu[1] / (cov_gpu[0] * cov_gpu[2] - cov_gpu[1]**2)
    c = cov_gpu[0] / (cov_gpu[0] * cov_gpu[2] - cov_gpu[1]**2)

    # Per-pixel analytical gradient.
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    px = np.stack([xs + 0.5, ys + 0.5], axis=-1).reshape(-1, 2)
    d = px - mu[None, :]
    power = -0.5 * (d[:, 0]**2 * a + 2 * d[:, 0] * d[:, 1] * b + d[:, 1]**2 * c)
    alpha_raw = 0.9 * np.exp(power)     # opacity * exp(power)
    # Single gaussian: T_0 = 1, so C = color_rgb * alpha, no contribution from others.
    # Apply same alpha cap as the shader.
    alpha = np.minimum(alpha_raw, 0.99)
    C = color_rgb[None, :] * alpha[:, None]
    residual = 2.0 * (C - target_linear.reshape(-1, 3))
    # dL/dalpha_pixel = dL/dC . dC/dalpha = residual . color_rgb
    dL_dalpha = (residual * color_rgb[None, :]).sum(axis=-1)
    # For the min(x, 0.99) path, grad flows only when alpha_raw <= 0.99.
    dL_dalpha = np.where(alpha_raw <= 0.99, dL_dalpha, 0.0)
    # dalpha / dmean_2d = alpha_raw * Sig^-1 (p - mu)
    # Note: d/dmu (exp(-0.5 d^T Sig^-1 d)) where d = p - mu
    # gradient in pixel coords is alpha_raw * Sig^-1 (p - mu)
    g_mean_2d_x = dL_dalpha * alpha_raw * (a * d[:, 0] + b * d[:, 1])
    g_mean_2d_y = dL_dalpha * alpha_raw * (b * d[:, 0] + c * d[:, 1])
    # Only pixels above the alpha gate contribute to our shader's accumulator.
    gate = alpha >= 1.0 / 255.0
    g_mean_2d_x_gated = np.where(gate, g_mean_2d_x, 0.0)
    g_mean_2d_y_gated = np.where(gate, g_mean_2d_y, 0.0)

    # New signal: norm of the per-iter sum vector (gsplat semantics).
    sum_x = g_mean_2d_x_gated.sum()
    sum_y = g_mean_2d_y_gated.sum()
    analytic_mag = np.sqrt(sum_x * sum_x + sum_y * sum_y)

    # Visibility count analytical: this gaussian is visible this iter → 1.
    contributing = int(gate.sum())

    print(f"\n--- Measurements ---")
    print(f"GPU tracking: mag_total={mag_total_gpu:.6f}, visibility={vis_gpu}")
    print(f"Analytical:   mag_total={analytic_mag:.6f}, contributing pixels={contributing}")
    print(f"GPU / analytical ratio: {mag_total_gpu / max(analytic_mag, 1e-12):.4f}")
    print(f"\nmag_avg (GPU): {mag_total_gpu / max(vis_gpu, 1):.4e}")
    print(f"mag_avg (analytical): {analytic_mag / 1.0:.4e}")


if __name__ == "__main__":
    main()
