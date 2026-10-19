# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Image-quality metrics (PSNR / SSIM), numpy-only.

The paper metric (3DGS / SAD-GS) is the MEAN of per-image PSNR/SSIM over the
held-out views. These primitives are shared by the trainer and the plotting /
inspection scripts; keeping them here gives a single validated implementation
(the torch reference lives in scripts/image_metrics.py).
"""

from __future__ import annotations

import math

import numpy as np


def psnr(rendered: np.ndarray, target: np.ndarray) -> float:
    """Per-image PSNR for data in [0, 1].

    Algebraically identical to INRIA 3DGS / SAD-GS ``image_utils.psnr``
    (20*log10(1/sqrt(mse)) == -10*log10(mse)). Compute per image, then average
    the dB across frames (mean-of-per-image PSNR) — do NOT pool MSE across
    frames and log once (dB-of-mean-MSE is a different, more-pessimistic number
    the paper never reports).
    """
    mse = float(np.mean((rendered - target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def ssim_gaussian_window(window_size: int = 11, sigma: float = 1.5) -> np.ndarray:
    """1D Gaussian window, matching INRIA 3DGS/SAD-GS loss_utils.gaussian."""
    coords = np.arange(window_size) - window_size // 2
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float64)


# Separable 2D window (11x11, sigma=1.5) == reference create_window(11, ...).
# _WIN_1D is the factor actually applied (once per axis); _WIN_2D remains the
# literal 11x11 kernel the equivalence test correlates against.
_WIN_1D = ssim_gaussian_window(11, 1.5)
_WIN_2D = np.outer(_WIN_1D, _WIN_1D)


def ssim(rendered: np.ndarray, target: np.ndarray) -> float:
    """Per-image SSIM for HxWxC data in [0, 1].

    NumPy port of INRIA 3DGS / SAD-GS ``loss_utils.ssim`` (torch reference in
    scripts/image_metrics.py). Matches it faithfully: 11x11 Gaussian window
    (sigma=1.5), 'same' convolution with ZERO padding (torch conv2d padding=5),
    C1=0.01^2, C2=0.03^2, per-channel SSIM map, then the GLOBAL mean over all
    pixels & channels (size_average=True). Validated against the torch
    reference to ~1e-6. Do NOT "optimize" the padding/window — fidelity to the
    reference is what makes it comparable to the SAD-GS / 3DGS SSIM column.

    The window IS applied separably (two 1D passes instead of one 11x11 pass),
    which is NOT such an optimization: ``_WIN_2D`` is by construction
    ``outer(g, g)``, and zero-extension factorizes across axes the same way, so
    the two forms are the same linear operator. Measured agreement is 1e-15 —
    float64 round-off — for a 5x speedup, and SSIM at ~1 MP dominates the cost
    of every evaluation sweep. tests/test_metrics_ssim_separable.py pins the
    equivalence against the literal 11x11 correlation, so a future edit that
    breaks it fails loudly rather than quietly moving every SSIM column.
    """
    from scipy.ndimage import correlate1d  # scipy is always present

    r = rendered.astype(np.float64)
    t = target.astype(np.float64)
    if r.ndim == 2:
        r = r[..., None]
        t = t[..., None]

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    w1d = _WIN_1D

    def filt(img2d: np.ndarray) -> np.ndarray:
        # 'constant' mode with cval=0 == torch conv2d zero-padding=5, applied
        # once per axis (see the docstring: same operator as the 11x11 window).
        rows = correlate1d(img2d, w1d, axis=0, mode="constant", cval=0.0)
        return correlate1d(rows, w1d, axis=1, mode="constant", cval=0.0)

    ssim_channels = []
    for c in range(r.shape[-1]):
        x = r[..., c]
        y = t[..., c]
        mu_x = filt(x)
        mu_y = filt(y)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = filt(x * x) - mu_x2
        sigma_y2 = filt(y * y) - mu_y2
        sigma_xy = filt(x * y) - mu_xy
        ssim_map = (((2 * mu_xy + C1) * (2 * sigma_xy + C2)) /
                    ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)))
        ssim_channels.append(ssim_map)
    return float(np.mean(ssim_channels))
