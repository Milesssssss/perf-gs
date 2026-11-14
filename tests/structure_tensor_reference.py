# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""scipy reference of the structure-tensor field, the oracle for the GPU builder."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import convolve, gaussian_filter

from trainer.backend.densify.structure_tensor import (
    GAMMA,
    MIN_ENERGY,
    NUM_LEVELS,
    OMEGA_CALIBRATION,
    RHO_FACTOR,
    SIGMA_BASE,
)

EPS = 1e-8
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def _sobel(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # scipy's convolve flips the kernel; flip it back to match torch conv2d.
    return convolve(channel, kernel[::-1, ::-1], mode="nearest")


def _base_tensor(image: np.ndarray, sigma: float, rho: float):
    """Di Zenzo multi-channel tensor, normalized by the image's max trace."""
    smoothed = gaussian_filter(image, sigma=(sigma, sigma, 0.0), mode="nearest")
    sxx = np.zeros(image.shape[:2])
    sxy = np.zeros(image.shape[:2])
    syy = np.zeros(image.shape[:2])
    for c in range(smoothed.shape[2]):
        ix = _sobel(smoothed[:, :, c], _SOBEL_X)
        iy = _sobel(smoothed[:, :, c], _SOBEL_Y)
        sxx += ix * ix
        sxy += ix * iy
        syy += iy * iy
    sxx = gaussian_filter(sxx, rho, mode="nearest")
    sxy = gaussian_filter(sxy, rho, mode="nearest")
    syy = gaussian_filter(syy, rho, mode="nearest")
    max_trace = float(np.max(sxx + syy)) + 1e-6
    return sxx / max_trace, sxy / max_trace, syy / max_trace


def structure_tensor_field(image_srgb: np.ndarray, num_levels: int = NUM_LEVELS,
                           sigma_base: float = SIGMA_BASE, rho_factor: float = RHO_FACTOR,
                           gamma: float = GAMMA) -> np.ndarray:
    """``(H, W, 3)`` float32 (Sxx, Sxy, Syy) of a raw sRGB image (alpha ignored)."""
    image = np.asarray(image_srgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected (H, W, >=3) image, got {image.shape}")
    image = image[:, :, :3]
    height, width = image.shape[:2]
    sxx_base, sxy_base, syy_base = _base_tensor(image, sigma=sigma_base, rho=1.0)
    num = np.zeros((height, width, 3))
    den = np.zeros((height, width))
    current = image.astype(np.float64)
    sigma_accum = 0.0
    for level in range(num_levels):
        target_sigma = sigma_base ** level
        sigma_inc = np.sqrt(max(1e-6, target_sigma ** 2 - sigma_accum ** 2))
        following = gaussian_filter(current, sigma=(sigma_inc, sigma_inc, 0.0), mode="nearest")
        band = np.sqrt(np.sum((current - following) ** 2, axis=-1))
        if level > 0:
            band = gaussian_filter(band, sigma=target_sigma * 2.0, mode="nearest")
        rho = rho_factor * target_sigma
        sxx = gaussian_filter(sxx_base, rho, mode="nearest")
        sxy = gaussian_filter(sxy_base, rho, mode="nearest")
        syy = gaussian_filter(syy_base, rho, mode="nearest")
        trace = sxx + syy + EPS
        weight = band ** gamma
        freq_sq = (OMEGA_CALIBRATION / target_sigma) ** 2
        num[:, :, 0] += sxx / trace * weight * freq_sq
        num[:, :, 1] += sxy / trace * weight * freq_sq
        num[:, :, 2] += syy / trace * weight * freq_sq
        den += weight
        current, sigma_accum = following, target_sigma
    return (num / (den[:, :, None] + MIN_ENERGY)).astype(np.float32)


def principal_eigenvalue(field: np.ndarray) -> np.ndarray:
    """Top eigenvalue of each packed symmetric 2x2 tensor [[a, b], [b, c]]."""
    a, b, c = field[..., 0], field[..., 1], field[..., 2]
    mid = 0.5 * (a + c)
    return (mid + np.sqrt(np.maximum((0.5 * (a - c)) ** 2 + b * b, 0.0))).astype(np.float32)
