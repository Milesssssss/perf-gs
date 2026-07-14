"""Multi-scale structure-tensor analysis for structure-aware densification.

Implements the image-side preprocessing of Lyu et al., "Faster 3D Gaussian
Splatting Convergence via Structure-Aware Densification" (SIGGRAPH 2026,
arXiv 2604.28016), Section 3.1.2 + 3.2. For each training image it produces a
per-pixel aggregated structure tensor ``S̄`` that encodes the dominant local
texture frequency (and orientation), used downstream to decide *how much* to
subdivide each Gaussian.

This runs ONCE per training image at load time (not in the hot loop), so it's
implemented on the CPU with scipy/numpy for clarity and testability. The paper
does it on the GPU (~0.7 s total); porting to Slang is a later optimization —
the output layout (a per-pixel float3 ``(Sxx, Sxy, Syy)`` texture) is chosen so
the GPU sampler in the eta kernel doesn't care how it was produced.

Pipeline (per image ``I`` in linear RGB, shape (H, W, 3)):
  1. Gaussian scale space: ``I_l = G_{sigma_l} * I`` for l in 0..L, full
     resolution (no downsampling), ``sigma_l = SIGMA_BASE ** l``.
  2. Per-level structure tensor ``S_l = G_{rho_l} * (grad I_l)(grad I_l)^T``
     summed over color channels (Di Zenzo multi-channel form), ``rho_l = 3 sigma_l``.
  3. Normalize each level: ``Ŝ_l = S_l / (tr(S_l) + eps)`` (strips contrast).
  4. Laplacian band-pass energy ``E_l = ||I_{l-1} - I_l||^2`` (per pixel).
  5. Aggregate: ``S̄ = sum_l E_l^gamma * omega_l^2 * Ŝ_l / (sum_l E_l^gamma + eps)``
     where ``omega_l = 1 / sigma_l`` is the band's characteristic frequency.

The principal eigenvalue ``lambda1`` of ``S̄`` gives the local minimum
wavelength ``Lambda_min = 1 / (sqrt(lambda1) + eps)`` in the eta metric.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

# Defaults from the paper (Section 3.2): L=4 levels, sigma_l = 1.5^l,
# integration scale rho_l = 3 sigma_l, energy sharpness gamma = 3.0.
NUM_LEVELS = 4
SIGMA_BASE = 1.5
RHO_FACTOR = 3.0
GAMMA = 3.0
EPS = 1e-8

# Per-band characteristic frequency omega_l = OMEGA_CALIBRATION / sigma_l,
# applied SQUARED in the Eq. 5 weighting. The reference SADGS implementation
# (utils/loss_utils.py) uses band_freq = 1/octave_step^l = 1/sigma_l with NO
# extra constant (no 1/(2*pi), no calibration knock-down), so we match it:
# OMEGA_CALIBRATION = 1.0. This means Lambda_min = 1/sqrt(lambda1) is NOT the
# absolute pixel wavelength (it's ~0.25x on a grating) — that is intentional
# and matches SADGS; their eta>1 split threshold is calibrated to THIS
# convention, so changing the constant would desync from their split behavior.
OMEGA_CALIBRATION = 1.0

# Energy floor for the Eq. 5 aggregation denominator (den = sum E^gamma). With
# [0,1] images, E^gamma in smooth regions is ~1e-9..1e-6, so a numerical-guard
# epsilon of 1e-8 would silently act as an energy floor. We make that explicit:
# in low-energy (flat) regions the tensor is driven toward zero -> tiny lambda1
# -> huge Lambda_min -> eta ~ 0 -> no split, which is the desired fail-safe.
# Named separately from EPS so "fixing" the numerical epsilon can't change it.
MIN_ENERGY = 1e-6

# Bumped whenever the field math changes so stale caches are ignored.
# v4: rewritten to match SADGS _v1 exactly (sqrt DoG band response, base tensor
# re-blurred per level, Sobel, 4 bands) — field values changed.
# v5: input is now RAW sRGB (matches SADGS train.py:184), was erroneously fed
# linearized RGB in v4 — field values changed again.
CACHE_VERSION = 6


def cache_path(scene_dir, width: int, height: int,
               num_levels: int = NUM_LEVELS, sigma_base: float = SIGMA_BASE,
               rho_factor: float = RHO_FACTOR, gamma: float = GAMMA):
    """Path of the on-disk structure-tensor cache for a scene at a given
    resolution + analysis params. Lives under ``<scene_dir>/structure_cache/``.

    The filename encodes every parameter that affects the field, so a cache
    built at one resolution / gamma won't be silently reused at another."""
    from pathlib import Path
    key = (f"sad_v{CACHE_VERSION}_{width}x{height}"
           f"_L{num_levels}_s{sigma_base:g}_r{rho_factor:g}_g{gamma:g}.npz")
    return Path(scene_dir) / "structure_cache" / key


# 3x3 Sobel kernels (unnormalized), matching SADGS get_structure_tensor_torch.
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)


def _sobel(chan: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve one (H, W) channel with a 3x3 kernel, replicate-padded."""
    from scipy.ndimage import convolve
    # scipy `convolve` correlates with the flipped kernel; flip so it matches a
    # true convolution / the torch conv2d SADGS uses.
    return convolve(chan, kernel[::-1, ::-1], mode="nearest")


def _structure_tensor(img_l: np.ndarray, sigma: float, rho: float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-scale Di Zenzo multi-channel structure tensor, matching SADGS
    ``get_structure_tensor_torch``: pre-smooth by sigma, per-channel Sobel
    gradients, sum products over channels, integrate by rho, then normalize by
    the per-image max of the trace (NOT by trace here — that division is done
    per-level in the aggregation).

        S_xx = G_rho * sum_c (d/dx G_sigma*I_c)^2   (and xy, yy)

    Returns (Sxx, Sxy, Syy), each (H, W)."""
    smoothed = gaussian_filter(img_l, sigma=(sigma, sigma, 0.0), mode="nearest")
    sxx = np.zeros(img_l.shape[:2], dtype=np.float64)
    sxy = np.zeros(img_l.shape[:2], dtype=np.float64)
    syy = np.zeros(img_l.shape[:2], dtype=np.float64)
    for c in range(smoothed.shape[2]):
        ix = _sobel(smoothed[:, :, c], _SOBEL_X)
        iy = _sobel(smoothed[:, :, c], _SOBEL_Y)
        sxx += ix * ix
        sxy += ix * iy
        syy += iy * iy
    # Spatial integration (the G_rho * (...) convolution).
    sxx = gaussian_filter(sxx, rho, mode="nearest")
    sxy = gaussian_filter(sxy, rho, mode="nearest")
    syy = gaussian_filter(syy, rho, mode="nearest")
    # Batch-max normalization (SADGS divides Sxx/Sxy/Syy by max(trace)+1e-6).
    max_trace = float(np.max(sxx + syy)) + 1e-6
    return sxx / max_trace, sxy / max_trace, syy / max_trace


def compute_structure_tensor_field(
    image_srgb: np.ndarray,
    num_levels: int = NUM_LEVELS,
    sigma_base: float = SIGMA_BASE,
    rho_factor: float = RHO_FACTOR,
    gamma: float = GAMMA,
) -> np.ndarray:
    """Aggregated multi-scale structure tensor ``S̄`` for one image.

    ``image_srgb`` is (H, W, 3) or (H, W, 4) RAW sRGB in [0, 1] (alpha, if
    present, is ignored). Pass the sRGB image directly — do NOT linearize —
    because SADGS computes the tensor on the raw loaded PNG (train.py:184).
    Returns a (H, W, 3) float32 array packing ``(Sxx, Sxy, Syy)`` per pixel —
    the symmetric 2x2 tensor S̄.

    Ported to match SADGS get_multiscale_structure_tensor_v1 (the mode they
    actually run, st_mode="v1"):
      - The orientation tensor is computed ONCE at the finest scale
        (base_sigma), then re-blurred (integrated) per level — NOT recomputed
        per level (that is their unused "_v2").
      - Per level: re-integrate the base tensor by rho = 3*target_sigma,
        trace-normalize to orientation-only, weight by band_response^gamma *
        band_freq^2, accumulate. band_response is the L2 DoG norm sqrt(sum_c
        (I_l - I_{l+1})^2) — note the sqrt (previously we squared, which
        over-sharpened the band weighting by 2x in the exponent).
      - band_freq = 1/octave^l (== 1/sigma_l), applied squared. Denominator
        accumulates band_response^gamma WITHOUT the freq factor.
      - num_levels loops range(num_levels) = 4 bands (was range(num_levels+1)).
    """
    img = np.asarray(image_srgb, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"expected (H, W, >=3) image, got {img.shape}")
    img = img[:, :, :3]
    H, W = img.shape[:2]

    # Orientation tensor computed ONCE at the finest scale (base tensor);
    # re-blurred per level below. (SADGS _v1.)
    sxx_base, sxy_base, syy_base = _structure_tensor(img, sigma=sigma_base, rho=1.0)

    num = np.zeros((H, W, 3), dtype=np.float64)  # weighted-tensor accumulator
    den = np.zeros((H, W), dtype=np.float64)     # weight accumulator

    # Cascaded Gaussian pyramid for the DoG band response: current_smooth is
    # progressively blurred so its accumulated sigma reaches target_sigma.
    current_smooth = img.astype(np.float64)
    sigma_accum = 0.0

    for level in range(num_levels):
        band_freq = 1.0 / (sigma_base ** level)     # == OMEGA_CALIBRATION/sigma_l w/ CAL=1
        target_sigma = sigma_base ** level
        rho_l = rho_factor * target_sigma

        # Incremental blur so total accumulated sigma == target_sigma.
        sigma_inc = np.sqrt(max(1e-6, target_sigma ** 2 - sigma_accum ** 2))
        next_smooth = gaussian_filter(current_smooth, sigma=(sigma_inc, sigma_inc, 0.0),
                                      mode="nearest")

        # Band response = L2 DoG norm across channels (SADGS: .pow(2).sum(c).sqrt()).
        band_response = np.sqrt(np.sum((current_smooth - next_smooth) ** 2, axis=-1))
        # Low-frequency spatial context smoothing for levels > 0.
        if level > 0:
            band_response = gaussian_filter(band_response, sigma=target_sigma * 2.0,
                                            mode="nearest")

        # Re-integrate the base orientation tensor at this scale, trace-normalize.
        sxx_i = gaussian_filter(sxx_base, rho_l, mode="nearest")
        sxy_i = gaussian_filter(sxy_base, rho_l, mode="nearest")
        syy_i = gaussian_filter(syy_base, rho_l, mode="nearest")
        tr_i = sxx_i + syy_i + EPS
        nxx, nxy, nyy = sxx_i / tr_i, sxy_i / tr_i, syy_i / tr_i

        weight = band_response ** gamma                 # E_l^gamma
        freq_sq = (band_freq * OMEGA_CALIBRATION) ** 2  # target trace = freq^2

        num[:, :, 0] += nxx * weight * freq_sq
        num[:, :, 1] += nxy * weight * freq_sq
        num[:, :, 2] += nyy * weight * freq_sq
        den += weight                                   # NOTE: no freq factor (SADGS)

        current_smooth = next_smooth
        sigma_accum = target_sigma

    sbar = num / (den[:, :, None] + MIN_ENERGY)
    return sbar.astype(np.float32)


def principal_eigenvalue(sbar: np.ndarray) -> np.ndarray:
    """Top eigenvalue lambda1 of the packed symmetric 2x2 field ``S̄``.

    ``sbar`` is (H, W, 3) = (Sxx, Sxy, Syy). Returns (H, W) float32.
    For [[a, b], [b, c]]: lambda1 = (a+c)/2 + sqrt(((a-c)/2)^2 + b^2)."""
    a = sbar[..., 0]
    b = sbar[..., 1]
    c = sbar[..., 2]
    mid = 0.5 * (a + c)
    disc = np.sqrt(np.maximum((0.5 * (a - c)) ** 2 + b * b, 0.0))
    return (mid + disc).astype(np.float32)
