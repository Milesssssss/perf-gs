"""Validation of the multi-scale structure-tensor field (structure_analysis.py).

The oracle is analytic: for a sinusoidal grating sin(2*pi*omega*x), the image
gradient is purely horizontal and its magnitude scales with omega, so the
aggregated structure tensor S̄ must (a) be dominated by its Sxx component,
(b) have its principal eigenvector aligned with x, and (c) have a principal
eigenvalue that GROWS with the grating frequency — which is exactly the signal
the eta metric turns into a "local wavelength". A vertical grating must produce
the transposed result (Syy-dominated). A flat image must produce ~zero tensor.
"""

from __future__ import annotations

import numpy as np

from trainer.structure_analysis import (
    EPS,
    compute_structure_tensor_field,
    principal_eigenvalue,
)


def _grating(h: int, w: int, omega: float, axis: str) -> np.ndarray:
    """RGB image of a sinusoidal grating of frequency omega (cycles/pixel)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coord = xx if axis == "x" else yy
    val = 0.5 + 0.5 * np.sin(2.0 * np.pi * omega * coord)
    return np.stack([val, val, val], axis=-1).astype(np.float32)


def test_flat_image_has_negligible_tensor() -> None:
    img = np.full((64, 64, 3), 0.5, dtype=np.float32)
    sbar = compute_structure_tensor_field(img)
    assert np.all(np.abs(sbar) < 1e-3), f"flat image tensor not ~0: max {np.abs(sbar).max()}"


def test_horizontal_grating_is_sxx_dominated() -> None:
    # Grating varying along x → gradients point in x → Sxx >> Syy.
    img = _grating(96, 96, omega=0.1, axis="x")
    sbar = compute_structure_tensor_field(img)
    # Sample the interior (avoid border effects).
    core = sbar[24:72, 24:72]
    sxx = np.median(core[..., 0])
    syy = np.median(core[..., 2])
    assert sxx > 10.0 * max(syy, 1e-6), f"expected Sxx >> Syy, got Sxx={sxx:.3e} Syy={syy:.3e}"


def test_vertical_grating_is_syy_dominated() -> None:
    img = _grating(96, 96, omega=0.1, axis="y")
    sbar = compute_structure_tensor_field(img)
    core = sbar[24:72, 24:72]
    sxx = np.median(core[..., 0])
    syy = np.median(core[..., 2])
    assert syy > 10.0 * max(sxx, 1e-6), f"expected Syy >> Sxx, got Sxx={sxx:.3e} Syy={syy:.3e}"


def test_principal_eigenvalue_grows_with_frequency() -> None:
    # Higher-frequency texture → finer detail → larger lambda1 → smaller
    # local wavelength. This monotonicity is what drives eta.
    lambdas = []
    for omega in (0.04, 0.08, 0.16):
        img = _grating(128, 128, omega=omega, axis="x")
        sbar = compute_structure_tensor_field(img)
        lam = principal_eigenvalue(sbar)[32:96, 32:96]
        lambdas.append(float(np.median(lam)))
    assert lambdas[0] < lambdas[1] < lambdas[2], (
        f"lambda1 not monotincreasing with frequency: {lambdas}"
    )


def test_output_shape_and_dtype() -> None:
    img = _grating(48, 72, omega=0.1, axis="x")
    sbar = compute_structure_tensor_field(img)
    assert sbar.shape == (48, 72, 3)
    assert sbar.dtype == np.float32
    # Accepts RGBA too (alpha ignored).
    rgba = np.concatenate([img, np.ones((48, 72, 1), np.float32)], axis=-1)
    sbar2 = compute_structure_tensor_field(rgba)
    assert sbar2.shape == (48, 72, 3)


def _stripe_wavelength(wavelength: float, size: int = 256, num_levels: int = 6) -> float:
    """Recovered local wavelength Lambda_min at the center of a horizontal
    stripe pattern of the given period (pixels)."""
    x = np.arange(size)
    row = 0.5 + 0.5 * np.sin(2.0 * np.pi * x / wavelength)
    img = np.stack([np.tile(row, (size, 1))] * 3, axis=-1).astype(np.float32)
    sbar = compute_structure_tensor_field(img, num_levels=num_levels)
    lam1 = principal_eigenvalue(sbar)[size // 4:3 * size // 4, size // 4:3 * size // 4]
    return 1.0 / (float(np.sqrt(lam1.mean())) + EPS)


def test_wavelength_monotone_and_stable() -> None:
    """With the SADGS convention (OMEGA_CALIBRATION=1.0), Lambda_min is a fixed
    fraction (~0.25x) of the true wavelength — NOT the absolute wavelength. What
    must hold for eta to be a valid frequency signal is that Lambda_min (a)
    increases monotonically with the true stripe wavelength and (b) does so with
    a roughly stable proportionality within the resolvable band. (Absolute
    recovery is intentionally not asserted — the eta>1 split threshold is
    calibrated to this convention to match SADGS.)"""
    wls = [8.0, 12.0, 16.0, 20.0]
    recovered = [_stripe_wavelength(wl) for wl in wls]
    # Monotone increasing.
    assert all(recovered[i] < recovered[i + 1] for i in range(len(recovered) - 1)), (
        f"Lambda_min not monotone in wavelength: {recovered}"
    )
    # Roughly stable proportionality within the band.
    ratios = np.array([r / wl for r, wl in zip(recovered, wls)])
    assert ratios.max() / ratios.min() < 2.0, (
        f"wavelength proportionality not stable across the band: ratios={ratios}"
    )


def test_anisotropy_eigenvector_direction() -> None:
    """Vertical stripes (intensity varies along x) → top eigenvector of S̄
    aligned with x (Sxx dominates)."""
    img = _grating(96, 96, omega=0.1, axis="x")
    sbar = compute_structure_tensor_field(img)
    a, b, c = (float(v) for v in sbar[48, 48])  # Sxx, Sxy, Syy at center
    assert a > c, f"expected Sxx > Syy for x-varying grating, got Sxx={a:.3e} Syy={c:.3e}"
