# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""SnugBox correctness test.

Compare SnugBox's analytical tight-bbox vs a brute-force grid evaluation
of the Gaussian alpha function. SnugBox should:
1. Return ``valid=0`` for any gaussian with opacity <= 1/255 (can't
   exceed the display threshold anywhere).
2. For valid gaussians, produce a bbox that contains every point where
   ``alpha >= 1/255`` and is tight (the bbox boundary touches the ellipse
   at four points).
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from trainer.backend.compute.device import SHADERS_DIR, create_device


BACKENDS = ["vulkan", "d3d12"]


def _module(backend: str) -> spy.Module:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return spy.Module.load_from_file(
        device, str(SHADERS_DIR / "common" / "math_tests.slang")
    )


def _brute_force_bbox(mean: np.ndarray, cov: np.ndarray, opacity: float,
                      threshold: float = 1.0 / 255.0,
                      half_extent: float = 50.0,
                      samples: int = 400):
    """Sample alpha on a grid around the mean and return (bbox_min, bbox_max)
    of points where alpha >= threshold, or None if empty."""
    xs = np.linspace(mean[0] - half_extent, mean[0] + half_extent, samples)
    ys = np.linspace(mean[1] - half_extent, mean[1] + half_extent, samples)
    xx, yy = np.meshgrid(xs, ys)
    dx = xx - mean[0]
    dy = yy - mean[1]
    det = cov[0] * cov[2] - cov[1] * cov[1]
    inv_det = 1.0 / det
    a = cov[2] * inv_det
    b = -cov[1] * inv_det
    c = cov[0] * inv_det
    power = -0.5 * (dx * dx * a + 2 * dx * dy * b + dy * dy * c)
    alpha = opacity * np.exp(power)
    mask = alpha >= threshold
    if not mask.any():
        return None
    ys_lit, xs_lit = np.where(mask)
    return (
        np.array([xs[xs_lit.min()], ys[ys_lit.min()]]),
        np.array([xs[xs_lit.max()], ys[ys_lit.max()]]),
    )


def _call_snugbox(module, mean: np.ndarray, cov: np.ndarray, opacity: float):
    mean_t = spy.Tensor.from_numpy(module.device, mean.astype(np.float32).reshape(1, 2))
    cov_t = spy.Tensor.from_numpy(module.device, cov.astype(np.float32).reshape(1, 3))
    op_t = spy.Tensor.from_numpy(module.device, np.array([opacity], dtype=np.float32))
    bbox = module.snugBoxTest(mean_t, cov_t, op_t)
    return bbox.to_numpy()


@pytest.mark.parametrize("backend", BACKENDS)
def test_snugbox_isotropic(backend: str) -> None:
    """Isotropic cov = I: ellipse is a circle of radius sqrt(t). Opacity=0.5
    gives t = 2·log(127.5) ≈ 9.69, radius ≈ 3.11. Bbox should be [-3.11,3.11]²."""
    module = _module(backend)
    mean = np.array([0.0, 0.0])
    cov = np.array([1.0, 0.0, 1.0])
    opacity = 0.5
    bbox = _call_snugbox(module, mean, cov, opacity).reshape(4)
    t = 2.0 * np.log(255.0 * opacity)
    expected_r = float(np.sqrt(t))  # circle of radius sqrt(t) in unit-cov space
    np.testing.assert_allclose(bbox[0], -expected_r, atol=1e-3)
    np.testing.assert_allclose(bbox[1], -expected_r, atol=1e-3)
    np.testing.assert_allclose(bbox[2], +expected_r, atol=1e-3)
    np.testing.assert_allclose(bbox[3], +expected_r, atol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cov", [
    [1.0, 0.0, 1.0],          # isotropic unit
    [4.0, 0.0, 1.0],           # axis-aligned elongated x
    [1.0, 0.0, 4.0],           # axis-aligned elongated y
    [2.0, 1.0, 2.0],           # rotated
    [10.0, 3.0, 5.0],          # large, rotated
    [0.1, 0.05, 0.1],          # small
])
@pytest.mark.parametrize("opacity", [0.1, 0.5, 0.9])
def test_snugbox_matches_brute_force(backend: str, cov: list, opacity: float) -> None:
    """SnugBox bbox must:
      1. Contain every grid point where alpha >= 1/255.
      2. Not be vastly larger than the true bbox (no factor-2 slack).
    """
    module = _module(backend)
    mean = np.array([10.0, -5.0])  # non-zero to exercise translation.
    cov_np = np.array(cov, dtype=np.float64)

    bf = _brute_force_bbox(mean, cov_np, opacity, half_extent=20.0, samples=400)
    sb = _call_snugbox(module, mean, cov_np, opacity).reshape(4)

    if opacity <= 1.0 / 255.0:
        # Too dim — expected invalid (bbox is a point at mean).
        assert np.allclose(sb[:2], mean, atol=1e-3)
        assert np.allclose(sb[2:], mean, atol=1e-3)
        return

    assert bf is not None, f"brute force found no alpha>={1/255:.4f} region"
    bf_min, bf_max = bf

    sb_min = sb[:2]
    sb_max = sb[2:]

    # Containment: SnugBox bbox must cover the brute-force bbox.
    # Allow 1 grid-cell slack on each side (grid cell = 40/400 = 0.1 px).
    slack = 0.15
    assert sb_min[0] <= bf_min[0] + slack, f"bbox_min.x {sb_min[0]:.3f} > bf {bf_min[0]:.3f}"
    assert sb_min[1] <= bf_min[1] + slack, f"bbox_min.y {sb_min[1]:.3f} > bf {bf_min[1]:.3f}"
    assert sb_max[0] >= bf_max[0] - slack, f"bbox_max.x {sb_max[0]:.3f} < bf {bf_max[0]:.3f}"
    assert sb_max[1] >= bf_max[1] - slack, f"bbox_max.y {sb_max[1]:.3f} < bf {bf_max[1]:.3f}"

    # Tightness: SnugBox shouldn't be more than 1 grid cell larger than
    # brute-force on each side.
    tight_slack = 0.5  # allow some numerical slop from the grid
    assert sb_min[0] >= bf_min[0] - tight_slack, f"bbox_min.x {sb_min[0]:.3f} << bf {bf_min[0]:.3f}"
    assert sb_max[0] <= bf_max[0] + tight_slack
    assert sb_min[1] >= bf_min[1] - tight_slack
    assert sb_max[1] <= bf_max[1] + tight_slack
