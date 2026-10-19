# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Pin ``metrics.ssim``'s separable window to the literal 11x11 correlation.

``ssim`` applies the Gaussian window as two 1D passes because SSIM at ~1 MP is
the dominant cost of an evaluation sweep. That is only legitimate because
``_WIN_2D`` is ``outer(_WIN_1D, _WIN_1D)`` and zero-extension factorizes across
axes, making the two forms the same linear operator.

This test is the guard on that claim: it recomputes SSIM with the literal 11x11
2D correlation — the form the docstring's "matches INRIA/SAD-GS" contract is
written against — and asserts the separable implementation agrees to float64
round-off. If someone later changes the window, the padding mode, or the pass
order, this fails instead of silently shifting every reported SSIM column.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import correlate

from trainer.common.metrics import _WIN_2D, ssim


def _ssim_2d_reference(rendered: np.ndarray, target: np.ndarray) -> float:
    """SSIM with the window applied as ONE 11x11 correlation (the reference
    form). Deliberately a verbatim restatement rather than a call into
    ``metrics``, so it cannot drift with the implementation it checks."""
    r = rendered.astype(np.float64)
    t = target.astype(np.float64)
    if r.ndim == 2:
        r = r[..., None]
        t = t[..., None]

    C1, C2 = 0.01 ** 2, 0.03 ** 2

    def filt(img2d: np.ndarray) -> np.ndarray:
        return correlate(img2d, _WIN_2D, mode="constant", cval=0.0)

    maps = []
    for c in range(r.shape[-1]):
        x, y = r[..., c], t[..., c]
        mu_x, mu_y = filt(x), filt(y)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x2 = filt(x * x) - mu_x2
        sigma_y2 = filt(y * y) - mu_y2
        sigma_xy = filt(x * y) - mu_xy
        maps.append(((2 * mu_xy + C1) * (2 * sigma_xy + C2))
                    / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)))
    return float(np.mean(maps))


def test_window_is_separable_by_construction() -> None:
    """The 2D window must factorize, or the two-pass form is not the same
    operator at all."""
    from trainer.common.metrics import _WIN_1D

    assert np.allclose(_WIN_2D, np.outer(_WIN_1D, _WIN_1D), atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(64, 64, 3), (37, 53, 3), (48, 91, 1)])
def test_separable_matches_2d_correlation(shape) -> None:
    """Random images, including non-square and single-channel, and sizes small
    enough that the zero-padded border is a large fraction of the frame — which
    is exactly where a padding mismatch between the two forms would show up."""
    rng = np.random.default_rng(20260818)
    rendered = rng.random(shape)
    target = np.clip(rendered + rng.normal(0.0, 0.05, shape), 0.0, 1.0)

    assert ssim(rendered, target) == pytest.approx(
        _ssim_2d_reference(rendered, target), abs=1e-12)


def test_separable_matches_2d_on_degenerate_inputs() -> None:
    """Constant and identical images drive the SSIM denominators to their
    stabilizing constants; the two forms must still agree there."""
    flat = np.full((32, 32, 3), 0.5)
    assert ssim(flat, flat) == pytest.approx(_ssim_2d_reference(flat, flat),
                                             abs=1e-12)
    assert ssim(flat, flat) == pytest.approx(1.0, abs=1e-9)

    black = np.zeros((32, 32, 3))
    white = np.ones((32, 32, 3))
    assert ssim(black, white) == pytest.approx(_ssim_2d_reference(black, white),
                                              abs=1e-12)
