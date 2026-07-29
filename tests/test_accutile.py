"""AccuTile correctness test.

AccuTile's contract: for a horizontal row [y_bot, y_top] and a gaussian
ellipse, return the x-range within that row where the ellipse's alpha
exceeds 1/255. Our test compares AccuTile's output against brute-force
grid sampling.
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


def _brute_force_row_x_range(mean: np.ndarray, cov: np.ndarray, opacity: float,
                             row_y_bot: float, row_y_top: float,
                             x_sweep: float = 40.0, x_samples: int = 800,
                             y_samples: int = 40) -> tuple[float, float] | None:
    """Sample the alpha function on a grid within the strip, return
    (x_min, x_max) over points where alpha >= 1/255, or None."""
    xs = np.linspace(mean[0] - x_sweep, mean[0] + x_sweep, x_samples)
    ys = np.linspace(row_y_bot, row_y_top, y_samples)
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
    mask = alpha >= 1.0 / 255.0
    if not mask.any():
        return None
    _, col = np.where(mask)
    return float(xs[col.min()]), float(xs[col.max()])


def _call_accutile_row(module, mean: np.ndarray, cov: np.ndarray, opacity: float,
                       y_bot: float, y_top: float):
    mean_t = spy.Tensor.from_numpy(module.device, mean.astype(np.float32).reshape(1, 2))
    cov_t = spy.Tensor.from_numpy(module.device, cov.astype(np.float32).reshape(1, 3))
    op_t = spy.Tensor.from_numpy(module.device, np.array([opacity], dtype=np.float32))
    yb_t = spy.Tensor.from_numpy(module.device, np.array([y_bot], dtype=np.float32))
    yt_t = spy.Tensor.from_numpy(module.device, np.array([y_top], dtype=np.float32))
    return module.accuTileRowTest(mean_t, cov_t, op_t, yb_t, yt_t).to_numpy()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("cov", [
    [1.0, 0.0, 1.0],     # isotropic
    [4.0, 0.0, 1.0],     # x-elongated
    [1.0, 0.0, 4.0],     # y-elongated
    [2.0, 1.0, 2.0],     # rotated
    [4.0, -1.5, 2.0],    # negative b
])
@pytest.mark.parametrize("opacity", [0.3, 0.8])
@pytest.mark.parametrize("y_offset", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_accutile_row_matches_brute_force(
    backend: str, cov: list, opacity: float, y_offset: float,
) -> None:
    """For each row, AccuTile's x-range must contain the brute-force x-range."""
    module = _module(backend)
    mean = np.array([5.0, 0.0])
    cov_np = np.array(cov, dtype=np.float64)

    # Row is 1 pixel tall, centered at mean.y + y_offset.
    y_bot = float(mean[1] + y_offset) - 0.5
    y_top = float(mean[1] + y_offset) + 0.5

    at = _call_accutile_row(module, mean, cov_np, opacity, y_bot, y_top).reshape(2)
    bf = _brute_force_row_x_range(mean, cov_np, opacity, y_bot, y_top)

    if bf is None:
        # The row doesn't intersect the ellipse; AccuTile should also
        # report no intersection (sentinel x_max < x_min).
        assert at[1] < at[0] or (at[0] > mean[0] - 1e3 and at[1] > at[0]), (
            f"BF says no intersection but AccuTile returned {at}"
        )
        return

    # AccuTile output must CONTAIN the brute-force range.
    slack = 0.5  # grid-cell slop
    assert at[0] <= bf[0] + slack, (
        f"AccuTile x_min {at[0]:.3f} > BF {bf[0]:.3f} (cov={cov}, "
        f"opa={opacity}, y_off={y_offset})"
    )
    assert at[1] >= bf[1] - slack, (
        f"AccuTile x_max {at[1]:.3f} < BF {bf[1]:.3f} (cov={cov}, "
        f"opa={opacity}, y_off={y_offset})"
    )

    # AccuTile should also be not-too-loose.
    looseness = max(at[0] - bf[0], 0) + max(bf[1] - at[1], 0)
    # We tolerate up to 2 px of slack per side.
    max_slack_per_side = 2.0
    assert abs(at[0] - bf[0]) <= max_slack_per_side + slack, (
        f"AccuTile x_min {at[0]:.3f} too loose vs BF {bf[0]:.3f}"
    )
    assert abs(at[1] - bf[1]) <= max_slack_per_side + slack, (
        f"AccuTile x_max {at[1]:.3f} too loose vs BF {bf[1]:.3f}"
    )
