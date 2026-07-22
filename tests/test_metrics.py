"""Sanity checks for trainer.metrics (the shared PSNR/SSIM primitives).

Pure numpy/scipy — no GPU backend needed, so these run fast and lock the
metric semantics the trainer + plotting scripts depend on.
"""

from __future__ import annotations

import numpy as np

from trainer.metrics import psnr, ssim


def test_psnr_identical_is_capped() -> None:
    a = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    assert psnr(a, a) == 99.0


def test_psnr_known_mse() -> None:
    a = np.zeros((4, 4, 3), np.float32)
    b = np.full((4, 4, 3), 0.1, np.float32)   # mse = 0.01
    assert abs(psnr(a, b) - 20.0) < 1e-6       # -10*log10(0.01) == 20


def test_ssim_identical_is_one() -> None:
    a = np.random.default_rng(1).random((16, 16, 3)).astype(np.float32)
    assert abs(ssim(a, a) - 1.0) < 1e-6


def test_ssim_symmetric() -> None:
    rng = np.random.default_rng(2)
    a = rng.random((16, 16, 3)).astype(np.float32)
    b = rng.random((16, 16, 3)).astype(np.float32)
    assert abs(ssim(a, b) - ssim(b, a)) < 1e-9
