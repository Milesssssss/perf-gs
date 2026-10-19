# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Unit tests for Slang math utilities via SlangPy's functional API.

These lock in correctness of ``sigmoid``, ``matrixFromQuaternion``,
``buildCovariance3D``, and ``evalGaussian2DFromCov`` — the pieces the
backward pass will differentiate through in v1. Running on both backends
catches driver-level regressions early.
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from trainer.backend.compute.device import SHADERS_DIR, create_device


BACKENDS = ["vulkan", "d3d12"]


def _load_math_module(backend: str) -> spy.Module:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return spy.Module.load_from_file(
        device, str(SHADERS_DIR / "common" / "math_tests.slang")
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_sigmoid(backend: str) -> None:
    module = _load_math_module(backend)
    xs = np.array([-10.0, -1.0, 0.0, 1.0, 10.0], dtype=np.float32)
    expected = 1.0 / (1.0 + np.exp(-xs))

    tensor = spy.Tensor.from_numpy(module.device, xs)
    result = module.sigmoidScalar(tensor).to_numpy()
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_quat_identity(backend: str) -> None:
    module = _load_math_module(backend)
    q = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # (w, x, y, z) identity
    tensor = spy.Tensor.from_numpy(module.device, q)
    result = module.rotMatFromQuat(tensor).to_numpy()[0]
    np.testing.assert_allclose(result, np.eye(3, dtype=np.float32), atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_quat_90deg_z(backend: str) -> None:
    module = _load_math_module(backend)
    # 90° rotation about +Z: q = (cos(45°), 0, 0, sin(45°)) in (w, x, y, z)
    half = np.sqrt(0.5).astype(np.float32)
    q = np.array([[half, 0.0, 0.0, half]], dtype=np.float32)
    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=np.float32)
    tensor = spy.Tensor.from_numpy(module.device, q)
    result = module.rotMatFromQuat(tensor).to_numpy()[0]
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cov_identity(backend: str) -> None:
    module = _load_math_module(backend)
    scale = spy.Tensor.from_numpy(module.device, np.array([[1.0, 1.0, 1.0]], dtype=np.float32))
    quat = spy.Tensor.from_numpy(module.device, np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    result = module.covFromScaleQuat(scale, quat).to_numpy()[0]
    np.testing.assert_allclose(result, np.eye(3, dtype=np.float32), atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cov_anisotropic(backend: str) -> None:
    module = _load_math_module(backend)
    scale = spy.Tensor.from_numpy(module.device, np.array([[2.0, 1.0, 1.0]], dtype=np.float32))
    quat = spy.Tensor.from_numpy(module.device, np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    expected = np.diag([4.0, 1.0, 1.0]).astype(np.float32)
    result = module.covFromScaleQuat(scale, quat).to_numpy()[0]
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gaussian2d_at_mean(backend: str) -> None:
    """alpha at the mean of an identity-covariance gaussian is ``opacity``."""
    module = _load_math_module(backend)
    pixel = spy.Tensor.from_numpy(module.device, np.array([[0.0, 0.0]], dtype=np.float32))
    mean = spy.Tensor.from_numpy(module.device, np.array([[0.0, 0.0]], dtype=np.float32))
    cov = spy.Tensor.from_numpy(module.device, np.array([[1.0, 0.0, 1.0]], dtype=np.float32))
    opacity = spy.Tensor.from_numpy(module.device, np.array([1.0], dtype=np.float32))
    result = module.evalGaussian2D(pixel, mean, cov, opacity).to_numpy()[0]
    np.testing.assert_allclose(result, 1.0, atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gaussian2d_one_sigma(backend: str) -> None:
    """alpha 1 unit out along x of identity-cov gaussian = ``opacity * exp(-0.5)``."""
    module = _load_math_module(backend)
    pixel = spy.Tensor.from_numpy(module.device, np.array([[1.0, 0.0]], dtype=np.float32))
    mean = spy.Tensor.from_numpy(module.device, np.array([[0.0, 0.0]], dtype=np.float32))
    cov = spy.Tensor.from_numpy(module.device, np.array([[1.0, 0.0, 1.0]], dtype=np.float32))
    opacity = spy.Tensor.from_numpy(module.device, np.array([1.0], dtype=np.float32))
    result = module.evalGaussian2D(pixel, mean, cov, opacity).to_numpy()[0]
    np.testing.assert_allclose(result, np.exp(-0.5), atol=1e-5)
