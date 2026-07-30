"""GPU-vs-CPU parity for the multi-scale structure-tensor field.

``structure_analysis_gpu.StructureTensorGPU`` ports the scipy reference
(``structure_analysis.compute_structure_tensor_field``) to a chain of Slang
compute passes. This asserts the GPU output matches the CPU oracle on small
images (random texture + analytic gratings), within float32 tolerance, so the
downstream eta metric sees the same field regardless of which path produced it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trainer.backend.compute.device import create_device
from trainer.backend.densify.structure_analysis import (
    compute_structure_tensor_field,
    principal_eigenvalue,
)
from trainer.backend.densify.structure_analysis_gpu import StructureTensorGPU

BACKENDS = ["vulkan", "d3d12"]


def _device(backend: str):
    try:
        return create_device(backend=backend)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"backend {backend!r} unavailable: {exc}")


def _grating(h: int, w: int, omega: float, axis: str) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    coord = xx if axis == "x" else yy
    val = 0.5 + 0.5 * np.sin(2.0 * np.pi * omega * coord)
    return np.stack([val, val, val], axis=-1).astype(np.float32)


def _assert_field_close(cpu: np.ndarray, gpu: np.ndarray) -> None:
    assert gpu.shape == cpu.shape
    assert gpu.dtype == np.float32
    # The field spans several orders of magnitude (freq_sq weighting), so use a
    # combined relative+absolute bound keyed off the field scale. The only
    # source of divergence is float32 (GPU) vs float64 (CPU) accumulation in the
    # blur/Sobel chain, which is small everywhere.
    scale = float(np.abs(cpu).max()) + 1e-8
    diff = np.abs(gpu - cpu)
    max_abs = float(diff.max())
    assert max_abs < 5e-3 * scale, (
        f"max abs diff {max_abs:.3e} exceeds {5e-3 * scale:.3e} (scale {scale:.3e})")
    np.testing.assert_allclose(gpu, cpu, rtol=2e-2, atol=1e-3 * scale)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpu_matches_cpu_random(backend: str) -> None:
    device = _device(backend)
    rng = np.random.default_rng(0)
    W, H = 48, 40
    img = rng.random((H, W, 3), dtype=np.float32)

    cpu = compute_structure_tensor_field(img)
    gpu = StructureTensorGPU(device, W, H).compute(img)
    _assert_field_close(cpu, gpu)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("axis", ["x", "y"])
def test_gpu_matches_cpu_grating(backend: str, axis: str) -> None:
    device = _device(backend)
    W, H = 64, 64
    img = _grating(H, W, omega=0.1, axis=axis)

    cpu = compute_structure_tensor_field(img)
    gpu = StructureTensorGPU(device, W, H).compute(img)
    _assert_field_close(cpu, gpu)

    # Principal eigenvalue (the eta driver) must also agree on the interior.
    lam_cpu = principal_eigenvalue(cpu)[16:48, 16:48]
    lam_gpu = principal_eigenvalue(gpu)[16:48, 16:48]
    np.testing.assert_allclose(lam_gpu, lam_cpu, rtol=3e-2,
                               atol=1e-3 * (float(np.abs(lam_cpu).max()) + 1e-8))


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpu_instance_reuse(backend: str) -> None:
    """One instance must produce correct fields for successive images (the
    scene-preprocessing use case: textures/kernels reused across views)."""
    device = _device(backend)
    W, H = 40, 32
    st = StructureTensorGPU(device, W, H)
    for omega in (0.05, 0.15):
        img = _grating(H, W, omega=omega, axis="x")
        cpu = compute_structure_tensor_field(img)
        gpu = st.compute(img)
        _assert_field_close(cpu, gpu)
