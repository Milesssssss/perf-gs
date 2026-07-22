"""P0 SPIKE (throwaway): validate the generic-primitive mechanism the refactor
depends on.

Two primitives (Gaussian-like, disk-like) conform to one ``[Differentiable]``
Slang interface and are evaluated + differentiated through ONE generic core
(``sampleGeneric<S>``), instantiated by per-type compute entry points. We check:
  1. Forward matches a numpy reference (the generic call produces correct math).
  2. The autodiff gradient (``bwd_diff`` through the generic) matches central
     finite differences.
on BOTH backends. If this passes, the plan's "monomorphized generic core,
swap-the-type" approach is proven on this slang/slang-rhi stack.

Delete once P2 lands the real IPrimitive core.
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from trainer.device import SHADERS_DIR, create_device

BACKENDS = ["vulkan", "d3d12"]
SPIKE = str(SHADERS_DIR / "_spike" / "generic_spike.slang")


def _kernel(backend: str, entry: str):
    try:
        device = create_device(backend=backend)
    except Exception as exc:  # pragma: no cover - backend availability
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    program = device.load_program(SPIKE, [entry])
    return device, device.create_compute_kernel(program)


def _run(device, kernel, params: np.ndarray, pixels: np.ndarray):
    n = params.shape[0]
    p = spy.NDBuffer(device=device, dtype="float", shape=(n * 4,))
    p.copy_from_numpy(params.reshape(-1).astype(np.float32))
    px = spy.NDBuffer(device=device, dtype="float", shape=(n * 2,))
    px.copy_from_numpy(pixels.reshape(-1).astype(np.float32))
    out = spy.NDBuffer(device=device, dtype="float", shape=(n,))
    grad = spy.NDBuffer(device=device, dtype="float", shape=(n * 4,))
    kernel.dispatch(
        thread_count=[n, 1, 1],
        params=p.storage, pixels=px.storage,
        outAlpha=out.storage, outGrad=grad.storage, n=n,
    )
    device.wait_for_idle()
    a = np.frombuffer(out.to_numpy().tobytes(), dtype=np.float32)[:n]
    g = np.frombuffer(grad.to_numpy().tobytes(), dtype=np.float32)[: n * 4].reshape(n, 4)
    return a, g


def _sig(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _gauss_fwd(pr: np.ndarray, px: np.ndarray) -> float:
    mean = pr[:2]
    s = np.exp(pr[2])
    d = px - mean
    power = -0.5 * (d[0] ** 2 + d[1] ** 2) / (s * s)
    return _sig(pr[3]) * np.exp(power)


def _disk_fwd(pr: np.ndarray, px: np.ndarray) -> float:
    mean = pr[:2]
    r = np.exp(pr[2])
    dist = np.hypot(px[0] - mean[0], px[1] - mean[1])
    t = np.clip(1.0 - dist / r, 0.0, 1.0)
    fall = t * t * (3.0 - 2.0 * t)
    return _sig(pr[3]) * fall


def _fd_grad(fwd, pr: np.ndarray, px: np.ndarray, h: float = 2e-3) -> np.ndarray:
    g = np.zeros(4)
    for k in range(4):
        pp = pr.copy(); pp[k] += h; fp = fwd(pp, px)
        pp = pr.copy(); pp[k] -= h; fm = fwd(pp, px)
        g[k] = (fp - fm) / (2 * h)
    return g


@pytest.mark.parametrize("backend", BACKENDS)
def test_spike_gauss(backend: str) -> None:
    device, kernel = _kernel(backend, "gaussEval")
    rng = np.random.default_rng(0)
    n = 8
    mean = rng.uniform(-1, 1, (n, 2))
    logs = rng.uniform(-1, 0, (n,))
    op = rng.uniform(-1, 1, (n,))
    pixels = mean + rng.uniform(-0.5, 0.5, (n, 2))
    params = np.concatenate([mean, logs[:, None], op[:, None]], axis=1)

    a, g = _run(device, kernel, params, pixels)
    for i in range(n):
        assert np.isclose(a[i], _gauss_fwd(params[i], pixels[i]), atol=1e-5, rtol=1e-4)
        gfd = _fd_grad(_gauss_fwd, params[i], pixels[i])
        np.testing.assert_allclose(g[i], gfd, atol=2e-3, rtol=3e-2,
                                   err_msg=f"gauss grad mismatch i={i}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_spike_disk(backend: str) -> None:
    device, kernel = _kernel(backend, "diskEval")
    rng = np.random.default_rng(1)
    n = 8
    mean = rng.uniform(-1, 1, (n, 2))
    logr = rng.uniform(-0.5, 0.5, (n,))
    op = rng.uniform(-1, 1, (n,))
    ang = rng.uniform(0, 2 * np.pi, (n,))
    r = np.exp(logr)
    pixels = mean + np.stack([np.cos(ang), np.sin(ang)], axis=1) * (0.5 * r)[:, None]
    params = np.concatenate([mean, logr[:, None], op[:, None]], axis=1)

    a, g = _run(device, kernel, params, pixels)
    for i in range(n):
        assert np.isclose(a[i], _disk_fwd(params[i], pixels[i]), atol=1e-5, rtol=1e-4)
        gfd = _fd_grad(_disk_fwd, params[i], pixels[i])
        np.testing.assert_allclose(g[i], gfd, atol=3e-3, rtol=4e-2,
                                   err_msg=f"disk grad mismatch i={i}")
