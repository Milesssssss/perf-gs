# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Cross-validate the transposed per-splat backward against the proven
per-pixel backward.

Both kernels must produce the same per-gaussian gradients (within
fixed-point + atomic-order noise) for the same scene. This is a stronger,
more direct correctness gate than the finite-difference test: it compares
the two analytic kernels head-to-head over a scene deep enough to exercise
multiple checkpoint buckets (>32 overlapping splats in a tile) and the
early-break mask.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.config.config import LossConfig
from trainer.data.dataset import simple_camera
from trainer.backend.compute.device import create_device

BACKENDS = ["vulkan", "d3d12"]

GRAD_NAMES = [
    "mean_x", "mean_y", "mean_z",
    "log_scale_x", "log_scale_y", "log_scale_z",
    "rot_w", "rot_x", "rot_y", "rot_z",
    "opacity", "color_r", "color_g", "color_b",
]


def _deep_scene(n: int, seed: int):
    rng = np.random.default_rng(seed)
    means = np.zeros((n, 3), dtype=np.float32)
    means[:, 0] = rng.uniform(-0.2, 0.2, n)
    means[:, 1] = rng.uniform(-0.2, 0.2, n)
    means[:, 2] = rng.uniform(2.0, 4.0, n)
    log_scales = np.log(np.full((n, 3), 0.35, dtype=np.float32))
    rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    opac = rng.uniform(0.3, 0.8, n).astype(np.float32)
    colors = rng.uniform(-2.0, 2.0, (n, 3)).astype(np.float32)
    return {
        "means": means, "log_scales": log_scales, "rot_quats": rot,
        "opacities": opac, "color_logits": colors,
    }


def _target(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= max(h - 1, 1)
    xx /= max(w - 1, 1)
    return np.stack([xx, yy, (1.0 - xx) * yy], axis=-1).astype(np.float32)


def _grads_with(pipeline: TestPipeline, scene, camera, use_persplat: bool,
                use_flat: bool = False, use_warp_forward: bool = False):
    pipeline.set_primitive_from_numpy(**scene)
    pipeline.use_persplat_backward = use_persplat
    pipeline.use_flat_bucket_backward = use_flat
    pipeline.use_warp_per_pixel_forward = use_warp_forward
    pipeline.forward_and_loss(camera)
    pipeline.optimizer.zero_grad()
    pipeline.run_backward()
    pipeline.device.wait_for_idle()
    return pipeline.read_gradients()


@pytest.mark.parametrize("backend", BACKENDS)
def test_persplat_matches_perpixel(backend: str) -> None:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    # 64x64 so the central tiles are deep; 100 clustered splats -> multiple
    # buckets per pixel and early-break in the densest tiles.
    W = H = 64
    n = 100
    pipeline = TestPipeline(device=device, width=W, height=H, max_splats=128,
                            loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0))

    scene = _deep_scene(n, seed=3)
    pipeline.targets.set_image(_target(H, W))
    camera = simple_camera(width=W, height=H, eye=(0.0, 0.0, 0.0),
                           target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                           fovy_degrees=60.0)

    g_ref = _grads_with(pipeline, scene, camera, use_persplat=False)
    g_new = _grads_with(pipeline, scene, camera, use_persplat=True)

    # Compare per component. Tolerance accounts for fp16 checkpoints +
    # atomic-order + the different summation order between the two kernels.
    mismatches = []
    for name in GRAD_NAMES:
        a = g_ref[name]
        b = g_new[name]
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
        # Relative error with an absolute floor; scale tolerance to the
        # magnitude of the per-component gradient.
        scale = max(1e-6, float(np.max(np.abs(a))))
        abs_err = np.abs(a - b)
        rel_err = abs_err / scale
        worst = int(np.argmax(rel_err))
        if rel_err[worst] > 0.02 and abs_err[worst] > 1e-4:
            mismatches.append(
                f"{name}[g{worst}]: ref={a[worst]:.6e} new={b[worst]:.6e} "
                f"rel={rel_err[worst]:.3f}")

    if mismatches:
        pytest.fail(f"{len(mismatches)} gradient mismatches between per-pixel "
                    f"and per-splat backward:\n" + "\n".join(mismatches))


@pytest.mark.parametrize("backend", BACKENDS)
def test_flat_bucket_matches_persplat(backend: str) -> None:
    """The flat (warp-per-bucket) backward must produce the same per-gaussian
    gradients as the per-tile per-splat backward. Same fused blend math and
    fixed-point atomics; only the work assignment (one warp per LIVE bucket via
    the compacted list) and the inline per-pixel reads differ, so the two
    should agree within fixed-point / atomic-order noise."""
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    # Same deep 64x64 scene: dense central tiles span multiple buckets and
    # early-break, exercising the live-bucket compaction + saturation skip.
    W = H = 64
    n = 100
    pipeline = TestPipeline(device=device, width=W, height=H, max_splats=128,
                            loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0))

    scene = _deep_scene(n, seed=3)
    pipeline.targets.set_image(_target(H, W))
    camera = simple_camera(width=W, height=H, eye=(0.0, 0.0, 0.0),
                           target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                           fovy_degrees=60.0)

    g_ref = _grads_with(pipeline, scene, camera, use_persplat=True, use_flat=False)
    g_new = _grads_with(pipeline, scene, camera, use_persplat=True, use_flat=True)

    mismatches = []
    for name in GRAD_NAMES:
        a = g_ref[name]
        b = g_new[name]
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
        scale = max(1e-6, float(np.max(np.abs(a))))
        abs_err = np.abs(a - b)
        rel_err = abs_err / scale
        worst = int(np.argmax(rel_err))
        if rel_err[worst] > 0.02 and abs_err[worst] > 1e-4:
            mismatches.append(
                f"{name}[g{worst}]: persplat={a[worst]:.6e} flat={b[worst]:.6e} "
                f"rel={rel_err[worst]:.3f}")

    if mismatches:
        pytest.fail(f"{len(mismatches)} gradient mismatches between per-tile "
                    f"per-splat and flat warp-per-bucket backward:\n"
                    + "\n".join(mismatches))


@pytest.mark.parametrize("backend", BACKENDS)
def test_warp_forward_backward_matches_baseline(backend: str) -> None:
    """End-to-end gradient check: the warp-per-pixel forward writes its own
    checkpoints + gaussian_count, which the backward then reads. Gradients
    must match the thread-per-pixel forward feeding the same backward (only
    the per-bucket vs per-splat early break differs -> sub-LSB)."""
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    W = H = 64
    n = 100
    pipeline = TestPipeline(device=device, width=W, height=H, max_splats=128,
                            loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0))

    scene = _deep_scene(n, seed=3)
    pipeline.targets.set_image(_target(H, W))
    camera = simple_camera(width=W, height=H, eye=(0.0, 0.0, 0.0),
                           target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                           fovy_degrees=60.0)

    g_ref = _grads_with(pipeline, scene, camera, use_persplat=True,
                        use_flat=False, use_warp_forward=False)
    g_new = _grads_with(pipeline, scene, camera, use_persplat=True,
                        use_flat=False, use_warp_forward=True)

    mismatches = []
    for name in GRAD_NAMES:
        a = g_ref[name]
        b = g_new[name]
        assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
        scale = max(1e-6, float(np.max(np.abs(a))))
        abs_err = np.abs(a - b)
        rel_err = abs_err / scale
        worst = int(np.argmax(rel_err))
        if rel_err[worst] > 0.02 and abs_err[worst] > 1e-4:
            mismatches.append(
                f"{name}[g{worst}]: threadfwd={a[worst]:.6e} "
                f"warpfwd={b[worst]:.6e} rel={rel_err[worst]:.3f}")

    if mismatches:
        pytest.fail(f"{len(mismatches)} gradient mismatches between thread-per-"
                    f"pixel and warp-per-pixel forward feeding the backward:\n"
                    + "\n".join(mismatches))


@pytest.mark.parametrize("backend", BACKENDS)
def test_projection_backward_resets_the_seeds(backend: str) -> None:
    """projection_backward consumes the 2D seeds and leaves them zero, so a
    second backward without any clear yields the same geometry gradients."""
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    W = H = 64
    pipeline = TestPipeline(device=device, width=W, height=H, max_splats=128,
                            loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0))
    pipeline.targets.set_image(_target(H, W))
    camera = simple_camera(width=W, height=H, eye=(0.0, 0.0, 0.0),
                           target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                           fovy_degrees=60.0)

    first = _grads_with(pipeline, _deep_scene(100, seed=3), camera,
                        use_persplat=True, use_flat=True)
    for name, buffer in pipeline.buffers.gradients.seeds.items():
        seeds = np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.int64)
        assert not seeds.any(), f"seed {name} not reset by projection_backward"

    pipeline.forward_and_loss(camera)
    pipeline.run_backward()
    pipeline.device.wait_for_idle()
    second = pipeline.read_gradients()
    assert np.any(first["mean_x"] != 0.0) and np.any(first["log_scale_x"] != 0.0)
    for name in GRAD_NAMES[:10]:   # geometry grads: stored, not accumulated
        np.testing.assert_array_equal(second[name], first[name], err_msg=name)
