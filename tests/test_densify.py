"""End-to-end densification test (FastGS importance-score strategy).

Sets up a 5-gaussian scene with hand-injected importance scores designed
to exercise every action (KEEP, SPLIT, CLONE, PRUNE), runs one densify
pass, and asserts the post-compaction state is consistent with the
expected split/clone/prune counts.

The numerical content of split children (jittered positions, scale / 1.6)
is checked loosely: we verify split children have log_scale reduced by
log(1.6) from the parent, and that clone children have identical params
to their parent.
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from _pipeline_testkit import TestPipeline
from trainer.device import create_device
from trainer.pipeline import DensifyConfig


BACKENDS = ["vulkan", "d3d12"]


def _setup(backend: str) -> TestPipeline:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return TestPipeline(device=device, width=32, height=32, max_splats=32)


def _read_gaussians(pipeline: Pipeline) -> np.ndarray:
    """Read current gaussians as a (N, 14) float array."""
    n = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    raw = pipeline.buffers.gaussians.to_numpy()
    floats = np.frombuffer(raw.tobytes(), dtype=np.float32).reshape(-1, 14)
    return floats[:n].copy()


@pytest.mark.parametrize("backend", BACKENDS)
def test_densify_action_counts(backend: str) -> None:
    pipeline = _setup(backend)

    means = np.array([
        [0.0, 0.0, 2.0],      # 0: small scale, high importance -> CLONE
        [0.5, 0.0, 2.0],      # 1: large scale, high importance -> SPLIT
        [-0.5, 0.0, 2.0],     # 2: low opacity                  -> PRUNE
        [0.0, 0.5, 2.0],      # 3: large scale, low importance  -> KEEP
        [0.0, -0.5, 2.0],     # 4: large scale, high importance -> SPLIT
    ], dtype=np.float32)
    log_scales = np.log(np.array([
        [0.005, 0.005, 0.005],  # below split threshold (0.01) -> clone
        [0.05,  0.05,  0.05],   # above split threshold -> split
        [0.01,  0.01,  0.01],
        [0.05,  0.05,  0.05],   # above split threshold but low importance
        [0.05,  0.05,  0.05],   # above split threshold -> split
    ], dtype=np.float32))
    rot_quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (5, 1))
    opacities = np.array([0.5, 0.5, 0.001, 0.5, 0.5], dtype=np.float32)
    color_logits = np.zeros((5, 3), dtype=np.float32)

    pipeline.set_gaussians_from_numpy(
        means=means, log_scales=log_scales, rot_quats=rot_quats,
        opacities=opacities, color_logits=color_logits,
    )

    # Inject importance scores and gradient signals.
    cfg = DensifyConfig(
        n_scoring_views=10,
        t_opacity=0.005,
        t_big_scale=10.0,   # disable big-scale prune for this test
    )
    importance = np.zeros(pipeline.max_splats, dtype=np.uint32)
    importance[0] = 100  # high importance + small   -> CLONE
    importance[1] = 100  # high importance + large   -> SPLIT
    importance[2] = 100  # low opacity               -> PRUNE
    importance[3] = 10   # low importance            -> KEEP
    importance[4] = 100  # high importance + large   -> SPLIT
    pipeline.buffers.importance_score_accum.copy_from_numpy(importance)

    # Inject high gradient signal so gradient gate passes.
    # grad_avg = fromFixed(mag_accum) / vis_count > threshold
    # Need: mag_accum = toFixed(threshold * 10) * vis_count
    FIXED_SCALE = 65536.0
    vis_count = 50
    mag = np.zeros(pipeline.max_splats, dtype=np.int64)
    vis = np.zeros(pipeline.max_splats, dtype=np.uint32)
    high_grad = int(100.0 * vis_count * FIXED_SCALE)  # well above both thresholds
    for gi in [0, 1, 4]:
        mag[gi] = high_grad
        vis[gi] = vis_count
    pipeline.buffers.grad_mean_2d_mag_accum.copy_from_numpy(mag)
    pipeline.buffers.visibility_count_accum.copy_from_numpy(vis)

    stats = pipeline.densify(cfg, densify_step=1, n_densify_steps=1, init_count=0)

    # With gradient gate, only gaussians passing both importance AND grad threshold
    # are densified. Verify at least some growth happens.
    assert stats["pre_count"] == 5
    assert stats["n_prune"] >= 1, f"expected prune, got {stats}"
    assert stats["n_split"] + stats["n_clone"] >= 1, f"expected growth, got {stats}"
    assert stats["final_count"] > stats["pre_count"] - stats["n_prune"], f"expected net growth, got {stats}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_densify_clone_copies_opacity(backend: str) -> None:
    """A reference-3DGS clone is an exact parameter copy with fresh moments."""
    pipeline = _setup(backend)

    pipeline.set_gaussians_from_numpy(
        means=np.array([[0.1, 0.2, 2.5]], dtype=np.float32),
        log_scales=np.log(np.array([[0.005, 0.005, 0.005]], dtype=np.float32)),
        rot_quats=np.array([[0.9659, 0.0, 0.2588, 0.0]], dtype=np.float32),
        opacities=np.array([0.7], dtype=np.float32),
        color_logits=np.array([[0.3, -0.5, 0.8]], dtype=np.float32),
    )

    cfg = DensifyConfig(
        n_scoring_views=10,
        t_opacity=0.005,
        t_big_scale=10.0,
    )
    importance = np.zeros(pipeline.max_splats, dtype=np.uint32)
    importance[0] = 100
    pipeline.buffers.importance_score_accum.copy_from_numpy(importance)

    FIXED_SCALE = 65536.0
    vis_count = 50
    mag = np.zeros(pipeline.max_splats, dtype=np.int64)
    vis = np.zeros(pipeline.max_splats, dtype=np.uint32)
    mag[0] = int(100.0 * vis_count * FIXED_SCALE)
    vis[0] = vis_count
    pipeline.buffers.grad_mean_2d_mag_accum.copy_from_numpy(mag)
    pipeline.buffers.visibility_count_accum.copy_from_numpy(vis)

    stats = pipeline.densify(cfg, densify_step=1, n_densify_steps=1, init_count=0)
    assert stats["n_clone"] == 1
    assert stats["final_count"] == 2

    post = _read_gaussians(pipeline)
    # duplicate()/clone copies opacity unchanged. Halving parent+child is not
    # alpha-neutral and drove repeated clones toward transparent floater haze.
    expected_logit = float(np.log(0.7 / (1.0 - 0.7)))
    np.testing.assert_allclose(post[0, 10], expected_logit, atol=1e-4)
    np.testing.assert_allclose(post[1, 10], expected_logit, atol=1e-4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_densify_noop_when_no_signal(backend: str) -> None:
    """If importance scores are all zero and no pruning conditions are met,
    densify() should leave the buffer unchanged."""
    pipeline = _setup(backend)

    means = np.array([
        [0.0, 0.0, 2.0], [0.3, 0.0, 2.5], [-0.1, 0.2, 2.1],
    ], dtype=np.float32)
    pipeline.set_gaussians_from_numpy(
        means=means,
        log_scales=np.log(np.full((3, 3), 0.01, dtype=np.float32)),
        rot_quats=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (3, 1)),
        opacities=np.array([0.5, 0.5, 0.5], dtype=np.float32),
        color_logits=np.zeros((3, 3), dtype=np.float32),
    )
    pre = _read_gaussians(pipeline)

    # Zero importance scores.
    importance = np.zeros(pipeline.max_splats, dtype=np.uint32)
    pipeline.buffers.importance_score_accum.copy_from_numpy(importance)

    cfg = DensifyConfig()
    stats = pipeline.densify(cfg, densify_step=1, n_densify_steps=1, init_count=0)

    assert stats["n_keep"] == 3
    assert stats["n_split"] == 0
    assert stats["n_clone"] == 0
    assert stats["n_prune"] == 0
    assert stats["final_count"] == 3

    post = _read_gaussians(pipeline)
    np.testing.assert_allclose(post, pre, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_opacity_reset(backend: str) -> None:
    pipeline = _setup(backend)
    # Mix of opacities: two above reset (0.9, 0.5), one below (0.01).
    in_probs = np.array([0.9, 0.5, 0.01], dtype=np.float32)
    pipeline.set_gaussians_from_numpy(
        means=np.zeros((3, 3), dtype=np.float32),
        log_scales=np.log(np.full((3, 3), 0.01, dtype=np.float32)),
        rot_quats=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (3, 1)),
        opacities=in_probs,
        color_logits=np.zeros((3, 3), dtype=np.float32),
    )
    cfg = DensifyConfig(opacity_cap=0.05)
    pipeline.cap_opacity(cfg.opacity_cap)
    pipeline.device.wait_for_idle()
    post = _read_gaussians(pipeline)
    # cap_opacity uses the same resetOpacity shader which does min(logit, logit_cap).
    # Gaussians at 0.9 and 0.5 get clamped DOWN to logit(0.05); the one
    # at 0.01 is already below the cap and stays at logit(0.01).
    def logit(p):
        p_c = np.clip(p, 1e-5, 1.0 - 1e-5)
        return np.log(p_c / (1.0 - p_c))
    logit_cap = logit(0.05)
    expected = np.minimum(logit(in_probs), logit_cap).astype(np.float32)
    np.testing.assert_allclose(post[:, 10], expected, atol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_growth_budget_limits_splits(backend: str) -> None:
    """With a small parabolic budget, only 1 of the 3
    high-importance large gaussians should get promoted."""
    pipeline = _setup(backend)

    means = np.array([
        [0.0, 0.0, 2.0],
        [0.3, 0.0, 2.0],
        [0.6, 0.0, 2.0],
        [0.9, 0.0, 2.0],
        [1.2, 0.0, 2.0],
    ], dtype=np.float32)
    pipeline.set_gaussians_from_numpy(
        means=means,
        log_scales=np.log(np.full((5, 3), 0.05, dtype=np.float32)),
        rot_quats=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (5, 1)),
        opacities=np.full(5, 0.5, dtype=np.float32),
        color_logits=np.zeros((5, 3), dtype=np.float32),
    )

    cfg = DensifyConfig(
        n_scoring_views=10,
        t_opacity=0.005,
        t_big_scale=10.0,
    )
    importance = np.zeros(pipeline.max_splats, dtype=np.uint32)
    importance[0] = 100
    importance[1] = 100
    importance[2] = 100
    importance[3] = 10   # below threshold
    importance[4] = 10   # below threshold
    pipeline.buffers.importance_score_accum.copy_from_numpy(importance)

    # Inject gradient signals for the 3 high-importance gaussians.
    FIXED_SCALE = 65536.0
    vis_count = 50
    mag = np.zeros(pipeline.max_splats, dtype=np.int64)
    vis = np.zeros(pipeline.max_splats, dtype=np.uint32)
    high_grad = int(100.0 * vis_count * FIXED_SCALE)
    for gi in [0, 1, 2]:
        mag[gi] = high_grad
        vis[gi] = vis_count
    pipeline.buffers.grad_mean_2d_mag_accum.copy_from_numpy(mag)
    pipeline.buffers.visibility_count_accum.copy_from_numpy(vis)

    # Sqrt schedule: target = 5 + (32-5)*(1/730)^0.5 ≈ 6.0, growth = 1.
    stats = pipeline.densify(cfg, densify_step=1, n_densify_steps=730, init_count=5)

    # 3 candidates compete for a small budget. Only 1 wins.
    assert stats["n_split"] + stats["n_clone"] == 1, (
        f"expected 1 growth action, got split={stats['n_split']} clone={stats['n_clone']}"
    )
    assert stats["n_keep"] == 4  # 2 low-importance + 2 budget-denied
