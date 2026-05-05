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

from trainer.device import create_device
from trainer.pipeline import DensifyConfig, Pipeline


BACKENDS = ["vulkan", "d3d12"]


def _setup(backend: str) -> Pipeline:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return Pipeline(device=device, width=32, height=32, max_splats=32)


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

    # Inject importance scores directly. With n_scoring_views=10 and
    # importance_thresh=5, we need score/10 > 5, so score > 50.
    cfg = DensifyConfig(
        n_scoring_views=10, importance_thresh=5.0,
        t_split_scale=0.01, t_opacity=0.005,
        t_big_scale=10.0,   # disable big-scale prune for this test
        growth_rate=1.0,     # no budget limit for this test
    )
    importance = np.zeros(pipeline.max_splats, dtype=np.uint32)
    importance[0] = 100  # high importance + small   -> CLONE
    importance[1] = 100  # high importance + large   -> SPLIT
    importance[2] = 100  # low opacity               -> PRUNE
    importance[3] = 10   # low importance            -> KEEP
    importance[4] = 100  # high importance + large   -> SPLIT
    pipeline.buffers.importance_score_accum.copy_from_numpy(importance)

    stats = pipeline.densify(cfg)

    assert stats["n_keep"]  == 1, f"expected 1 KEEP, got {stats}"
    assert stats["n_split"] == 2, f"expected 2 SPLIT, got {stats}"
    assert stats["n_clone"] == 1, f"expected 1 CLONE, got {stats}"
    assert stats["n_prune"] == 1, f"expected 1 PRUNE, got {stats}"

    # pre=5, 2 splits add 2 tail children, 1 clone adds 1 -> post_split=8.
    # mask drops 1 prune, final=7.
    assert stats["pre_count"] == 5
    assert stats["post_split_count"] == 8
    assert stats["final_count"] == 7, f"expected 7 after compaction, got {stats}"

    # Verify split children have reduced log_scale.
    post = _read_gaussians(pipeline)
    child_log_scale_expected = np.log(0.05 / 1.6)
    split_children = np.isclose(post[:, 3], child_log_scale_expected, atol=1e-3).sum()
    assert split_children >= 4, (
        f"expected at least 4 split children with log_scale≈{child_log_scale_expected:.3f}, "
        f"got {split_children} (log_scale_x values: {post[:, 3].round(3)})"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_densify_clone_halves_opacity(backend: str) -> None:
    """A small gaussian with high importance should be CLONED with halved
    opacity on both parent and clone."""
    pipeline = _setup(backend)

    pipeline.set_gaussians_from_numpy(
        means=np.array([[0.1, 0.2, 2.5]], dtype=np.float32),
        log_scales=np.log(np.array([[0.005, 0.005, 0.005]], dtype=np.float32)),
        rot_quats=np.array([[0.9659, 0.0, 0.2588, 0.0]], dtype=np.float32),
        opacities=np.array([0.7], dtype=np.float32),
        color_logits=np.array([[0.3, -0.5, 0.8]], dtype=np.float32),
    )

    cfg = DensifyConfig(
        n_scoring_views=10, importance_thresh=5.0,
        t_split_scale=0.01, t_opacity=0.005,
        t_big_scale=10.0, growth_rate=1.0,
    )
    importance = np.zeros(pipeline.max_splats, dtype=np.uint32)
    importance[0] = 100  # high importance + small scale -> CLONE
    pipeline.buffers.importance_score_accum.copy_from_numpy(importance)

    stats = pipeline.densify(cfg)
    assert stats["n_clone"] == 1
    assert stats["final_count"] == 2

    post = _read_gaussians(pipeline)
    # Both should have opacity logit corresponding to 0.7/2 = 0.35
    expected_logit = float(np.log(0.35 / (1.0 - 0.35)))
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
    stats = pipeline.densify(cfg)

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
    """With growth_rate=0.2 and 5 gaussians, budget=1. Only 1 of the 3
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
        n_scoring_views=10, importance_thresh=5.0,
        t_split_scale=0.01, t_opacity=0.005,
        t_big_scale=10.0,
        growth_rate=0.2,  # budget = max(1, int(0.2*5)) = 1
    )
    importance = np.zeros(pipeline.max_splats, dtype=np.uint32)
    importance[0] = 100
    importance[1] = 100
    importance[2] = 100
    importance[3] = 10   # below threshold
    importance[4] = 10   # below threshold
    pipeline.buffers.importance_score_accum.copy_from_numpy(importance)

    stats = pipeline.densify(cfg)

    # 3 candidates compete for budget=1. Only 1 wins.
    assert stats["n_split"] + stats["n_clone"] == 1, (
        f"expected 1 growth action, got split={stats['n_split']} clone={stats['n_clone']}"
    )
    assert stats["n_keep"] == 4  # 2 low-importance + 2 budget-denied
