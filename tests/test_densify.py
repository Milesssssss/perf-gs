"""End-to-end densification test.

Sets up a 5-gaussian scene with hand-injected tracking signals designed
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
    # max_splats small so the action_buffer readback in densify() stays fast.
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
        [0.0, 0.0, 2.0],      # 0: small scale, high grad  -> CLONE
        [0.5, 0.0, 2.0],      # 1: large scale, high grad  -> SPLIT
        [-0.5, 0.0, 2.0],     # 2: low opacity             -> PRUNE
        [0.0, 0.5, 2.0],      # 3: small scale, low grad   -> KEEP
        [0.0, -0.5, 2.0],     # 4: huge scale              -> PRUNE
    ], dtype=np.float32)
    log_scales = np.log(np.array([
        [0.005, 0.005, 0.005],  # below split threshold (0.01)
        [0.05,  0.05,  0.05],   # above split threshold
        [0.01,  0.01,  0.01],
        [0.005, 0.005, 0.005],
        [1.0,   1.0,   1.0],    # above big-scale threshold (0.5)
    ], dtype=np.float32))
    rot_quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (5, 1))
    opacities = np.array([0.5, 0.5, 0.001, 0.5, 0.5], dtype=np.float32)
    color_logits = np.zeros((5, 3), dtype=np.float32)

    pipeline.set_gaussians_from_numpy(
        means=means, log_scales=log_scales, rot_quats=rot_quats,
        opacities=opacities, color_logits=color_logits,
    )

    # Inject tracking signals directly: high mag for 0 and 1, low for others.
    # Normalization: mag_avg = mag_total / visibility_count. Aim for
    # mag_avg = 10 * t_grad with visibility_count = 50 iters.
    mag = np.zeros(pipeline.max_splats, dtype=np.int32)
    vis = np.zeros(pipeline.max_splats, dtype=np.uint32)
    cfg = DensifyConfig(
        t_grad=2e-4, t_split_scale=0.01, t_opacity=0.005, t_big_scale=0.5,
    )
    vis_count = 50
    high_mag_fixed = int(cfg.t_grad * 10.0 * vis_count * 65536.0)
    mag[0] = high_mag_fixed
    mag[1] = high_mag_fixed
    vis[0] = vis_count
    vis[1] = vis_count
    pipeline.buffers.grad_mean_2d_mag_accum.copy_from_numpy(mag)
    pipeline.buffers.visibility_count_accum.copy_from_numpy(vis)

    parent_logscale_1 = log_scales[1, 0]  # 0.05

    stats = pipeline.densify(cfg)

    assert stats["n_keep"]  == 1, f"expected 1 KEEP, got {stats}"
    assert stats["n_split"] == 1, f"expected 1 SPLIT, got {stats}"
    assert stats["n_clone"] == 1, f"expected 1 CLONE, got {stats}"
    assert stats["n_prune"] == 2, f"expected 2 PRUNE, got {stats}"

    # pre=5, split adds 1 tail child, clone adds 1 tail child, so post_split=7.
    # mask drops 2 prunes, final=5.
    assert stats["pre_count"] == 5
    assert stats["post_split_count"] == 7
    assert stats["final_count"] == 5, f"expected 5 after compaction, got {stats}"

    # Verify there exists at least one gaussian with the SPLIT-reduced
    # log_scale (parent was 0.05, child is log(0.05) - log(1.6)).
    post = _read_gaussians(pipeline)
    child_log_scale_expected = np.log(0.05 / 1.6)
    # log_scale cols 3..6 of the 14-float struct layout.
    split_children = np.isclose(post[:, 3], child_log_scale_expected, atol=1e-3).sum()
    assert split_children >= 2, (
        f"expected at least 2 split children with log_scale≈{child_log_scale_expected:.3f}, "
        f"got {split_children} (log_scale_x values: {post[:, 3].round(3)})"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_densify_clone_preserves_parent(backend: str) -> None:
    """A single gaussian flagged for CLONE should produce exactly one
    extra gaussian, and one of the two (parent) should have identical
    params to the pre-densify state."""
    pipeline = _setup(backend)

    pipeline.set_gaussians_from_numpy(
        means=np.array([[0.1, 0.2, 2.5]], dtype=np.float32),
        log_scales=np.log(np.array([[0.005, 0.005, 0.005]], dtype=np.float32)),
        rot_quats=np.array([[0.9659, 0.0, 0.2588, 0.0]], dtype=np.float32),
        opacities=np.array([0.7], dtype=np.float32),
        color_logits=np.array([[0.3, -0.5, 0.8]], dtype=np.float32),
    )
    pre = _read_gaussians(pipeline)

    # Pump high 2d grad to trigger CLONE (small scale → clone, not split).
    cfg = DensifyConfig()
    vis_count = 50
    high_mag_fixed = int(cfg.t_grad * 10.0 * vis_count * 65536.0)
    mag = np.zeros(pipeline.max_splats, dtype=np.int32)
    vis = np.zeros(pipeline.max_splats, dtype=np.uint32)
    mag[0] = high_mag_fixed
    vis[0] = vis_count
    pipeline.buffers.grad_mean_2d_mag_accum.copy_from_numpy(mag)
    pipeline.buffers.visibility_count_accum.copy_from_numpy(vis)

    stats = pipeline.densify(cfg)
    assert stats["n_clone"] == 1
    assert stats["final_count"] == 2

    post = _read_gaussians(pipeline)
    # gsplat-style CLONE: clone is an exact duplicate (no position jitter,
    # zero Adam). Both surviving gaussians must equal the parent's
    # pre-densify state, full 14 fields.
    np.testing.assert_allclose(post[0], pre[0], atol=1e-6)
    np.testing.assert_allclose(post[1], pre[0], atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_densify_noop_when_no_signal(backend: str) -> None:
    """If tracking signals are all zero and no pruning conditions are met,
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

    # Zero tracking buffers.
    pipeline.clear_tracking()

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
    cfg = DensifyConfig(opacity_reset_value=0.05)
    pipeline.reset_opacity(cfg)
    post = _read_gaussians(pipeline)
    # gsplat semantics: opacity = min(opacity, logit(0.05)). Gaussians at
    # 0.9 and 0.5 get clamped DOWN to logit(0.05); the one at 0.01 is
    # already below the cap and stays at logit(0.01).
    def logit(p):
        p_c = np.clip(p, 1e-5, 1.0 - 1e-5)
        return np.log(p_c / (1.0 - p_c))
    logit_cap = logit(0.05)
    expected = np.minimum(logit(in_probs), logit_cap).astype(np.float32)
    np.testing.assert_allclose(post[:, 10], expected, atol=1e-5)
