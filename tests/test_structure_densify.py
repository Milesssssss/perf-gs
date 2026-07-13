"""End-to-end test of the structure-aware densification path (SAD-GS).

Two levels:
  1. Decision + anisotropic split: inject eta accumulators directly (bypassing
     the scoring kernel) so the multiview decision and grid-split child counts
     are checked against a hand-computed oracle.
  2. Scoring kernel: a large Gaussian viewed against a high-frequency (large
     lambda1) structure field must produce eta > 1 and vote to split; a tiny
     Gaussian against a low-frequency field must not.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.densify import StructureConfig
from trainer.dataset import simple_camera
from trainer.device import create_device

BACKENDS = ["vulkan", "d3d12"]


def _setup(backend: str, max_splats: int = 256) -> TestPipeline:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return TestPipeline(device=device, width=64, height=64, max_splats=max_splats)


def _count(p: TestPipeline) -> int:
    return int(p.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])


def _count_actions(p: TestPipeline, pre_count: int) -> dict[str, int]:
    """Read the action buffer post-densify and count each action label.

    ``densify_structure_aware`` is wait-free now (returns None counts), so tests
    that check the decision's action distribution must read the buffer from
    the GPU themselves. ``pre_count`` is the count BEFORE the densify so we
    only look at the pre-densify slots (post-densify slots include appended
    children, whose action tag is stale KEEP)."""
    raw = p.buffers.action_buffer.to_numpy()
    a = np.frombuffer(raw.tobytes(), dtype=np.uint32)[:pre_count]
    return {
        "n_keep":  int((a == 0).sum()),
        "n_split": int((a == 1).sum()),
        "n_clone": int((a == 2).sum()),
        "n_prune": int((a == 3).sum()),
    }


def _set_eta(p: TestPipeline, idx: int, *, n_total: int,
             n_high: int = 0, n_low: int = 0, eta_max=(0.0, 0.0, 0.0),
             grad_high: bool = True) -> None:
    """Write eta accumulators for one gaussian (others stay zero). Matches the
    SADGS model: n_high/n_low are SCALAR view counts (max-over-axes eta), and
    the split gate also requires is_grad_high, so we inject the avg-grad signal
    (grad_mean_2d_mag_accum / visibility_count) above the grad_threshold."""
    b = p.buffers
    N = p.max_splats

    def w(buf, val, dtype=np.uint32):
        a = np.zeros(N, dtype=dtype)
        a[idx] = val
        buf.copy_from_numpy(a)

    w(b.eta_n_total, n_total)
    w(b.eta_buffers["n_high"], n_high)
    w(b.eta_buffers["n_low"], n_low)
    for ax, key in enumerate(("max_x", "max_y", "max_z")):
        w(b.eta_buffers[key], int(eta_max[ax] * 65536.0))

    # is_grad_high signal: avg_grad = (mag/65536)/vis, compared against the
    # convention-scaled floor grad_threshold*W*H*3 (~0.12 at 64x64). Set vis=1
    # and a mag comfortably above it (or at zero, below).
    vis = 1
    mag = int(1.0 * 65536.0) if grad_high else 0  # avg_grad = 1.0 (>> ~0.12 floor), or 0
    w(b.visibility_count_accum, vis)
    a = np.zeros(N, dtype=np.int64)
    a[idx] = mag
    b.grad_mean_2d_mag_accum.copy_from_numpy(a)


def _one_gaussian(p: TestPipeline, scale=0.1, opacity=0.9) -> None:
    p.set_gaussians_from_numpy(
        means=np.array([[0.0, 0.0, 3.0]], dtype=np.float32),
        log_scales=np.log(np.full((1, 3), scale, dtype=np.float32)),
        rot_quats=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([opacity], dtype=np.float32),
        color_logits=np.zeros((1, 3), dtype=np.float32),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_anisotropic_split_child_count(backend: str) -> None:
    """A gaussian that violates on x (eta_max=4) and y (eta_max=2) only should
    grid-split into ceil(sqrt(4)) * ceil(sqrt(2)) * 1 = 2 * 2 * 1 = 4 children."""
    p = _setup(backend)
    _one_gaussian(p)
    # high_ratio = 10/10 = 1.0 > tau_split, gradient high. Per-axis eta_max
    # drives the factor: x=4, y=2, z clean.
    _set_eta(p, 0, n_total=10, n_high=10, n_low=0,
             eta_max=(4.0, 2.0, 0.05), grad_high=True)

    # dense_scale=0 so the (scale 0.1) gaussian is "large" -> SPLIT path.
    cfg = StructureConfig(tau_split=0.8, tau_prune=0.8, split_power=0.5,
                          max_split_factor=4, dense_scale=0.0)
    pre_count = _count(p)
    p.densify_structure_aware(cfg)

    # n_x = ceil(sqrt(4)) = 2, n_y = ceil(sqrt(2)) = 2, n_z = 1 -> 4 children.
    actions = _count_actions(p, pre_count)
    assert actions["n_split"] == 1, f"expected 1 split, got {actions}"
    assert _count(p) == 4, f"expected 4 grid children, got {_count(p)}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_small_gaussian_clones_not_splits(backend: str) -> None:
    """A triggered SMALL gaussian (max scale <= dense_scale) should CLONE
    (exact duplicate, +1) rather than anisotropically split — SADGS routes the
    same eta trigger to clone-if-small / split-if-large."""
    p = _setup(backend)
    _one_gaussian(p, scale=0.1)  # world-space max scale = 0.1
    _set_eta(p, 0, n_total=10, n_high=10, n_low=0,
             eta_max=(4.0, 2.0, 0.05), grad_high=True)

    # dense_scale=0.2 > 0.1, so the gaussian is "small" -> CLONE path.
    cfg = StructureConfig(tau_split=0.8, tau_prune=0.8, split_power=0.5,
                          max_split_factor=4, dense_scale=0.2)
    pre_count = _count(p)
    p.densify_structure_aware(cfg)

    actions = _count_actions(p, pre_count)
    assert actions["n_clone"] == 1, f"expected 1 clone, got {actions}"
    assert actions["n_split"] == 0
    assert _count(p) == 2, f"clone should append exactly 1 duplicate, got {_count(p)}"

    # The clone is an exact duplicate of the parent (same first-14 floats).
    g = np.frombuffer(p.buffers.gaussians.to_numpy().tobytes(),
                      dtype=np.float32).reshape(-1, 14)
    assert np.allclose(g[0], g[1]), "clone is not an exact duplicate of the parent"


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_split_when_ratio_below_threshold(backend: str) -> None:
    """high_ratio = 3/10 = 0.3 < tau_split=0.8 -> no split, count unchanged
    (even though eta_max=9 and gradient is high)."""
    p = _setup(backend)
    _one_gaussian(p)
    _set_eta(p, 0, n_total=10, n_high=3, n_low=0, eta_max=(9.0, 0.0, 0.0), grad_high=True)

    cfg = StructureConfig(tau_split=0.8)
    pre_count = _count(p)
    p.densify_structure_aware(cfg)
    actions = _count_actions(p, pre_count)
    assert actions["n_split"] == 0, f"expected 0 splits, got {actions}"
    assert _count(p) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_split_when_gradient_low(backend: str) -> None:
    """high_ratio passes (10/10) but is_grad_high is False (well-placed large
    gaussian) -> no split. This is the SADGS gradient AND-gate."""
    p = _setup(backend)
    _one_gaussian(p)
    _set_eta(p, 0, n_total=10, n_high=10, n_low=0, eta_max=(4.0, 0.0, 0.0), grad_high=False)

    cfg = StructureConfig(tau_split=0.8)
    pre_count = _count(p)
    p.densify_structure_aware(cfg)
    actions = _count_actions(p, pre_count)
    assert actions["n_split"] == 0, f"expected 0 splits, got {actions}"
    assert _count(p) == 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_prune_low_ratio(backend: str) -> None:
    """low_ratio = 10/10 = 1.0 > tau_prune -> prune (over-resolved)."""
    p = _setup(backend)
    _one_gaussian(p)
    _set_eta(p, 0, n_total=10, n_high=0, n_low=10, eta_max=(0.05, 0.05, 0.05))

    cfg = StructureConfig(tau_prune=0.8)
    pre_count = _count(p)
    p.densify_structure_aware(cfg)
    actions = _count_actions(p, pre_count)
    assert actions["n_prune"] == 1, f"expected 1 prune, got {actions}"
    assert _count(p) == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_split_factor_capped(backend: str) -> None:
    """eta_max=100 with cap 3 -> n_k clamped to 3, not ceil(sqrt(100))=10."""
    p = _setup(backend)
    _one_gaussian(p)
    _set_eta(p, 0, n_total=10, n_high=10, n_low=0,
             eta_max=(100.0, 0.0, 0.0), grad_high=True)

    cfg = StructureConfig(tau_split=0.8, split_power=0.5, max_split_factor=3)
    p.densify_structure_aware(cfg)
    # x: min(ceil(sqrt(100))=10, cap 3) = 3; y,z = 1 -> 3 children.
    assert _count(p) == 3


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Replicate shaders/common/math.slang::matrixFromQuaternion EXACTLY
    (q = (w, x, y, z), normalized, row-major). Must match bit-for-bit in
    convention so the Python oracle and the GPU split agree on R."""
    q = q / np.linalg.norm(q)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),       2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


@pytest.mark.parametrize("backend", BACKENDS)
def test_anisotropic_split_preserves_covariance(backend: str) -> None:
    """SADGS split geometry invariant: for ks_scale_power == 1, the equal-weight
    mixture of the grid children must reproduce the PARENT's 3D covariance
    exactly (SADGS gaussian_model.py Eq. 8-9; the sqrt(12) child spacing is
    precisely the factor that makes this hold).

    This is the regression guard for the sqrt(12) fix: the previous code spaced
    children at coord/n with NO sqrt(12), which contracts the mixture to ~31% of
    the parent variance at n=2 (and worse for larger n) — that version fails
    this test dramatically; the sqrt(12) version passes to fp32 precision.

    Uses a non-trivial rotation, anisotropic scale, and UNEQUAL per-axis split
    factors n = (3, 2, 1) so the test exercises multi-factor grids and the
    n_k == 1 no-op axis, not just the isotropic k=2 case."""
    p = _setup(backend)

    # Anisotropic parent scale + a real (non-identity) rotation.
    scale = np.array([0.30, 0.20, 0.12], dtype=np.float64)
    quat = np.array([0.9, 0.1, 0.2, 0.3], dtype=np.float32)   # (w, x, y, z), unnormalized
    p.set_gaussians_from_numpy(
        means=np.array([[0.1, -0.2, 3.0]], dtype=np.float32),
        log_scales=np.log(scale.astype(np.float32))[None, :],
        rot_quats=quat[None, :],
        opacities=np.array([0.9], dtype=np.float32),
        color_logits=np.zeros((1, 3), dtype=np.float32),
    )

    # eta_max=(9,4,0.05), split_power=0.5 -> n=(ceil(sqrt 9), ceil(sqrt 4),
    # ceil(sqrt 0.05)) = (3, 2, 1) -> 6 children. high_ratio=1.0, grad high.
    _set_eta(p, 0, n_total=10, n_high=10, n_low=0,
             eta_max=(9.0, 4.0, 0.05), grad_high=True)

    # dense_scale=0 so the (large) gaussian takes the SPLIT path; ks_scale_power
    # = 1.0 is the variance-preserving choice paired with the sqrt(12) spacing.
    cfg = StructureConfig(tau_split=0.8, tau_prune=0.8, split_power=0.5,
                          max_split_factor=8, dense_scale=0.0, ks_scale_power=1.0)
    p.densify_structure_aware(cfg)

    m = _count(p)
    assert m == 6, f"expected 3*2*1 = 6 grid children, got {m}"

    # Read back the 6 children: packed 14 floats each (mean[0:3], log_scale[3:6],
    # rot_quat[6:10], opacity[10], sh0[11:14]).
    g = np.frombuffer(p.buffers.gaussians.to_numpy().tobytes(),
                      dtype=np.float32).reshape(-1, 14)[:m].astype(np.float64)
    child_means = g[:, 0:3]
    child_scales = np.exp(g[:, 3:6])

    R = _quat_to_matrix(quat)
    # Parent covariance Sigma = R diag(scale^2) R^T.
    parent_cov = R @ np.diag(scale ** 2) @ R.T

    # Equal-weight Gaussian-mixture covariance:
    #   Sigma_mix = mean_i(Sigma_i) + mean_i((mu_i - mu_bar)(mu_i - mu_bar)^T)
    # where Sigma_i = R diag(child_scale_i^2) R^T (children share the parent R).
    within = np.zeros((3, 3))
    for s_i in child_scales:
        within += R @ np.diag(s_i ** 2) @ R.T
    within /= m

    mu_bar = child_means.mean(axis=0)
    d = child_means - mu_bar
    between = (d[:, :, None] * d[:, None, :]).mean(axis=0)

    mixture_cov = within + between

    # fp32 GPU math on cov entries ~1e-2..1e-1: atol 2e-4 is comfortable.
    assert np.allclose(mixture_cov, parent_cov, atol=2e-4, rtol=1e-3), (
        f"grid-child mixture covariance does not preserve the parent's.\n"
        f"parent=\n{parent_cov}\nmixture=\n{mixture_cov}\n"
        f"diag ratio (mixture/parent) = "
        f"{np.diag(mixture_cov) / np.diag(parent_cov)}")

    # The mixture mean must also match the parent mean (grid is centered).
    assert np.allclose(mu_bar, np.array([0.1, -0.2, 3.0]), atol=2e-4), (
        f"grid is not centered on the parent: mean-of-means = {mu_bar}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_scoring_kernel_flags_large_gaussian(backend: str) -> None:
    """The scoring kernel itself: a LARGE gaussian against a high-frequency
    structure field (large lambda1) yields eta > 1 (splits), while a TINY
    gaussian against the same field does not."""
    import slangpy as spy

    cam = simple_camera(width=64, height=64, eye=(0.0, 0.0, 0.0),
                        target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                        fovy_degrees=60.0)
    cfg = StructureConfig(n_scoring_views=1, tau_split=0.5)

    # Structure field with lambda1 = 0.25 everywhere -> wavelength ~ 2px.
    # Chosen so the LARGE gaussian (extent ~tens of px) has eta >> 1, while the
    # TINY gaussian stays below eta=1 EVEN after the renderer's dilation floors
    # its footprint to ~sqrt(0.3)=0.55px (0.55 * sqrt(0.25) = 0.27 < 1).
    # Packed (Sxx, Sxy, Syy, 0); lambda1 = Sxx here (Sxy=Syy=0).
    lam = 0.25
    field = np.zeros((64, 64, 4), dtype=np.float32)
    field[..., 0] = lam

    def _make_tex(p):
        tex = p.device.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
            width=64, height=64,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="sbar")
        tex.copy_from_numpy(np.ascontiguousarray(field))
        return tex

    def _set_grad_high(p):
        # The split gate ANDs in is_grad_high; inject an above-threshold avg
        # grad so the test isolates the eta/frequency behaviour.
        N = p.max_splats
        p.buffers.visibility_count_accum.copy_from_numpy(np.ones(N, dtype=np.uint32))
        mag = np.full(N, int(1.0 * 65536.0), dtype=np.int64)
        p.buffers.grad_mean_2d_mag_accum.copy_from_numpy(mag)

    # Large gaussian -> big projected extent -> eta > 1 -> should split.
    p_big = _setup(backend)
    _one_gaussian(p_big, scale=0.5)
    p_big.compute_structure_scores([cam], [_make_tex(p_big)], cfg)
    _set_grad_high(p_big)
    pre_big = _count(p_big)
    p_big.densify_structure_aware(cfg)
    actions_big = _count_actions(p_big, pre_big)

    # Tiny gaussian -> small extent -> eta < 1 -> no split.
    p_small = _setup(backend)
    _one_gaussian(p_small, scale=0.001)
    p_small.compute_structure_scores([cam], [_make_tex(p_small)], cfg)
    _set_grad_high(p_small)
    pre_small = _count(p_small)
    p_small.densify_structure_aware(cfg)
    actions_small = _count_actions(p_small, pre_small)

    assert actions_big["n_split"] >= 1, "large gaussian should violate frequency and split"
    assert actions_small["n_split"] == 0, "tiny gaussian should not split"
