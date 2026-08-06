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
from trainer.backend.densify.densify import DensifyConfig, StructureConfig
from trainer.data.dataset import simple_camera
from trainer.backend.compute.device import create_device

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
    p.set_primitive_from_numpy(
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
    p.set_primitive_from_numpy(
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


def _read_gaussians(p: TestPipeline, n: int) -> np.ndarray:
    """Read back the first n gaussians as an (n, 14) float array (mean[0:3],
    log_scale[3:6], rot_quat[6:10], opacity[10], sh0[11:14])."""
    g = np.frombuffer(p.buffers.gaussians.to_numpy().tobytes(),
                      dtype=np.float32).reshape(-1, 14)
    return g[:n]


def _read_eta(p: TestPipeline, n: int) -> dict[str, np.ndarray]:
    """Read back the first n slots of every eta accumulator."""
    def r(buf):
        return np.frombuffer(buf.to_numpy().tobytes(), dtype=np.uint32)[:n]
    return {
        "n_total": r(p.buffers.eta_n_total),
        "n_high":  r(p.buffers.eta_buffers["n_high"]),
        "n_low":   r(p.buffers.eta_buffers["n_low"]),
        "max_x":   r(p.buffers.eta_buffers["max_x"]),
        "max_y":   r(p.buffers.eta_buffers["max_y"]),
        "max_z":   r(p.buffers.eta_buffers["max_z"]),
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_eta_carried_through_filler_densify(backend: str) -> None:
    """SAD-GS eta persistence (#2): a FILLER (AbsGS) densify with carry_eta=True
    must permute the slot-indexed eta accumulators through the same
    valid_mask/scan as the gaussians, carrying the multiview window of surviving
    gaussians and zeroing it for brand-new primitives.

    Carry rule under test (matches SADGS):
      - KEEP survivor        -> eta carried
      - CLONE parent (kept)  -> eta carried; the appended clone -> zero
      - SPLIT parent slot    -> overwritten with a NEW child-1 -> eta zeroed;
                                the appended child-2 -> zero
      - PRUNE                -> dropped

    We drive the four actions deterministically via decide_actions_simple's
    gates (opacity<t_opacity -> PRUNE; grad>thresh & scale>scale_thresh ->
    SPLIT; grad>thresh & small -> CLONE; grad<=thresh -> KEEP), tag each parent
    with a UNIQUE eta signature, and after the densify match each surviving
    gaussian back to its parent by POSITION (the gaussian scatter is co-indexed
    with the eta scatter, so a correct implementation keeps them aligned)."""
    p = _setup(backend, max_splats=64)

    # 4 parents at distinct x-positions. Scale: 3 small (0.01) so KEEP/CLONE/
    # PRUNE stay in the clone-size band; 1 large (0.30) so the SPLIT candidate
    # takes the split path. Opacity: parent 3 near-zero so it PRUNES.
    #   idx 0: KEEP   (grad low)                          -> carry
    #   idx 1: CLONE  (grad high, small scale)            -> carry (+appended 0)
    #   idx 2: SPLIT  (grad high, large scale)            -> zero  (+appended 0)
    #   idx 3: PRUNE  (opacity below floor)               -> dropped
    xs = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
    scales = np.array([0.01, 0.01, 0.30, 0.01], dtype=np.float32)
    opac = np.array([0.9, 0.9, 0.9, 1e-4], dtype=np.float32)
    means = np.stack([xs, np.zeros(4, np.float32), np.full(4, 5.0, np.float32)], axis=1)
    p.set_primitive_from_numpy(
        means=means,
        log_scales=np.log(np.repeat(scales[:, None], 3, axis=1)),
        rot_quats=np.tile(np.array([1.0, 0, 0, 0], np.float32), (4, 1)),
        opacities=opac,
        color_logits=np.zeros((4, 3), dtype=np.float32),
    )
    p.buffers.num_gaussians_counter.copy_from_numpy(np.array([4], dtype=np.uint32))

    # Unique eta signature per parent so we can identify who landed where.
    # (values are arbitrary but distinct; max_* stored as fixed-point uint.)
    N = p.max_splats
    def wbuf(buf, vals, dtype=np.uint32):
        a = np.zeros(N, dtype=dtype)
        a[:4] = vals
        buf.copy_from_numpy(a)
    wbuf(p.buffers.eta_n_total, [11, 22, 33, 44])
    wbuf(p.buffers.eta_buffers["n_high"], [1, 2, 3, 4])
    wbuf(p.buffers.eta_buffers["n_low"], [5, 6, 7, 8])
    wbuf(p.buffers.eta_buffers["max_x"], [100, 200, 300, 400])
    wbuf(p.buffers.eta_buffers["max_y"], [110, 210, 310, 410])
    wbuf(p.buffers.eta_buffers["max_z"], [120, 220, 320, 420])

    # Grad/visibility signal driving the KEEP/CLONE/SPLIT decision. avg_grad =
    # (mag/65536)/vis; the internal floor is grad_thresh * W*H*3 = 0.0002 *
    # 64*64*3 = ~2.46. Give idx 1,2 avg_grad = 5.0 (>> floor) and idx 0 zero
    # grad (-> KEEP). vis=1 for all live slots.
    vis = np.zeros(N, dtype=np.uint32); vis[:4] = 1
    p.buffers.visibility_count_accum.copy_from_numpy(vis)
    mag = np.zeros(N, dtype=np.int64)
    mag[1] = mag[2] = int(5.0 * 65536.0)   # avg_grad = 5.0 >> ~2.46 floor
    mag[0] = mag[3] = 0                     # KEEP / (PRUNE decided by opacity)
    p.buffers.grad_mean_2d_mag_accum.copy_from_numpy(mag)

    cfg = DensifyConfig(t_opacity=0.005, t_big_scale=1e9)
    # scale_thresh between the small (0.01) and large (0.30) scales so idx 2
    # splits and idx 1 clones. grad_thresh at gsplat's 0.0002. Disable big-scale
    # prune (t_big_scale huge + allow_scale_prune False) so only opacity prunes.
    p.densify_simple(cfg, grad_thresh=0.0002, scale_thresh=0.1,
                     prune_big_scale=1e9, allow_scale_prune=False,
                     carry_eta=True)

    n = _count(p)
    # 4 parents - 1 prune (idx3) + 1 clone-append (idx1) + 1 split-append (idx2)
    # = 5 live gaussians.
    assert n == 5, f"expected 5 gaussians after densify, got {n}"

    g = _read_gaussians(p, n)
    eta = _read_eta(p, n)
    xpos = g[:, 0]

    # Locate the KEEP survivor (x=-3) and the CLONE parent (x=-1) by position.
    keep_slots = np.where(np.isclose(xpos, -3.0, atol=1e-3))[0]
    clone_parent_slots = np.where(np.isclose(xpos, -1.0, atol=1e-3))[0]
    assert len(keep_slots) == 1, f"KEEP survivor not found uniquely: x={xpos}"
    # CLONE produces two gaussians at x=-1 (parent + exact duplicate).
    assert len(clone_parent_slots) == 2, f"expected 2 gaussians at clone x=-1: x={xpos}"

    ks = int(keep_slots[0])
    # KEEP survivor keeps its full eta signature (parent 0).
    assert eta["n_total"][ks] == 11, f"KEEP eta_n_total not carried: {eta['n_total'][ks]}"
    assert eta["n_high"][ks] == 1 and eta["n_low"][ks] == 5
    assert eta["max_x"][ks] == 100 and eta["max_y"][ks] == 110 and eta["max_z"][ks] == 120

    # Exactly ONE of the two x=-1 gaussians carries parent-1's eta (the surviving
    # parent slot); the other (the appended clone) is zeroed.
    c_ntotal = eta["n_total"][clone_parent_slots]
    assert sorted(c_ntotal.tolist()) == [0, 22], (
        f"clone eta: exactly one slot should carry n_total=22, other 0; got {c_ntotal}")
    carrier = clone_parent_slots[np.argmax(c_ntotal)]
    assert eta["n_high"][carrier] == 2 and eta["max_x"][carrier] == 200

    # The SPLIT children must ALL be zeroed — a split replaces the parent with
    # brand-new children. split_and_clone jitters child positions within the
    # parent covariance (parent scale 0.30), so they land NEAR x=+1 but not
    # exactly; identify them as the slots that are neither the KEEP survivor
    # nor either clone-x=-1 gaussian (and there must be exactly 2).
    identified = set(keep_slots.tolist()) | set(clone_parent_slots.tolist())
    split_slots = [s for s in range(n) if s not in identified]
    assert len(split_slots) == 2, f"expected 2 split children, got {split_slots}: x={xpos}"
    # Sanity: they really are the split children (positioned near the split
    # parent's x=+1, within a few sigma of the 0.30-scale jitter).
    for s in split_slots:
        assert abs(xpos[s] - 1.0) < 2.0, f"unexpected split-child position x={xpos[s]}"
        assert eta["n_total"][s] == 0, f"SPLIT child eta not zeroed: slot {s} = {eta['n_total'][s]}"
        assert eta["max_x"][s] == 0 and eta["n_high"][s] == 0

    # Parent 3 (x=+3) was PRUNED — it must not appear at all (no gaussian near x=+3).
    assert not np.any(np.abs(xpos - 3.0) < 0.5), "pruned gaussian survived"

    # Global sanity: total carried n_total across all slots = 11 (keep) + 22
    # (clone parent) = 33; everything else zero. This catches any stray carry.
    assert int(eta["n_total"].sum()) == 33, (
        f"total carried eta_n_total should be 11+22=33, got {int(eta['n_total'].sum())}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_eta_carried_through_morton_sort(backend: str) -> None:
    """mortonScatterEta: a Morton re-sort with carry_eta=True permutes the eta
    accumulators by the SAME sorted_values as the gaussians (pure permutation,
    no zeroing). After the sort, each gaussian must still be paired with its
    original eta signature — verified by matching on position."""
    p = _setup(backend, max_splats=64)

    # Place 5 gaussians deliberately OUT of Morton order (descending x) so the
    # sort actually permutes them; tag each with a position-keyed eta.
    k = 5
    xs = np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    means = np.stack([xs, np.zeros(k, np.float32), np.zeros(k, np.float32)], axis=1)
    p.set_primitive_from_numpy(
        means=means,
        log_scales=np.log(np.full((k, 3), 0.05, dtype=np.float32)),
        rot_quats=np.tile(np.array([1.0, 0, 0, 0], np.float32), (k, 1)),
        opacities=np.full(k, 0.9, dtype=np.float32),
        color_logits=np.zeros((k, 3), dtype=np.float32),
    )
    p.buffers.num_gaussians_counter.copy_from_numpy(np.array([k], dtype=np.uint32))

    # eta signature = round(10 * (x + 1)) so it's uniquely tied to position.
    N = p.max_splats
    sig = np.zeros(N, dtype=np.uint32)
    sig[:k] = (10 * (xs + 1)).astype(np.uint32)   # [50,40,30,20,10]
    p.buffers.eta_n_total.copy_from_numpy(sig)
    for key in ("n_high", "n_low", "max_x", "max_y", "max_z"):
        p.buffers.eta_buffers[key].copy_from_numpy(sig)

    p.morton_sort(np.array([-1, -1, -1], np.float32),
                  np.array([5, 5, 5], np.float32), carry_eta=True)

    g = _read_gaussians(p, k)
    eta = _read_eta(p, k)
    # For every surviving gaussian, its eta must equal its position signature.
    for i in range(k):
        expected = int(round(10 * (g[i, 0] + 1)))
        assert eta["n_total"][i] == expected, (
            f"slot {i} (x={g[i,0]:.2f}) expected eta {expected}, got {eta['n_total'][i]}")
        for key in ("n_high", "n_low", "max_x", "max_y", "max_z"):
            assert eta[key][i] == expected, f"slot {i} {key} misaligned after sort"
