# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Importance-guided sparse backward sampling.

Four claims carry this feature, and each one is checkable directly rather than
by watching PSNR:

  BUDGET       the solver holds ``(1/N) sum p_i = rho`` whatever the importance
               field looks like, including where the map saturates. Normalizing
               and clamping instead would lose the clipped mass and quietly
               undershoot, so this is pinned across alpha and rho.

  EQUIVALENCE  at ``alpha = 0`` the whole path must collapse onto the EXISTING
               uniform mask — and it does bitwise, both in the selected set and
               in the gradient, not just to a tolerance. That is the strongest
               available statement that the adaptive path adds a marginal and
               changes nothing else; see ``uniform_volume`` for the rank identity
               it rests on.

  UNBIASEDNESS ``E[M_i h_i / p_i] = h_i``, so the Gaussian gradient stays an
               estimate of the DENSE one and the training objective is untouched.
               Verified by Monte Carlo, and separately as an exact scaling
               identity against the existing TILE_AREA/k weight.

  STRUCTURE    the importance field must be independent of the 8x8 rasterizer
               tiling. A per-tile field is exactly the construction this design
               rejects, and "it is smooth" is not the same claim as "it has no
               tile-periodic component", so that component is measured.

The GPU tests additionally pin the packed table the backward will read, because
that is where a compaction bug hides as a quietly wrong gradient rather than as a
failure.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

from trainer.backend.render import adaptive_stbn as ad
from trainer.backend.render import stbn
from trainer.contracts.codegen import TILE_AREA, TILE_WIDTH

W = H = 128
BUDGET = 16                       # 16/64 = rho 0.25, the primary density
RHO = BUDGET / TILE_AREA


def _asset_or_skip() -> stbn.StbnMaskSpec:
    spec = stbn.StbnMaskSpec()
    if not spec.resolved_path().is_file():
        pytest.skip("NVIDIA STBN asset absent; run scripts/data/download_stbn.py")
    return spec


def _hotspot_field(height: int = H, width: int = W, seed: int = 0) -> np.ndarray:
    """A synthetic screen-gradient magnitude with the character of a real one:
    noisy at the pixel level, with a smooth high-error region."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width] / float(max(height, width))
    hotspot = np.exp(-(((yy - 0.35) ** 2 + (xx - 0.6) ** 2) / 0.01))
    return np.abs(rng.standard_normal((height, width))) * (0.2 + 6.0 * hotspot)


# --------------------------------------------------------------- the budget

@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 0.9])
@pytest.mark.parametrize("fraction", [0.09375, 0.15625, 0.25, 0.5])
def test_solver_holds_the_expected_budget(alpha: float, fraction: float) -> None:
    """``(1/N) sum p_i = rho`` to numerical tolerance, saturation included.

    This is the property that makes an adaptive arm comparable to its uniform
    control at all: the two must spend the SAME expected number of pixels, or the
    comparison measures budget rather than allocation.
    """
    spec = ad.AdaptiveSpec(fraction=fraction, alpha=alpha)
    p, diag = ad.probability_map(_hotspot_field(), spec)
    rho = spec.budget_fraction()
    assert diag["p_mean"] == pytest.approx(rho, abs=1e-9), diag
    assert 0.0 < diag["p_min"] <= diag["p_max"] <= 1.0, diag
    # A saturating map is the interesting case, so make sure the sweep reaches it
    # rather than only testing the easy branch.
    assert diag["saturated_fraction"] >= 0.0


def test_saturation_is_reached_and_still_budgeted() -> None:
    """A field peaked enough to clip must still land on the budget.

    The failure mode this guards is the tempting shortcut: normalize a_i, clamp to
    1, and ship it. That loses the clipped mass, so the realized density comes out
    BELOW rho by however much was clipped — silently, and by more the peakier the
    field.
    """
    field = np.full((H, W), 0.01)
    field[40:60, 40:60] = 50.0           # a hard hot spot
    spec = ad.AdaptiveSpec(fraction=RHO, alpha=0.9, sigma=8.0)
    p, diag = ad.probability_map(field, spec)
    assert diag["saturated_fraction"] > 0.0, "expected the map to clip somewhere"
    assert diag["p_mean"] == pytest.approx(RHO, abs=1e-9)

    # The naive alternative, for the record: normalize then clamp.
    s_tilde = ad.smooth_importance(field, spec)
    naive = np.minimum(1.0, RHO * ad.strength(s_tilde, spec.alpha))
    assert naive.mean() < RHO - 1e-4, (
        "the clamp-without-correction baseline is supposed to undershoot; if it "
        "does not, this test is not exercising saturation")


def test_alpha_zero_is_exactly_uniform() -> None:
    """alpha=0 must give p_i = rho for EVERY pixel, exactly.

    Exactly, not approximately: 1/p_i then equals the TILE_AREA/k the existing
    loss shaders apply as a literal, which is what makes the alpha=0 gradient
    bitwise identical rather than merely close.
    """
    spec = ad.AdaptiveSpec(fraction=RHO, alpha=0.0)
    p, diag = ad.probability_map(_hotspot_field(), spec)
    assert np.all(p == RHO)
    assert diag["lambda"] == RHO
    assert ad.ipw_scale(p, spec)[0, 0] == float(TILE_AREA) / BUDGET


def test_lambda_is_monotone_in_the_budget() -> None:
    """A larger budget cannot need a smaller lambda — a sanity check on the
    bracket, which would otherwise fail silently by returning an endpoint."""
    field = _hotspot_field()
    spec = ad.AdaptiveSpec(alpha=0.75)
    a = ad.strength(ad.smooth_importance(field, spec), spec.alpha)
    lams = [ad.solve_lambda(a, rho) for rho in (0.05, 0.1, 0.25, 0.5, 0.75)]
    assert all(b > a_ for a_, b in zip(lams, lams[1:])), lams


# ------------------------------------------------------- the smoothing field

@pytest.mark.parametrize("sigma", [4.0, 8.0, 16.0])
@pytest.mark.parametrize("downsample", [1, 2, 4, 8])
def test_effective_sigma_tracks_the_request(sigma: float, downsample: int) -> None:
    """The composed (box, Gaussian, bilinear) operator must smooth at ``sigma``
    OUTPUT-IMAGE pixels, not at sigma/downsample or at whatever the composition
    happens to give.

    Without the variance compensation in :func:`adaptive_stbn.coarse_sigma` the
    effective scale drifts with the downsample, so a sigma sweep would be
    measuring the downsample instead of sigma.
    """
    size = 257
    impulse = np.zeros((size, size))
    impulse[size // 2, size // 2] = 1.0
    field = ad.smooth_importance(
        impulse, ad.AdaptiveSpec(sigma=sigma, downsample=downsample))
    total = field.sum()
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = (field * yy).sum() / total, (field * xx).sum() / total
    variance = (field * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / total / 2.0
    effective = float(np.sqrt(variance))
    assert effective == pytest.approx(sigma, rel=0.05), (
        f"sigma={sigma} downsample={downsample} -> effective {effective:.3f}")


def test_importance_field_has_no_tile_periodic_structure() -> None:
    """The smoothed field must carry no 8-pixel (rasterizer-tile) periodicity.

    Measured, not asserted from the construction: the field is built on a coarse
    grid whose divisor happens to divide 8, so "it is smooth" would not by itself
    rule out a tile-aligned artifact. Compares the field's power at exactly the
    tile frequency against the neighbouring bins.
    """
    field = ad.smooth_importance(_hotspot_field(seed=5),
                                 ad.AdaptiveSpec(sigma=8.0, downsample=4))
    row = field.mean(axis=0)
    row = row - row.mean()
    power = np.abs(np.fft.rfft(row)) ** 2
    tile_bin = row.size // TILE_WIDTH          # period-8 component
    neighbourhood = power[max(tile_bin - 4, 1):tile_bin + 5]
    assert power[tile_bin] <= 3.0 * float(np.median(neighbourhood)) + 1e-30, (
        f"period-{TILE_WIDTH} power {power[tile_bin]:.3e} stands out of its "
        f"neighbourhood (median {np.median(neighbourhood):.3e})")


def test_smoothing_is_translation_equivariant_up_to_the_grid() -> None:
    """Shifting the input by a whole coarse cell must shift the field, not
    reshape it — i.e. the coarse grid is a sampling lattice, not a partition the
    importance is defined per."""
    spec = ad.AdaptiveSpec(sigma=8.0, downsample=4)
    base = _hotspot_field(seed=7)
    shifted = np.roll(base, 4, axis=1)
    a = ad.smooth_importance(base, spec)
    b = ad.smooth_importance(shifted, spec)
    # Only away from the x edges: np.roll WRAPS, the blur CLAMPS, so the two
    # disagree within one blur support (radius 6 coarse cells = 24 px) of either
    # side. That is a property of the test's shift, not of the operator.
    margin = 4 * (2 + int(np.ceil(3.0 * ad.coarse_sigma(8.0, 4))))
    interior = (slice(None), slice(margin, -margin))
    assert np.abs(np.roll(a, 4, axis=1)[interior] - b[interior]).max() < 1e-12


# ------------------------------------------------- the correlated uniform

def test_uniform_volume_reproduces_the_existing_global_mask() -> None:
    """Thresholding U at k/TILE_AREA must give the EXISTING selected set, bitwise.

    This is the identity the whole equivalence argument rests on: the adaptive
    path samples ``U < p``, so if ``U < rho`` is not exactly the current mask then
    alpha=0 is a different sampling process and every comparison against the
    uniform baseline is confounded.
    """
    base = _asset_or_skip()
    volume = ad.uniform_volume(base)
    for fraction in (0.09375, 0.15625, 0.25, 0.5):
        spec = dataclasses.replace(base, fraction=fraction)
        reference = stbn.global_selected_mask(spec)
        budget = stbn.pixels_per_tile(fraction, TILE_AREA)
        assert np.array_equal(volume < budget / TILE_AREA, reference), fraction


def test_uniform_volume_is_a_uniform_grid_per_slice() -> None:
    """Each slice's U values are a permutation of the uniform grid, so
    ``P(U < q)`` is exact and free of the asset's 8-bit quantization."""
    volume = ad.uniform_volume(_asset_or_skip())
    total = volume.shape[1] * volume.shape[2]
    expected = (np.arange(total) + 0.5) / total
    for index in range(volume.shape[0]):
        assert np.allclose(np.sort(volume[index].reshape(-1)), expected)


def test_hash_uniform_matches_the_shader_hash() -> None:
    """The host's IID stream must be the SHADER's, not a look-alike from numpy's
    RNG — otherwise the offline calibration diagnostic measures a different
    sampler than the trainer runs."""
    u = ad.hash_uniform(64, 64, seed=1234)
    assert u.min() >= 0.0 and u.max() < 1.0
    # Uniformity to the accuracy 4096 draws allow, and independence of the seed.
    assert abs(float(u.mean()) - 0.5) < 0.02
    assert not np.array_equal(u, ad.hash_uniform(64, 64, seed=1235))


# ----------------------------------------------------- the estimator itself

def test_inverse_probability_weighting_is_unbiased() -> None:
    """``E[M_i h_i / p_i] = h_i``, by Monte Carlo over independent draws.

    Done with IID uniforms on purpose: the blue-noise field has only 64 slices, so
    its empirical mean over any realistic number of visits carries a structural
    residual that would confound a bias measurement. Whether the STBN field is
    correctly CALIBRATED is a separate question, measured by the sampling
    diagnostics.
    """
    rng = np.random.default_rng(11)
    spec = ad.AdaptiveSpec(fraction=RHO, alpha=0.75)
    h = rng.standard_normal((32, 32))
    p, _ = ad.probability_map(_hotspot_field(32, 32, seed=2), spec)
    scale = ad.ipw_scale(p, spec)

    draws = 20000
    total = np.zeros_like(h)
    for _ in range(draws):
        mask = rng.random(p.shape) < p
        total += mask * h * scale
    estimate = total / draws
    # Per-pixel standard error of the mean is |h| sqrt((1-p)/(p*draws)).
    sigma = np.abs(h) * np.sqrt((1.0 - p) / (p * draws))
    z = np.abs(estimate - h) / np.maximum(sigma, 1e-12)
    assert float(z.max()) < 5.0, f"worst z-score {z.max():.2f}"
    # The aggregate L1 error is compared against what MC noise ALONE predicts
    # (E|N(0,s)| = s sqrt(2/pi)) rather than against a round number: a fixed
    # threshold here would either pass a real bias or fail on sampling noise,
    # depending only on how many draws someone happened to pick.
    observed = float(np.abs(estimate - h).sum() / np.abs(h).sum())
    predicted = float(np.sqrt(2.0 / np.pi) * sigma.sum() / np.abs(h).sum())
    assert observed < 1.5 * predicted, f"{observed:.5f} vs noise {predicted:.5f}"


def test_uniform_weight_equals_the_existing_tile_area_over_k() -> None:
    """At uniform ``p_i = rho`` the inverse-probability weight IS the weight the
    loss shaders already apply, exactly.

    Section 6 of the design: the loss uses the SUM-over-pixels convention (see
    shaders/losses/l1_l2_loss.slang), so h_i is already the derivative of what the
    dense backward accumulates and there is no extra 1/N to introduce. If one were
    added the sparse gradient would be N times too small and the learning rate
    would silently need retuning.
    """
    for budget in range(1, TILE_AREA + 1):
        spec = ad.AdaptiveSpec(fraction=budget / TILE_AREA, alpha=0.0)
        p = np.full((4, 4), spec.budget_fraction())
        assert ad.ipw_scale(p, spec)[0, 0] == float(TILE_AREA) / budget


def test_biased_control_uses_a_global_rescale_not_per_pixel() -> None:
    """The labelled control must keep the gradient MAGNITUDE matched.

    Omitting 1/p entirely would shrink the gradient by ~rho as well as reweighting
    it, which changes the effective learning rate and the meaning of the
    densifier's ABSOLUTE --grad-thresh — confounding the one question the control
    is asking.
    """
    spec = ad.AdaptiveSpec(fraction=RHO, alpha=0.75, unbiased=False)
    p, _ = ad.probability_map(_hotspot_field(), spec)
    scale = ad.ipw_scale(p, spec)
    assert np.all(scale == 1.0 / RHO)
    unbiased = ad.ipw_scale(p, dataclasses.replace(spec, unbiased=True))
    assert not np.allclose(scale, unbiased)


def test_residual_is_standardized() -> None:
    """``Z = (M - p)/sqrt(p(1-p))`` must be ~zero-mean unit-variance under the
    sampling, so a spectrum of Z reports arrangement rather than the mean field."""
    rng = np.random.default_rng(3)
    p = np.clip(0.25 + 0.2 * rng.standard_normal((64, 64)), 0.02, 0.98)
    z = np.stack([ad.residual(rng.random(p.shape) < p, p) for _ in range(400)])
    assert abs(float(z.mean())) < 0.02
    assert abs(float(z.std()) - 1.0) < 0.05


# ------------------------------------------------- host / shader agreement

def test_shader_constants_match_the_host() -> None:
    """The Slang layout constants and their Python mirrors must agree.

    A drift here is silent — the host would read a field the shader never wrote —
    so the shader text is grepped rather than trusted to a comment.
    """
    from trainer.backend.render import adaptive_sampler as sampler

    source = (Path(__file__).resolve().parents[1] / "shaders" / "losses"
              / "adaptive_importance.slang").read_text(encoding="utf-8")
    expected = {
        "ADAPTIVE_SCALAR_MEAN": sampler.SCALAR_MEAN,
        "ADAPTIVE_SCALAR_A_MIN": sampler.SCALAR_A_MIN,
        "ADAPTIVE_SCALAR_A_MAX": sampler.SCALAR_A_MAX,
        "ADAPTIVE_SCALAR_LAMBDA": sampler.SCALAR_LAMBDA,
        "ADAPTIVE_SCALAR_BIN_LO": sampler.SCALAR_BIN_LO,
        "ADAPTIVE_SCALAR_BIN_WIDTH": sampler.SCALAR_BIN_WIDTH,
        "ADAPTIVE_SCALAR_DEGENERATE": sampler.SCALAR_DEGENERATE,
        "ADAPTIVE_SCALAR_COUNT": sampler.SCALAR_COUNT,
        "ADAPTIVE_BINS": sampler.BINS,
        "ADAPTIVE_STAT_SUM_P": sampler.STAT_SUM_P,
        "ADAPTIVE_STAT_SELECTED": sampler.STAT_SELECTED,
        "ADAPTIVE_STAT_SATURATED": sampler.STAT_SATURATED,
        "ADAPTIVE_STAT_LAMBDA": sampler.STAT_LAMBDA,
        "ADAPTIVE_STAT_I64_STRIDE": sampler.STAT_I64_STRIDE,
        "ADAPTIVE_STAT_MIN_P": sampler.STAT_MIN_P,
        "ADAPTIVE_STAT_MAX_P": sampler.STAT_MAX_P,
        "ADAPTIVE_STAT_PIXELS": sampler.STAT_PIXELS,
        "ADAPTIVE_STAT_U32_STRIDE": sampler.STAT_U32_STRIDE,
        "ADAPTIVE_U_TABLE": ad.U_MODE_TABLE,
        "ADAPTIVE_U_HASH": ad.U_MODE_HASH,
    }
    import re

    def declared(name: str) -> str:
        match = re.search(rf"^static const \w+ {name}\s*=\s*([^;]+);",
                          source, re.MULTILINE)
        assert match, f"{name} is not declared in the shader"
        return match.group(1).strip()

    for name, value in expected.items():
        assert declared(name) == f"{value}u", name
    assert float(declared("ADAPTIVE_HIST_SCALE")) == sampler.HIST_SCALE
    assert float(declared("ADAPTIVE_P_SCALE")) == sampler.P_SCALE


# ------------------------------------------------------- configuration guards

def _validate(extra: list[str]):
    from trainer.config.config_types import validate_training_config
    from trainer.frontend.cli import build_parser

    args = build_parser().parse_args(
        ["--scene", "datasets/360_v2/bicycle"] + extra)
    return validate_training_config(args, n_cams=200, n_train=175, n_test=25)


ADAPTIVE_ON = ["--stbn-adaptive", "--stbn-pixel-fraction", "0.25"]


@pytest.mark.parametrize("extra,needle", [
    (["--stbn-adaptive", "--stbn-pixel-fraction", "1.0"], "sparse budget"),
    (ADAPTIVE_ON + ["--stbn-importance-alpha", "1.0"], "alpha=1"),
    (ADAPTIVE_ON + ["--stbn-importance-alpha", "-0.1"], "must be in [0, 1)"),
    (ADAPTIVE_ON + ["--stbn-importance-sigma", "-1"], "must be >= 0"),
    (ADAPTIVE_ON + ["--stbn-importance-downsample", "0"], "must be >= 1"),
    (ADAPTIVE_ON + ["--stbn-warmup-iters", "100"], "warmup"),
    (ADAPTIVE_ON + ["--stbn-max-fraction", "0.5"], "max-fraction"),
    (ADAPTIVE_ON + ["--stbn-adaptive-uniform", "blue_temporal",
                    "--stbn-independent"], "visit counter"),
    (ADAPTIVE_ON + ["--stbn-adaptive-uniform", "blue_independent"],
     "needs --stbn-independent"),
    (ADAPTIVE_ON + ["--stbn-adaptive-uniform", "blue_independent",
                    "--stbn-independent", "--stbn-source", "white"],
     "white noise"),
])
def test_invalid_adaptive_configurations_are_rejected(extra: list[str],
                                                      needle: str) -> None:
    """Each guard catches a mismatch that would otherwise train silently wrong.

    The ``--stbn-max-fraction`` one is the sharpest: a shrunken
    STBN_MAX_TILE_PIXELS caps the backward's compacted visit list, and the
    adaptive marginal lets a hot tile select MORE than the nominal budget — so
    the cap would truncate exactly the tiles the importance map targeted, dropping
    gradient the loss had already weighted in. That is a biased estimator with no
    error reported anywhere.
    """
    from trainer.config.config_types import ConfigError

    with pytest.raises(ConfigError, match=re.escape(needle)):
        _validate(extra)


def test_valid_adaptive_configurations_are_accepted() -> None:
    for uniform, extra in (("iid", ["--stbn-source", "white",
                                    "--stbn-independent"]),
                           ("blue_independent", ["--stbn-source", "nvidia",
                                                 "--stbn-independent"]),
                           ("blue_temporal", ["--stbn-source", "nvidia"])):
        _validate(ADAPTIVE_ON + ["--stbn-adaptive-uniform", uniform] + extra)


def test_adaptive_defaults_leave_every_existing_regime_untouched() -> None:
    """Without ``--stbn-adaptive`` the feature must be entirely inert.

    The four published regimes are the reference baseline for this whole study, so
    the flag being off has to mean the pre-existing code path, not a
    reconstruction of it.
    """
    from trainer.frontend.app import resolve_adaptive_config
    from trainer.frontend.cli import build_parser

    for extra in ([], ["--stbn-pixel-fraction", "0.25"],
                  ["--stbn-pixel-fraction", "0.25", "--stbn-source", "white",
                   "--stbn-independent"]):
        args = build_parser().parse_args(
            ["--scene", "datasets/360_v2/bicycle"] + extra)
        assert resolve_adaptive_config(args).enabled is False


# --------------------------------------------------------------- on the GPU

BACKENDS = ["vulkan", "d3d12"]
N_SPLATS = 100


def _scene(seed: int = 3):
    rng = np.random.default_rng(seed)
    means = np.zeros((N_SPLATS, 3), dtype=np.float32)
    means[:, 0] = rng.uniform(-0.3, 0.3, N_SPLATS)
    means[:, 1] = rng.uniform(-0.3, 0.3, N_SPLATS)
    means[:, 2] = rng.uniform(2.0, 4.0, N_SPLATS)
    return {
        "means": means,
        "log_scales": np.log(np.full((N_SPLATS, 3), 0.35, dtype=np.float32)),
        "rot_quats": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                             (N_SPLATS, 1)),
        "opacities": rng.uniform(0.3, 0.8, N_SPLATS).astype(np.float32),
        "color_logits": rng.uniform(-2.0, 2.0, (N_SPLATS, 3)).astype(np.float32),
    }


def _target() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yy /= max(H - 1, 1)
    xx /= max(W - 1, 1)
    return np.stack([xx, yy, (1.0 - xx) * yy], axis=-1).astype(np.float32)


def _pipeline(backend: str, *, alpha: float, adaptive: bool = True,
              compacted: bool = True, uniform: str = "blue_temporal",
              unbiased: bool = True, sigma: float = 8.0):
    from _pipeline_testkit import TestPipeline
    from trainer.backend.compute.device import create_device
    from trainer.config.config import (AdaptiveStbnConfig, LossConfig,
                                       StbnConfig)

    _asset_or_skip()
    defines = {"STBN_ENABLED": "1"} if compacted else None
    try:
        device = create_device(backend=backend, defines=defines)
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return TestPipeline(
        device=device, width=W, height=H, max_splats=128,
        loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0),
        stbn_config=StbnConfig(pixel_fraction=RHO, backward_compacted=compacted,
                               independent=(uniform == "blue_independent")),
        adaptive_config=AdaptiveStbnConfig(
            enabled=adaptive, alpha=alpha, sigma=sigma, downsample=4,
            uniform=uniform, unbiased=unbiased),
    )


def _run_loss(pipeline, variables):
    from trainer.data.dataset import simple_camera
    camera = simple_camera(width=W, height=H, eye=(0.0, 0.0, 0.0),
                          target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                          fovy_degrees=60.0)
    pipeline.set_primitive_from_numpy(**_scene())
    pipeline.set_target_image(_target())
    pipeline.forward_and_loss(camera, stbn=variables)
    pipeline.device.wait_for_idle()
    return camera


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpu_alpha_zero_matches_the_uniform_mask_bitwise(backend: str) -> None:
    """THE equivalence test (design section 8).

    At alpha=0 the adaptive path must select the identical pixels the existing
    uniform STBN mask selects for the same slice and density — and the mask is
    read back from the PACKED TABLE the backward will consume, so this pins the
    bytes rather than an intermediate.
    """
    from trainer.config.config import StbnVariables

    slice_index = 11
    pipeline = _pipeline(backend, alpha=0.0)
    _run_loss(pipeline, StbnVariables(slice_index=slice_index,
                                      pixels_per_tile=BUDGET))
    sampler = pipeline.adaptive_sampler

    reference = ad.wrap_to_screen(
        stbn.global_selected_mask(
            dataclasses.replace(_asset_or_skip(), fraction=RHO))[slice_index],
        H, W)
    assert np.array_equal(sampler.read_mask(), reference)
    probabilities = sampler.read_probabilities()
    assert np.all(probabilities == RHO)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpu_alpha_zero_matches_the_uniform_gradient_bitwise(backend: str) -> None:
    """And the resulting image gradient is bit-identical too.

    Mask equality alone would still allow a different WEIGHT. The adaptive path
    applies 1/p_i and the uniform path applies the literal TILE_AREA/k; at alpha=0
    the solver returns lambda = k/TILE_AREA exactly (a dyadic rational), so the two
    agree to the last bit rather than to 1e-7. That is worth pinning, because a
    1e-7 discrepancy would be invisible here and would make every "equivalent"
    claim downstream approximate.
    """
    from trainer.config.config import StbnVariables

    variables = StbnVariables(slice_index=11, pixels_per_tile=BUDGET)
    adaptive = _pipeline(backend, alpha=0.0)
    _run_loss(adaptive, variables)
    adaptive_grad = np.asarray(adaptive.buffers.dL_dC_texture.to_numpy())

    uniform = _pipeline(backend, alpha=0.0, adaptive=False)
    _run_loss(uniform, variables)
    uniform_grad = np.asarray(uniform.buffers.dL_dC_texture.to_numpy())

    assert np.array_equal(adaptive_grad, uniform_grad)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("alpha", [0.5, 0.75, 0.9])
def test_gpu_probability_map_matches_the_host_reference(backend: str,
                                                        alpha: float) -> None:
    """The float32 device path must agree with the float64 host definition.

    Tolerances reflect what float32 can do on a blur plus a bisection, not a
    guess: p to ~1e-4, and the BUDGET — the quantity the estimator's correctness
    depends on — to 1e-3 relative.
    """
    from trainer.config.config import StbnVariables

    pipeline = _pipeline(backend, alpha=alpha)
    # Dense pass first, to capture the h the host reference must be built from.
    _run_loss(pipeline, None)
    h = np.asarray(pipeline.buffers.dL_dC_texture.to_numpy(), dtype=np.float64)
    _run_loss(pipeline, StbnVariables(slice_index=11, pixels_per_tile=BUDGET))

    spec = ad.AdaptiveSpec(fraction=RHO, alpha=alpha, sigma=8.0, downsample=4)
    host, diag = ad.probability_map(ad.screen_importance(h), spec)
    device = pipeline.adaptive_sampler.read_probabilities()

    assert np.abs(device - host).max() < 1e-3
    assert float(device.mean()) == pytest.approx(RHO, rel=1e-3)
    assert diag["p_mean"] == pytest.approx(RHO, abs=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("alpha", [0.0, 0.75])
def test_gpu_packed_table_is_self_consistent(backend: str, alpha: float) -> None:
    """The three packed fields must agree with each other for every tile.

    The backward trusts all three independently — the slot answers "is this pixel
    supervised", the compacted list answers "which pixel is the i-th", the count
    bounds the loop — so a disagreement between them is a gradient that drops or
    double-counts pixels with nothing reporting an error.
    """
    from trainer.config.config import StbnVariables

    pipeline = _pipeline(backend, alpha=alpha)
    _run_loss(pipeline, StbnVariables(slice_index=7, pixels_per_tile=BUDGET))
    sampler = pipeline.adaptive_sampler
    tiles_x, tiles_y = sampler.num_tiles
    packed = np.frombuffer(
        memoryview(sampler.mask_table.to_numpy()), dtype=np.uint32)[
            :tiles_x * tiles_y * TILE_AREA].reshape(tiles_y, tiles_x, TILE_AREA)

    slots = packed & 0xFF
    compacted = (packed >> 8) & 0xFF
    counts = (packed >> 16) & 0xFF
    assert np.all(counts[..., 0:1] == counts), "count must be replicated per tile"
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_slots = slots[ty, tx]
            count = int(counts[ty, tx, 0])
            selected = np.flatnonzero(tile_slots != 0xFF)
            assert selected.size == count
            # Slots are 0..count-1 in ascending pixel order, and the compacted
            # list inverts that map.
            assert np.array_equal(tile_slots[selected], np.arange(count))
            assert np.array_equal(compacted[ty, tx, :count], selected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpu_gradient_is_the_masked_reweighted_dense_one(backend: str) -> None:
    """dL_dC after the adaptive pass is exactly ``M_i h_i / p_i``.

    Checked against the SAME pipeline's dense pass, so the photometric math
    cancels and this is a statement about the mask and the weight alone.
    """
    from trainer.config.config import StbnVariables

    pipeline = _pipeline(backend, alpha=0.75)
    _run_loss(pipeline, None)
    dense = np.asarray(pipeline.buffers.dL_dC_texture.to_numpy(),
                       dtype=np.float64)[..., :3]
    _run_loss(pipeline, StbnVariables(slice_index=5, pixels_per_tile=BUDGET))
    sparse = np.asarray(pipeline.buffers.dL_dC_texture.to_numpy(),
                        dtype=np.float64)[..., :3]

    sampler = pipeline.adaptive_sampler
    mask = sampler.read_mask()
    p = sampler.read_probabilities()

    assert np.all(sparse[~mask] == 0.0), "unsampled pixels must be exactly zero"
    expected = dense[mask] / p[mask][:, None]
    assert np.abs(sparse[mask] - expected).max() < 1e-4 * max(
        float(np.abs(expected).max()), 1.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpu_biased_control_applies_the_global_weight(backend: str) -> None:
    """The biased arm must scale by 1/rho, not by 1/p_i — and must sample from the
    same p as the unbiased arm."""
    from trainer.config.config import StbnVariables

    pipeline = _pipeline(backend, alpha=0.75, unbiased=False)
    _run_loss(pipeline, None)
    dense = np.asarray(pipeline.buffers.dL_dC_texture.to_numpy(),
                       dtype=np.float64)[..., :3]
    _run_loss(pipeline, StbnVariables(slice_index=5, pixels_per_tile=BUDGET))
    sparse = np.asarray(pipeline.buffers.dL_dC_texture.to_numpy(),
                        dtype=np.float64)[..., :3]
    mask = pipeline.adaptive_sampler.read_mask()
    expected = dense[mask] / RHO
    assert np.abs(sparse[mask] - expected).max() < 1e-4 * max(
        float(np.abs(expected).max()), 1.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("uniform", ["iid", "blue_independent", "blue_temporal"])
def test_gpu_every_uniform_mode_hits_the_budget(backend: str,
                                                uniform: str) -> None:
    """All three regimes share the probability map, so all three must realize the
    same expected budget — that is what makes the ladder a controlled
    comparison rather than three different densities."""
    from trainer.config.config import StbnVariables

    pipeline = _pipeline(backend, alpha=0.75, uniform=uniform)
    _run_loss(pipeline, StbnVariables(slice_index=3, pixels_per_tile=BUDGET,
                                      sample_seed=99))
    stats = pipeline.adaptive_sampler.stats_summary()
    assert stats["p_mean"] == pytest.approx(RHO, rel=2e-3), stats
    # A single draw of ~16k Bernoullis has sd ~0.0034 around rho, so 5 sd.
    assert abs(stats["selected_fraction_mean"] - RHO) < 0.02, stats


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpu_compacted_backward_agrees_with_the_full_pixel_reference(
        backend: str) -> None:
    """The compacted visit loop must reproduce the full-pixel backward.

    Same claim the uniform path's equivalence test makes, re-checked here because
    the adaptive marginal changes what the table contains: per-tile counts now
    range over the whole 0..64 span rather than clustering near the nominal
    budget, so a capacity or loop-bound bug that the uniform mask never provoked
    would surface here.
    """
    from trainer.config.config import StbnVariables

    names = ["mean_x", "mean_y", "mean_z", "opacity",
             "color_r", "color_g", "color_b", "rot_w", "rot_x", "rot_y", "rot_z",
             "log_scale_x", "log_scale_y", "log_scale_z"]
    variables = StbnVariables(slice_index=13, pixels_per_tile=BUDGET)

    results = {}
    for compacted in (False, True):
        pipeline = _pipeline(backend, alpha=0.75, compacted=compacted)
        _run_loss(pipeline, variables)
        pipeline.clear_gradients()
        pipeline.backward(stbn=variables)
        pipeline.device.wait_for_idle()
        results[compacted] = pipeline.read_gradients()
        del pipeline

    for name in names:
        reference, fast = results[False][name], results[True][name]
        scale = max(1e-6, float(np.max(np.abs(reference))))
        worst = float(np.max(np.abs(reference - fast)) / scale)
        assert worst < 0.02, f"{name}: worst relative disagreement {worst:.4f}"
