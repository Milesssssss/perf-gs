# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Importance-guided sparse backward sampling: the math, as a host reference.

The existing sparse regimes (``full``, ``iid_white``, ``indep_blue``, ``stbn``)
supervise a fixed fraction rho of pixels chosen with a UNIFORM marginal: every
pixel is equally likely, and only the ARRANGEMENT of the chosen set differs
between them. This module adds the other axis — spending the same expected
budget preferentially where the screen-space loss gradient is large — while
leaving the arrangement machinery untouched. Nothing here modifies any existing
regime; ``trainer.backend.render.stbn`` is still the reference baseline.

THE DECOMPOSITION, and it is the whole design:

    the importance field decides WHERE computation goes  (the marginal p_i)
    the blue-noise field decides HOW those samples are ARRANGED (the U_i,t)

Concretely, with U_{i,t} in [0,1) the existing STBN scalar for pixel i at visit t,

    M_{i,t} = 1[ U_{i,t} < p_i ]

so P(M=1) = p_i comes entirely from the importance map and the joint spatial /
temporal structure comes entirely from U. The STBN scalar is never warped, added
to, multiplied, ranked or otherwise touched — ONLY the threshold moves. Setting
p_i = rho for every pixel recovers the existing global-threshold mask exactly
(see :func:`uniform_volume` for why that equality is bitwise, not approximate).

THE PIPELINE, in the order the GPU runs it:

  1. s_i     = ||dL/dC_i||_2 from the DENSE image-space loss pass. This is the
               exact screen-space gradient, not a proxy for it: the loss kernels
               already materialize dL/dC for the whole image before the expensive
               Gaussian backward is dispatched, so no fallback to a per-pixel
               error magnitude is needed (see the module note below).
  2. s~      = G_sigma * s, smoothed at a scale set in OUTPUT-IMAGE pixels and
               deliberately independent of the 8x8 rasterizer tiling
               (:func:`smooth_importance`).
  3. a_i     = (1-alpha) + alpha * s~_i / (mean(s~) + eps), so mean(a) = 1 and
               alpha=0 is exactly uniform (:func:`strength`).
  4. lambda  solves sum_i min(1, lambda a_i) = rho N by bisection, so the
               EXPECTED budget is held at rho N even where the map saturates
               (:func:`solve_lambda`). Clamping a normalized map instead would
               lose the clipped mass and undershoot the budget.
  5. p_i     = min(1, lambda a_i), treated as a constant: nothing differentiates
               through the probabilities or the sampling.
  6. h'_i    = M_i h_i / p_i, the inverse-probability-weighted gradient, whose
               conditional expectation is h_i — so the Gaussian gradient built
               from it is an unbiased estimator of the DENSE one, and the
               training objective is unchanged (:func:`ipw_scale`).

WHY THE SMOOTHING. |J_i^T h_i| is what we would like to sample proportionally to
and cannot afford to know; h_i alone is available but is a very noisy stand-in
for it, because L1's gradient is sign-valued and a single pixel's h says little
about whether the Gaussians covering it need work. Smoothing at sigma ~ 8 output
pixels turns it into a statement about a NEIGHBOURHOOD, which is the scale a
splat actually covers. It also keeps the marginal field low-frequency, which
matters for the diagnostics: a nonuniform density necessarily injects
low-frequency structure into the raw binary mask, so blue-noise quality has to be
read off the variance-normalized residual (M_i - p_i)/sqrt(p_i(1-p_i)) rather
than off M itself.

THIS MODULE IS THE DEFINITION. The GPU path (:mod:`adaptive_sampler` +
``shaders/losses/adaptive_importance.slang``) reimplements it in float32 for
speed; ``tests/test_adaptive_stbn.py`` pins the two together. The offline
sampling and calibration diagnostics run against this module, so a claim about
"the probability field" is a claim about one implementation, not two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trainer.backend.render.stbn import (SOURCE_NVIDIA, SOURCE_WHITE,
                                         StbnMaskSpec, pixels_per_tile,
                                         scalar_field)
from trainer.contracts.codegen import TILE_AREA

# Which uniform sequence the threshold is compared against. These are the ONLY
# difference between the three adaptive regimes — the probability map is
# identical, which is what makes the ladder a controlled comparison.
#
#   iid              a fresh independent uniform per pixel per visit
#   blue_independent a spatially blue-noise uniform field per visit, slice drawn
#                    independently each step (the existing ``indep_blue``
#                    semantics: NVIDIA volume, --stbn-independent)
#   blue_temporal    the same volume walked by each camera's own visit counter
#                    (the existing ``stbn`` semantics)
UNIFORM_IID = "iid"
UNIFORM_BLUE_INDEPENDENT = "blue_independent"
UNIFORM_BLUE_TEMPORAL = "blue_temporal"
UNIFORM_MODES = (UNIFORM_IID, UNIFORM_BLUE_INDEPENDENT, UNIFORM_BLUE_TEMPORAL)

# Shader-side enum for the uniform source; must match ADAPTIVE_U_* in
# shaders/losses/adaptive_importance.slang.
U_MODE_TABLE = 0
U_MODE_HASH = 1

# The importance proxy. ``screen_grad`` is the preferred (and the only
# implemented) mode: s_i = ||dL/dC_i||_2 from the dense image-space loss pass.
#
# The spec this was built to allows a fallback to a per-pixel loss/error
# magnitude IF the exact screen-space gradient cannot be obtained before the
# Gaussian backward. In this architecture it can: ImageLoss.record writes dL/dC
# for the entire image, and the mask has always been applied in that pass rather
# than inside the rasterizer. So the fallback is not needed and is deliberately
# NOT implemented — an error-magnitude mode would be a strictly worse proxy
# offered for no architectural reason.
IMPORTANCE_SCREEN_GRAD = "screen_grad"

# mean(s~) can only be zero if s~ is identically zero (it is non-negative), in
# which case any positive epsilon yields the same a = 1-alpha. So this guards a
# division by zero without biasing any reachable configuration.
EPSILON = 1e-12

# Number of histogram bins the GPU lambda solver uses. Mirrored here so the host
# reference can reproduce (and a test can bound) its quantization error.
LAMBDA_BINS = 1024


@dataclass(frozen=True)
class AdaptiveSpec:
    """Everything that determines the probability map and the sampling process.

    ``fraction`` is the requested rho; the budget actually enforced is
    :meth:`budget_fraction`, quantized exactly like the existing uniform path so
    the two are density-matched to the last pixel (see that method).
    """

    fraction: float = 0.25
    alpha: float = 0.75
    # Smoothing scale in OUTPUT-IMAGE pixels. 0 disables smoothing entirely and
    # samples the raw per-pixel s_i, which is the sigma=0 arm of the sweep.
    sigma: float = 8.0
    # Resolution the importance field is evaluated at, as a divisor of the image
    # dimensions. Chosen for cost, not for structure: the blur is applied at this
    # resolution and reconstructed bilinearly, and :func:`coarse_sigma`
    # compensates the extra variance so the composed operator still smooths at
    # ``sigma`` full-res pixels. NOT related to the 8x8 rasterizer tile and NOT
    # related to the 128x128 STBN texture; a per-tile importance field is exactly
    # the construction this design rejects.
    downsample: int = 4
    uniform: str = UNIFORM_BLUE_TEMPORAL
    # False = the DIAGNOSTIC control: sample from p_i but omit the 1/p_i weight,
    # i.e. deliberately optimize a different (error-weighted) objective. Never
    # the proposed method. See :func:`ipw_scale` for the global rescale it keeps.
    unbiased: bool = True
    importance: str = IMPORTANCE_SCREEN_GRAD
    # Only meaningful for the blue_* uniform modes.
    source: str = SOURCE_NVIDIA
    file_path: str = ""
    seed: int = 0

    def validate(self) -> None:
        if self.uniform not in UNIFORM_MODES:
            raise ValueError(
                f"unknown adaptive uniform mode {self.uniform!r}; expected one "
                f"of {list(UNIFORM_MODES)}")
        if self.importance != IMPORTANCE_SCREEN_GRAD:
            raise ValueError(
                f"unknown importance source {self.importance!r}; only "
                f"{IMPORTANCE_SCREEN_GRAD!r} is implemented (see the module "
                f"docstring for why the error-magnitude fallback is not)")
        if not 0.0 <= self.alpha < 1.0:
            # alpha == 1 would let a zero-importance pixel have p_i = 0 exactly
            # and therefore an infinite 1/p_i weight if it were ever selected;
            # the estimator is only defined for p_i > 0.
            raise ValueError(
                f"--stbn-importance-alpha must be in [0, 1), got {self.alpha}. "
                f"alpha=1 admits p_i=0 pixels, for which the inverse-probability "
                f"weight is undefined.")
        if self.sigma < 0.0:
            raise ValueError(f"--stbn-importance-sigma must be >= 0, got {self.sigma}")
        if int(self.downsample) < 1:
            raise ValueError(
                f"--stbn-importance-downsample must be >= 1, got {self.downsample}")
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError(
                f"adaptive pixel fraction must be in (0, 1], got {self.fraction}")

    def budget_fraction(self) -> float:
        """The rho the budget constraint is actually solved for.

        Quantized to ``pixels_per_tile(rho)/TILE_AREA`` — the SAME quantization
        the existing uniform path bakes into its table and its TILE_AREA/k
        weight. Two reasons, both correctness:

        1. Comparing an adaptive run against a uniform one at "rho = 0.10" must
           compare equal budgets. The uniform path's real density at rho=0.10 is
           round(6.4)/64 = 0.09375, not 0.10. Solving the adaptive constraint for
           0.10 would hand the adaptive arm 6.7% more supervised pixels than its
           control — a confound large enough to manufacture the result.
        2. It makes the alpha=0 case reduce to the uniform path EXACTLY rather
           than approximately: p_i = k/TILE_AREA, whose reciprocal is the
           TILE_AREA/k the loss shaders apply.
        """
        return pixels_per_tile(float(self.fraction), TILE_AREA) / float(TILE_AREA)

    @property
    def uses_table(self) -> bool:
        return self.uniform in (UNIFORM_BLUE_INDEPENDENT, UNIFORM_BLUE_TEMPORAL)

    @property
    def u_mode(self) -> int:
        return U_MODE_TABLE if self.uses_table else U_MODE_HASH

    @property
    def independent_slice(self) -> bool:
        """Whether the slice index is drawn per step rather than per camera visit."""
        return self.uniform != UNIFORM_BLUE_TEMPORAL


# ---------------------------------------------------------------------------
# The smooth importance field
# ---------------------------------------------------------------------------

def resolve_downsample(sigma: float, downsample: int) -> int:
    """The coarse-grid divisor actually used.

    Capped at ``floor(sigma)`` so the coarse-resolution blur always has a
    kernel wider than one cell — past that point the downsample IS the smoothing
    and the nominal sigma stops meaning anything. sigma=0 forces 1, i.e. the raw
    per-pixel field with no reconstruction at all.
    """
    if sigma <= 0.0:
        return 1
    return max(1, min(int(downsample), int(np.floor(sigma))))


def coarse_sigma(sigma: float, downsample: int) -> float:
    """Gaussian sigma to apply AT THE COARSE GRID, in coarse cells.

    The composed operator is (box average over D x D) then (Gaussian sigma_c)
    then (bilinear reconstruction). Treating the three as independent blurs,
    their variances in full-res pixels add:

        sigma^2  =  D^2/12   +   (D sigma_c)^2   +   D^2/6
                    box          gaussian           bilinear (triangle, width D)

    so sigma_c = sqrt(sigma^2 - D^2/4) / D. Solving for sigma_c rather than just
    using sigma/D is what keeps ``--stbn-importance-sigma 8`` a statement about
    the OUTPUT IMAGE: without it the effective scale would be ~8.3 px at D=4 and
    would drift further as D grew, so the sigma sweep would not be measuring
    sigma. The variance addition is exact for the Gaussian factor and approximate
    for the box and triangle (both are near-Gaussian at these widths);
    ``test_effective_sigma_tracks_the_request`` bounds the residual error.
    """
    if downsample <= 1:
        return float(sigma)
    variance = float(sigma) ** 2 - (float(downsample) ** 2) / 4.0
    return float(np.sqrt(max(variance, 0.0)) / float(downsample))


def gaussian_kernel(sigma: float) -> np.ndarray:
    """Normalized 1D Gaussian taps, radius ``ceil(3 sigma)``. Length 1 at sigma=0."""
    if sigma <= 0.0:
        return np.ones(1, dtype=np.float64)
    radius = int(np.ceil(3.0 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    taps = np.exp(-0.5 * (offsets / sigma) ** 2)
    return taps / taps.sum()


def _blur_axis(field: np.ndarray, taps: np.ndarray, axis: int) -> np.ndarray:
    """One separable pass with EDGE-CLAMPED boundaries.

    Clamped rather than wrapped or zero-padded: the image is not toroidal, and
    zero padding would depress the importance of a border region purely because
    it is at a border, which would then show up as a systematically lower p there.
    Clamping is also what the shader does, so the two agree.
    """
    if taps.size == 1:
        return field.astype(np.float64, copy=True)
    radius = taps.size // 2
    length = field.shape[axis]
    out = np.zeros(field.shape, dtype=np.float64)
    for offset, weight in zip(range(-radius, radius + 1), taps):
        index = np.clip(np.arange(length) + offset, 0, length - 1)
        out += weight * np.take(field, index, axis=axis)
    return out


def box_downsample(field: np.ndarray, downsample: int) -> np.ndarray:
    """Mean of each ``D x D`` block; partial blocks at the right/bottom edge are
    averaged over their VALID pixels only, so an edge cell is not biased toward
    zero by pixels that do not exist."""
    if downsample <= 1:
        return field.astype(np.float64, copy=True)
    height, width = field.shape
    ch, cw = coarse_dims(width, height, downsample)
    padded = np.zeros((ch * downsample, cw * downsample), dtype=np.float64)
    counts = np.zeros_like(padded)
    padded[:height, :width] = field
    counts[:height, :width] = 1.0
    blocks = padded.reshape(ch, downsample, cw, downsample).sum(axis=(1, 3))
    valid = counts.reshape(ch, downsample, cw, downsample).sum(axis=(1, 3))
    return blocks / np.maximum(valid, 1.0)


def coarse_dims(width: int, height: int, downsample: int) -> tuple[int, int]:
    """``(coarse_height, coarse_width)`` for an image of ``(height, width)``."""
    d = max(1, int(downsample))
    return (int(-(-int(height) // d)), int(-(-int(width) // d)))


def bilinear_upsample(coarse: np.ndarray, height: int, width: int,
                      downsample: int) -> np.ndarray:
    """Reconstruct a full-resolution field from the coarse grid.

    Uses the half-pixel-centred mapping ``c = (x + 0.5)/D - 0.5`` with clamped
    edges — the same convention as a GPU bilinear sample, and the same one the
    shader implements by hand. Identity at D=1.
    """
    if downsample <= 1:
        return coarse.astype(np.float64, copy=True)
    ch, cw = coarse.shape
    d = float(downsample)

    def axis_weights(size: int, limit: int):
        centre = (np.arange(size, dtype=np.float64) + 0.5) / d - 0.5
        centre = np.clip(centre, 0.0, limit - 1.0)
        lo = np.floor(centre).astype(np.int64)
        hi = np.minimum(lo + 1, limit - 1)
        return lo, hi, centre - lo

    y0, y1, fy = axis_weights(height, ch)
    x0, x1, fx = axis_weights(width, cw)
    top = coarse[y0][:, x0] * (1.0 - fx)[None, :] + coarse[y0][:, x1] * fx[None, :]
    bottom = coarse[y1][:, x0] * (1.0 - fx)[None, :] + coarse[y1][:, x1] * fx[None, :]
    return top * (1.0 - fy)[:, None] + bottom * fy[:, None]


def screen_importance(image_gradient: np.ndarray) -> np.ndarray:
    """``s_i = ||h_i||_2`` from an ``(H, W, >=3)`` dL/dC image.

    RGB only. The fourth channel of the gradient texture is dL/dA (alpha
    supervision, weight 0 by default), which is not part of the dC the Jacobian
    J_i = dC_i/dtheta acts on, so including it would change what the proxy is
    proxying for.
    """
    grad = np.asarray(image_gradient, dtype=np.float64)
    if grad.ndim != 3 or grad.shape[2] < 3:
        raise ValueError(
            f"image gradient must be (H, W, >=3), got {grad.shape}")
    return np.sqrt((grad[..., :3] ** 2).sum(axis=-1))


def smooth_importance(s: np.ndarray, spec: AdaptiveSpec) -> np.ndarray:
    """``s~ = G_sigma * s`` at full resolution, via the coarse grid."""
    height, width = s.shape
    d = resolve_downsample(spec.sigma, spec.downsample)
    coarse = box_downsample(np.asarray(s, dtype=np.float64), d)
    taps = gaussian_kernel(coarse_sigma(spec.sigma, d))
    coarse = _blur_axis(_blur_axis(coarse, taps, axis=1), taps, axis=0)
    return bilinear_upsample(coarse, height, width, d)


def strength(s_tilde: np.ndarray, alpha: float,
             epsilon: float = EPSILON) -> np.ndarray:
    """``a_i = (1-alpha) + alpha * s~_i / (mean(s~) + eps)``.

    ``mean(a) = 1`` by construction (up to the epsilon), which is what makes the
    lambda bracket in :func:`solve_lambda` tight and makes alpha=0 exactly
    uniform rather than approximately so.
    """
    field = np.asarray(s_tilde, dtype=np.float64)
    mean = float(field.mean())
    return (1.0 - alpha) + alpha * field / (mean + epsilon)


# ---------------------------------------------------------------------------
# The budget constraint
# ---------------------------------------------------------------------------

def expected_count(a: np.ndarray, lam: float) -> float:
    return float(np.minimum(1.0, lam * np.asarray(a, dtype=np.float64)).sum())


def solve_lambda(a: np.ndarray, rho: float, *, iterations: int = 100,
                 tolerance: float = 1e-10) -> float:
    """Scalar ``lambda > 0`` with ``sum_i min(1, lambda a_i) = rho N``.

    ``sum_i min(1, lambda a_i)`` is continuous and non-decreasing in lambda, so
    bisection is unconditionally robust here — no assumption about the shape of
    the importance field, which a Newton iteration on this piecewise-linear
    function would need.

    The bracket needs no search: the sum is 0 at lambda=0 and N >= rho N at
    ``lambda = 1/min(a)``, where every pixel saturates. Deliberately started at 0
    rather than at the analytically tempting ``rho`` (which would be the exact
    answer if ``mean(a)`` were exactly 1 and nothing saturated): a float32 mean
    can land a hair above 1, putting the true root just BELOW rho and outside
    such a bracket. Sixty bisection steps over [0, 1/min(a)] resolve lambda to
    ~1e-17 regardless, so the tighter bracket buys nothing for the risk.

    A degenerate field (all a equal, i.e. alpha=0) is solved analytically rather
    than by bisection, so that case is EXACT: p_i = rho and 1/p_i is bit-for-bit
    the TILE_AREA/k the existing uniform path applies. Bisection would land
    within ~1e-7 of it, which is close enough not to move any mask bit but not
    close enough to claim bitwise gradient equality.
    """
    field = np.asarray(a, dtype=np.float64)
    if field.size == 0:
        return float(rho)
    lo_a, hi_a = float(field.min()), float(field.max())
    if hi_a <= 0.0:
        # Every pixel has zero importance: fall back to the uniform marginal.
        return float(rho)
    if hi_a - lo_a <= 0.0:
        return float(rho) / hi_a
    target = float(rho) * field.size
    lo, hi = 0.0, 1.0 / max(lo_a, 1e-30)
    if expected_count(field, hi) < target:
        # Unreachable budget: too much of the field is pinned at a_i ~ 0 for the
        # expected count to reach rho N even at p=1 everywhere else. Cannot
        # happen for alpha < 1 (then a_i >= 1-alpha > 0 and hi saturates all of
        # them), so this is a guard, not a supported regime.
        return hi
    for _ in range(int(iterations)):
        mid = 0.5 * (lo + hi)
        value = expected_count(field, mid)
        if abs(value - target) <= tolerance * field.size:
            return mid
        if value < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def probabilities(a: np.ndarray, lam: float) -> np.ndarray:
    """``p_i = min(1, lambda a_i)``, the per-pixel inclusion probability."""
    return np.minimum(1.0, float(lam) * np.asarray(a, dtype=np.float64))


def probability_map(s: np.ndarray, spec: AdaptiveSpec) -> tuple[np.ndarray, dict]:
    """``(p, diagnostics)`` for a raw importance image ``s``.

    The one entry point the offline diagnostics use, so every figure and every
    table is built from the same construction the trainer runs.
    """
    spec.validate()
    rho = spec.budget_fraction()
    s_tilde = smooth_importance(s, spec)
    a = strength(s_tilde, spec.alpha)
    lam = solve_lambda(a, rho)
    p = probabilities(a, lam)
    return p, probability_diagnostics(p, lam, rho)


def probability_diagnostics(p: np.ndarray, lam: float, rho: float) -> dict:
    """The debugging record section 4 of the design asks to log every time."""
    field = np.asarray(p, dtype=np.float64)
    saturated = float((field >= 1.0 - 1e-9).mean()) if field.size else 0.0
    mean = float(field.mean()) if field.size else 0.0
    return {
        "lambda": float(lam),
        "rho_target": float(rho),
        "p_mean": mean,
        "p_min": float(field.min()) if field.size else 0.0,
        "p_max": float(field.max()) if field.size else 0.0,
        "saturated_fraction": saturated,
        "expected_samples": mean * field.size,
        "n_pixels": int(field.size),
        "budget_absolute_error": abs(mean - float(rho)),
    }


# ---------------------------------------------------------------------------
# The correlated uniform variable
# ---------------------------------------------------------------------------

def uniform_volume(spec: StbnMaskSpec) -> np.ndarray:
    """``(slices, height, width)`` in [0,1): the STBN scalar as a UNIFORM.

    Built as the per-slice GLOBAL normalized rank, ``(rank + 0.5) / (H*W)``,
    rather than by rescaling the published 8-bit values. Two properties follow,
    and both are load-bearing:

    * Each slice's values are an exact permutation of the uniform grid, so
      ``P(U < q) = round-to-grid(q)`` with no dependence on the asset's
      quantization. A value threshold would land mid-level — the 128x128 asset
      packs 64 pixels into each of 256 levels — and overshoot or undershoot the
      requested density by tens of pixels.
    * Thresholding at ``q = k/TILE_AREA`` reproduces
      :func:`stbn.global_selected_mask` EXACTLY: that function keeps the lowest
      ``k * blocks`` ranks, and ``(rank + 0.5)/16384 < k*256/16384`` iff
      ``rank < k*256``. That equality is what makes the alpha=0 arm of the
      adaptive path a bitwise reproduction of the existing uniform mask rather
      than a near-copy of it.

    Ranking is per SLICE, over the whole slice — not per tile. A per-tile
    normalization would be a different sampling process (see the ``tile_rank``
    discussion in :mod:`trainer.backend.render.stbn`).
    """
    spec.validate()
    volume = scalar_field(spec)
    slices, height, width = volume.shape
    total = height * width
    out = np.empty((slices, total), dtype=np.float64)
    for index in range(slices):
        order = np.argsort(volume[index].reshape(-1), kind="stable")
        ranks = np.empty(total, dtype=np.float64)
        ranks[order] = np.arange(total, dtype=np.float64)
        out[index] = (ranks + 0.5) / float(total)
    return out.reshape(slices, height, width)


def spec_to_mask_spec(spec: AdaptiveSpec) -> StbnMaskSpec:
    """The volume-identifying :class:`StbnMaskSpec` behind a blue_* uniform mode.

    ``fraction`` is irrelevant to the U volume (a normalized rank does not depend
    on a density) but the dataclass requires one; it is left at the spec's so a
    cache key stays readable.
    """
    return StbnMaskSpec(source=spec.source, path=spec.file_path,
                        seed=spec.seed, fraction=float(spec.fraction))


def wrap_to_screen(field: np.ndarray, height: int, width: int) -> np.ndarray:
    """Tile a (h, w) field over an (height, width) screen, as the shaders do."""
    rows = np.arange(height) % field.shape[0]
    cols = np.arange(width) % field.shape[1]
    return field[np.ix_(rows, cols)]


def hash_uniform(height: int, width: int, seed: int) -> np.ndarray:
    """Per-pixel IID uniforms from the SAME integer hash the shader uses.

    A host-side ``default_rng`` would be a different stream, so the ``iid``
    regime's host reference and GPU path would sample different masks and the
    calibration diagnostic would not be measuring the shipped code. Wang hash of
    (x, y, seed), 24 bits mantissa.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.uint32)
    key = (xx * np.uint32(0x9E3779B1)) ^ (yy * np.uint32(0x85EBCA77)) \
        ^ np.uint32(seed & 0xFFFFFFFF)
    key = _wang(key)
    return (key >> np.uint32(8)).astype(np.float64) / float(1 << 24)


def _wang(key: np.ndarray) -> np.ndarray:
    """Thomas Wang's 32-bit integer hash, in numpy, matching adaptiveHash()."""
    with np.errstate(over="ignore"):
        key = np.asarray(key, dtype=np.uint32)
        key = (key ^ np.uint32(61)) ^ (key >> np.uint32(16))
        key = key * np.uint32(9)
        key = key ^ (key >> np.uint32(4))
        key = key * np.uint32(0x27D4EB2D)
        key = key ^ (key >> np.uint32(15))
    return key


def mask_from_uniform(u: np.ndarray, p: np.ndarray) -> np.ndarray:
    """``M_i = 1[U_i < p_i]`` — the whole sampling step.

    Strictly less-than, matching the shader and matching the rank identity in
    :func:`uniform_volume`.
    """
    return np.asarray(u) < np.asarray(p)


def ipw_scale(p: np.ndarray, spec: AdaptiveSpec) -> np.ndarray:
    """The multiplier applied to ``h_i`` for a selected pixel.

    ``1/p_i`` for the proposed estimator: E[M_i h_i / p_i] = h_i, so the Gaussian
    gradient is unbiased for the dense one and the training objective is
    untouched. Note there is NO extra 1/N: the loss kernels use the
    SUM-over-pixels convention (see shaders/losses/l1_l2_loss.slang), so h_i is
    already the derivative of the quantity the dense backward accumulates.

    For the biased control, ``1/rho`` — a GLOBAL constant, not per pixel. The
    control's purpose is to ask whether re-weighting the OBJECTIVE toward
    high-error regions helps; leaving out the 1/p entirely would also shrink the
    gradient by a factor ~rho, which changes the effective learning rate and,
    worse, changes what the densifier's ABSOLUTE --grad-thresh means. Rescaling
    by the constant 1/rho holds the overall magnitude matched to the unbiased arm
    so the comparison isolates the spatial reweighting.
    """
    field = np.asarray(p, dtype=np.float64)
    if spec.unbiased:
        return np.where(field > 0.0, 1.0 / np.maximum(field, 1e-30), 0.0)
    return np.full(field.shape, 1.0 / spec.budget_fraction(), dtype=np.float64)


def residual(mask: np.ndarray, p: np.ndarray,
             epsilon: float = 1e-9) -> np.ndarray:
    """``Z_i = (M_i - p_i) / sqrt(p_i(1-p_i) + eps)``.

    The right object to take a spectrum of when the density is nonuniform. A
    variable-density binary mask necessarily carries the low-frequency content of
    p itself, so the raw spectrum of M would report "less blue" for any adaptive
    method purely because its mean field is not flat. Centring by p and
    normalizing by the Bernoulli standard deviation removes exactly that, leaving
    the arrangement — which is the thing the three uniform sources differ in.
    """
    q = np.asarray(p, dtype=np.float64)
    return (np.asarray(mask, dtype=np.float64) - q) / np.sqrt(q * (1.0 - q) + epsilon)


__all__ = [
    "AdaptiveSpec", "EPSILON", "IMPORTANCE_SCREEN_GRAD", "LAMBDA_BINS",
    "SOURCE_NVIDIA", "SOURCE_WHITE", "UNIFORM_BLUE_INDEPENDENT",
    "UNIFORM_BLUE_TEMPORAL", "UNIFORM_IID", "UNIFORM_MODES", "U_MODE_HASH",
    "U_MODE_TABLE", "bilinear_upsample", "box_downsample", "coarse_dims",
    "coarse_sigma", "expected_count", "gaussian_kernel", "hash_uniform",
    "ipw_scale", "mask_from_uniform", "probabilities",
    "probability_diagnostics", "probability_map", "residual",
    "resolve_downsample", "screen_importance", "smooth_importance",
    "solve_lambda", "spec_to_mask_spec", "strength", "uniform_volume",
    "wrap_to_screen",
]
