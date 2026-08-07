// Evaluate a 2D gaussian at a pixel, given its screen-space mean,
// packed (cov_00, cov_01, cov_11) covariance, and pre-sigmoid opacity
// logit. Returns the opacity-modulated alpha: ``sigmoid(opacity_logit) *
// exp(power)``.
//
// Opacity is parameterized as an unbounded logit (like gsplat): this
// stops runaway prune when Adam's opacity LR is applied directly to a
// near-zero probability, and puts effective gradient magnitude for
// opacity into the same regime as the other parameters.
//
// `max(det, 1e-20)` is kept as a true div-by-zero guard at the numerical
// floor — not a "safe magnitude" clamp.

#ifndef TRAINER_GAUSSIAN_EVAL_SLANG
#define TRAINER_GAUSSIAN_EVAL_SLANG

#include "../../common/math.slang"

//[PreferRecompute]
[Differentiable]
float evalGaussian2DFromCov(
    no_diff float2 pixel,
    float2 mean_2d,
    float3 cov2d,
    float opacity_logit,
    float opacity_compensation)   // Mip-Splatting alpha factor (differentiable; 1.0 = off)
{
    float2 d = pixel - mean_2d;

    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det <= 0.0)
        return 0.0;

    float inv_det = 1.0 / det;
    float cov_inv_00 =  cov2d.z * inv_det;
    float cov_inv_01 = -cov2d.y * inv_det;
    float cov_inv_11 =  cov2d.x * inv_det;

    float power = -0.5 * (d.x * d.x * cov_inv_00
                        + 2.0 * d.x * d.y * cov_inv_01
                        + d.y * d.y * cov_inv_11);

    if (power > 0.0)
        return 0.0;

    // Mip-Splatting: alpha = sigmoid(opacity) * (coef3d*coef2d) * exp(power).
    // opacity_compensation is DIFFERENTIABLE: bwd_diff yields dL/dcomp (routed
    // back to scale/cov by projection_backward) AND scales dL/dopacity,
    // dL/dcov2d by comp — matching mip-splatting's backward. 1.0 when off.
    return sigmoid(opacity_logit) * opacity_compensation * exp(power);
}

// SPLAT-INVARIANT precompute for the tile blend loops. A lane owns one splat and
// evaluates its coverage at up to 256 pixels; det, 1/det, the inverse-covariance
// entries, and sigmoid(opacity) do NOT depend on the pixel, so computing them
// once per splat instead of once per (pixel,splat) hoists a divide + an exp +
// ~7 muls out of the 256-iteration inner loop. The compiler does not reliably
// hoist these itself across the loop's early-continues + the [PreferRecompute]
// sigmoid, so we do it explicitly. Bit-identical arithmetic to the per-pixel
// eval (same operations, just evaluated once).
struct GaussCovConst
{
    float s;                   // sigmoid(opacity_logit)  (pure — comp kept apart)
    float comp;                // mip opacity_compensation (1.0 when off)
    float s_comp;              // s * comp (folded scale for the per-pixel alpha)
    float inv_det;
    float ci00, ci01, ci11;    // inverse-covariance entries
    float2 mean;               // screen-space center
    bool  det_positive;        // det > 0 (else the gaussian never contributes)
};

GaussCovConst gaussCovPrecompute(
    float2 mean_2d, float3 cov2d, float opacity_logit, float opacity_compensation)
{
    GaussCovConst c;
    c.mean = mean_2d;
    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    c.det_positive = (det > 0.0);
    float inv_det = c.det_positive ? (1.0 / det) : 0.0;
    c.inv_det = inv_det;
    c.ci00 =  cov2d.z * inv_det;
    c.ci01 = -cov2d.y * inv_det;
    c.ci11 =  cov2d.x * inv_det;
    c.s = sigmoid(opacity_logit);          // pure sigmoid (backward needs s*(1-s))
    c.comp = opacity_compensation;
    c.s_comp = c.s * opacity_compensation; // folded scale for the alpha product
    return c;
}

// Reusable intermediates of one evalGaussian2DFromCov evaluation, so the
// backward can be computed WITHOUT re-running the expensive primal (the exp,
// sigmoid, and 1/det that dominate the coverage cost). The backward rasterizer
// already evaluates coverage once to get alpha; caching lets the gradient reuse
// that work instead of a second full eval (autodiff or hand-code both otherwise
// recompute the primal — measured ~10ms, the bulk of the backward kernel).
struct GaussCovCache
{
    float  e;             // exp(power)  (only meaningful when valid)
    float  s;             // sigmoid(opacity_logit)
    float  inv_det;
    float  ci00, ci01, ci11;   // inverse-covariance entries
    float2 d;             // pixel - mean_2d
    bool   valid;         // det > 0 AND power <= 0 (i.e. alpha != 0)
};

// Per-pixel coverage from the splat-invariant precompute. Returns alpha and
// fills the cache for the backward. Only the per-pixel work runs here: d, the
// quadratic form, exp(power), and the two early-outs. Same arithmetic as
// evalGaussian2DFromCov, just with the splat-invariant terms precomputed.
float evalGaussian2DAtPixel(
    float2 pixel, GaussCovConst pre, out GaussCovCache cache)
{
    cache = (GaussCovCache)0;
    cache.valid = false;
    cache.inv_det = pre.inv_det;
    cache.ci00 = pre.ci00; cache.ci01 = pre.ci01; cache.ci11 = pre.ci11;
    cache.s = pre.s;   // pure sigmoid; comp is applied via pre.s_comp below
    float2 d = pixel - pre.mean;
    cache.d = d;

    if (!pre.det_positive)
        return 0.0;

    float power = -0.5 * (d.x * d.x * pre.ci00
                        + 2.0 * d.x * d.y * pre.ci01
                        + d.y * d.y * pre.ci11);
    if (power > 0.0)
        return 0.0;

    cache.e = exp(power);
    cache.valid = true;
    return pre.s_comp * cache.e;   // sigmoid * comp * exp(power)
}

// Cached analytic backward of evalGaussian2DFromCov: consumes the intermediates
// from evalGaussian2DAtPixel (GaussCovCache) instead of recomputing the primal.
// This is the exact same Jacobian Slang's bwd_diff produces (verified vs autodiff to
// fp32 epsilon in tests/test_gradients.py), returning the 4 gradients:
//   dL/d(mean2d), dL/d(cov2d) packed (00,01,11), dL/d(opacity logit), dL/d(comp).
// `d_alpha` is the upstream dL/dalpha. When !cache.valid (culled: det<=0 or
// power>0 => alpha==0) all grads are 0, matching the forward's contribution mask.
//
// Derivation: s=sigmoid(o), e=exp(power), alpha=s*comp*e.
//   power = -0.5*Q*inv_det, Q = dx*dx*czz - 2*dx*dy*cxy + dy*dy*cxx.
//   dL/dpower = d_alpha*alpha ; dpower/dmean = (dx*ci00+dy*ci01, dx*ci01+dy*ci11).
//   dpower/dc = -0.5*(dQ/dc*inv_det + Q*dinv_det/dc), with dinv_det/dcxx =
//   -inv_det^2*czz, /dczz = -inv_det^2*cxx, /dcxy = +2*cxy*inv_det^2, and
//   dQ/dcxx=dy*dy, dQ/dczz=dx*dx, dQ/dcxy=-2*dx*dy.
void evalGaussian2DFromCov_bwd_cached(
    GaussCovCache cache, float3 cov2d, float opacity_compensation, float d_alpha,
    out float2 d_mean, out float3 d_cov2d, out float d_opacity, out float d_comp)
{
    d_mean = float2(0.0);
    d_cov2d = float3(0.0);
    d_opacity = 0.0;
    d_comp = 0.0;
    if (!cache.valid) return;

    float s = cache.s, e = cache.e;
    float2 d = cache.d;
    float alpha = s * opacity_compensation * e;

    d_comp = d_alpha * s * e;
    d_opacity = d_alpha * opacity_compensation * e * s * (1.0 - s);

    float dL_dpower = d_alpha * alpha;
    d_mean.x = dL_dpower * (d.x * cache.ci00 + d.y * cache.ci01);
    d_mean.y = dL_dpower * (d.x * cache.ci01 + d.y * cache.ci11);

    float Q = d.x * d.x * cov2d.z - 2.0 * d.x * d.y * cov2d.y + d.y * d.y * cov2d.x;
    float inv_det2 = cache.inv_det * cache.inv_det;
    float dpow_dcxx = -0.5 * (d.y * d.y * cache.inv_det - Q * inv_det2 * cov2d.z);
    float dpow_dczz = -0.5 * (d.x * d.x * cache.inv_det - Q * inv_det2 * cov2d.x);
    float dpow_dcxy = -0.5 * (-2.0 * d.x * d.y * cache.inv_det + Q * inv_det2 * 2.0 * cov2d.y);
    d_cov2d = dL_dpower * float3(dpow_dcxx, dpow_dcxy, dpow_dczz);
}

#endif // TRAINER_GAUSSIAN_EVAL_SLANG
