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

[Differentiable]
[PreferRecompute]
float evalGaussian2DFromCov(
    no_diff float2 pixel,
    float2 mean_2d,
    float3 cov2d,
    float opacity_logit)
{
    float2 d = pixel - mean_2d;

    float det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    det = max(det, 1e-20);

    float inv_det = 1.0 / det;
    float cov_inv_00 =  cov2d.z * inv_det;
    float cov_inv_01 = -cov2d.y * inv_det;
    float cov_inv_11 =  cov2d.x * inv_det;

    float power = -0.5 * (d.x * d.x * cov_inv_00
                        + 2.0 * d.x * d.y * cov_inv_01
                        + d.y * d.y * cov_inv_11);

    return sigmoid(opacity_logit) * exp(power);
}

#endif // TRAINER_GAUSSIAN_EVAL_SLANG
