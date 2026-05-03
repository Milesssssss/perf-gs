// Evaluate a 2D gaussian at a pixel, given its screen-space mean,
// packed (cov_00, cov_01, cov_11) covariance, and scalar opacity.
// Returns the opacity-modulated alpha contribution `opacity * exp(power)`.
//
// `max(det, 1e-20)` is kept as a true div-by-zero guard at the numerical
// floor — not a "safe magnitude" clamp.

#ifndef TRAINER_GAUSSIAN_EVAL_SLANG
#define TRAINER_GAUSSIAN_EVAL_SLANG

[Differentiable]
float evalGaussian2DFromCov(
    no_diff float2 pixel,
    float2 mean_2d,
    float3 cov2d,
    float opacity)
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

    return opacity * exp(power);
}

#endif // TRAINER_GAUSSIAN_EVAL_SLANG
