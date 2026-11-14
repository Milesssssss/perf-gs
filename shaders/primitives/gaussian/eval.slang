// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Alpha of a 2D gaussian at a pixel from its screen-space mean, packed (cov_00, cov_01, cov_11)
// covariance and pre-sigmoid opacity logit; SPLAT_KERNEL picks how the footprint becomes alpha.

#ifndef TRAINER_GAUSSIAN_EVAL_SLANG
#define TRAINER_GAUSSIAN_EVAL_SLANG

#include "../../common/math.slang"

// Compile-time splat kernel (CLI --splat-kernel); F = opacity_compensation * exp(power):
//   0 gaussian:    alpha = sigmoid(l) * F.
//   1 density:     alpha = 1 - exp(-softplus(l) * F) = 1 - (1 - sigmoid(l))^F.
//   2 log_density: alpha = 1 - exp(-exp(l) * F).
//   3 opaque_core: log_density whose falloff power rises from 1 toward SPLAT_PROFILE_POWER with sigmoid(l).
#ifndef SPLAT_KERNEL
#define SPLAT_KERNEL 0
#endif

// SPLAT_PROFILE_POWER p (defined only when != 1): F = comp * exp(-m^p), m = d^T Sigma^-1 d / 2.
#ifdef SPLAT_PROFILE_POWER
static const float SPLAT_PROFILE_MAX = SPLAT_PROFILE_POWER;
#else
static const float SPLAT_PROFILE_MAX = 1.0;
#endif

// opaque_core's power ramps with sigmoid(l - SPLAT_PROFILE_ONSET): a splat sharpens only once that opaque.
#ifndef SPLAT_PROFILE_ONSET
#define SPLAT_PROFILE_ONSET 0.0
#endif

// Keeps the optical depth and its gradient finite; exp(20) already gives a ~6 sigma opaque core.
static const float LOG_DENSITY_MAX = 20.0;
// Optical depth at which alpha reaches the blend loop's 1/255 floor.
static const float BLEND_MIN_OPTICAL_DEPTH = 0.0039292783;   // -log(1 - 1/255)

[Differentiable]
float splatProfilePower(float opacity_logit)
{
#if SPLAT_KERNEL == 3
    return 1.0 + (SPLAT_PROFILE_MAX - 1.0) * sigmoid(opacity_logit - SPLAT_PROFILE_ONSET);
#else
    return SPLAT_PROFILE_MAX;
#endif
}

// m^p. The floor on m keeps pow's derivative finite at a splat's exact center.
[Differentiable]
float splatFalloffExponent(float m, float opacity_logit)
{
#if SPLAT_KERNEL == 3 || defined(SPLAT_PROFILE_POWER)
    return pow(max(m, 1e-12), splatProfilePower(opacity_logit));
#else
    return m;
#endif
}

// Conservative Mahalanobis bound q = d^T Sigma^-1 d within which alpha can reach the 1/255 floor
// (ignores the Mip compensation <= 1); <= 0 means the splat never reaches it.
float splatCutoffQ(float opacity_logit)
{
    // The value of m^p at which alpha falls to the floor.
#if SPLAT_KERNEL == 1
    float floor_exponent = log(softplus(opacity_logit) / BLEND_MIN_OPTICAL_DEPTH);
#elif SPLAT_KERNEL == 2 || SPLAT_KERNEL == 3
    float floor_exponent = min(opacity_logit, LOG_DENSITY_MAX) - log(BLEND_MIN_OPTICAL_DEPTH);
#else
    float floor_exponent = log(255.0 * sigmoid(opacity_logit));
#endif
#if SPLAT_KERNEL == 3 || defined(SPLAT_PROFILE_POWER)
    return floor_exponent > 0.0
        ? 2.0 * pow(floor_exponent, 1.0 / splatProfilePower(opacity_logit)) : floor_exponent;
#else
    return 2.0 * floor_exponent;
#endif
}

[Differentiable]
float evalGaussian2DFromCov_out(
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

    // Mip-Splatting alpha; comp stays differentiable so dL/dcomp reaches scale/cov via projection_backward.
    return sigmoid(opacity_logit) * opacity_compensation * exp(power);
}

[Differentiable]
float evalGaussian2DFromCov(
    no_diff float2 pixel,
    float2 mean_2d,
    float3 cov2d,
    float opacity_logit,
    float opacity_compensation)
{
    float2 d = pixel - mean_2d;

    float q =
        cov2d.z * d.x * d.x
        - 2.0 * cov2d.y * d.x * d.y
        + cov2d.x * d.y * d.y;

    float det =
        cov2d.x * cov2d.z
        - cov2d.y * cov2d.y;

#if SPLAT_KERNEL == 0 && !defined(SPLAT_PROFILE_POWER)
    return opacity_compensation * sigmoid(opacity_logit) * exp(-0.5 * q / det);
#else
    float exponent = splatFalloffExponent(0.5 * q / det, opacity_logit);
#if SPLAT_KERNEL == 1
    return 1.0 - exp(-softplus(opacity_logit) * opacity_compensation * exp(-exponent));
#elif SPLAT_KERNEL == 2 || SPLAT_KERNEL == 3
    return 1.0 - exp(-opacity_compensation
                     * exp(min(opacity_logit, LOG_DENSITY_MAX) - exponent));
#else
    return opacity_compensation * sigmoid(opacity_logit) * exp(-exponent);
#endif
#endif
}

#endif // TRAINER_GAUSSIAN_EVAL_SLANG
