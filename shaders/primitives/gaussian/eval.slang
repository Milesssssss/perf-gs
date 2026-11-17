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

// Conservative Mahalanobis bound q = d^T Sigma^-1 d within which alpha can reach the 1/255 floor;
// <= 0 means the splat never reaches it. amplitude comes from splatAmplitude.
float splatCutoffQ(float amplitude, float opacity_logit)
{
    // The value of m^p at which alpha falls to the floor.
#if SPLAT_KERNEL == 0
    float floor_exponent = log(255.0 * amplitude);
#else
    float floor_exponent = log(amplitude / BLEND_MIN_OPTICAL_DEPTH);
#endif
#if SPLAT_KERNEL == 3 || defined(SPLAT_PROFILE_POWER)
    return floor_exponent > 0.0
        ? 2.0 * pow(floor_exponent, 1.0 / splatProfilePower(opacity_logit)) : floor_exponent;
#else
    return 2.0 * floor_exponent;
#endif
}

// Peak optical scale per splat: alpha = A*exp(-m^p) (gaussian) or 1 - exp(-A*exp(-m^p)) (density kernels).
[Differentiable]
float splatAmplitude(float opacity_logit, float opacity_compensation)
{
#if SPLAT_KERNEL == 1
    return opacity_compensation * softplus(opacity_logit);
#elif SPLAT_KERNEL == 2 || SPLAT_KERNEL == 3
    return opacity_compensation * exp(min(opacity_logit, LOG_DENSITY_MAX));
#else
    return opacity_compensation * sigmoid(opacity_logit);
#endif
}

// Cull-time bound before projection, when the Mip compensation (<= 1) is not known yet.
float splatCutoffQ(float opacity_logit)
{
    return splatCutoffQ(splatAmplitude(opacity_logit, 1.0), opacity_logit);
}

// Inverse 2D covariance in the same (00, 01, 11) packing; projection dilation keeps det > 0.
[Differentiable]
float3 conicFromCov(float3 cov2d)
{
    float inv_det = 1.0 / (cov2d.x * cov2d.z - cov2d.y * cov2d.y);
    return float3(cov2d.z, -cov2d.y, cov2d.x) * inv_det;
}

// Alpha at a pixel from the per-splat record bin_count precomputes; the logit only shapes opaque_core.
[Differentiable]
float splatCoverage(no_diff float2 pixel, float2 mean_2d, float3 conic, float amplitude,
                    float opacity_logit)
{
    float2 d = pixel - mean_2d;
    float m = 0.5 * (conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y);
    float exponent = splatFalloffExponent(m, opacity_logit);
#if SPLAT_KERNEL == 0
    return amplitude * exp(-exponent);
#else
    return 1.0 - exp(-amplitude * exp(-exponent));
#endif
}

[Differentiable]
float evalGaussian2DFromCov(
    no_diff float2 pixel,
    float2 mean_2d,
    float3 cov2d,
    float opacity_logit,
    float opacity_compensation)   // Mip-Splatting alpha factor (differentiable; 1.0 = off)
{
    return splatCoverage(pixel, mean_2d, conicFromCov(cov2d),
                         splatAmplitude(opacity_logit, opacity_compensation), opacity_logit);
}

#endif // TRAINER_GAUSSIAN_EVAL_SLANG
