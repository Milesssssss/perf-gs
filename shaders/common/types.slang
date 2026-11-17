// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Shared type definitions.

#ifndef TRAINER_TYPES_SLANG
#define TRAINER_TYPES_SLANG

#include "camera.slang"
#include "fixed_point.slang"
#include "primitives/gaussian/types.slang"

struct Projected2DGaussian
{
    float2 mean;            // screen-space center (in pixels)
    float3 cov2d;           // 2x2 covariance packed as (cov_00, cov_01, cov_11)
    float depth_vs;         // view-space depth (positive-z is forward in our convention)
    float opacity_compensation; // Mip-Splatting alpha factor coef3d*coef2d (1.0 when off)
};

// Per-view splat record the blend passes gather, precomputed once by bin_count (and sh_forward for
// color) so the per-pixel loops evaluate only a quadratic form and one exp.
struct BlendData
{
    float2 mean;            // == Projected2DGaussian.mean
    float3 conic;           // conicFromCov(Projected2DGaussian.cov2d)
    float  amplitude;       // splatAmplitude(opacity logit, Mip compensation)
    float3 color;           // this view's color: shDC(sh0), or the SH evaluation
    float  opacity;         // == Gaussian3D.opacity (pre-sigmoid logit); shapes opaque_core only
};

#endif // TRAINER_TYPES_SLANG
