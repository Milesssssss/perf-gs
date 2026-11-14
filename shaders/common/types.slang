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

// Just the fields the blend backward's random (tile, depth)-order gather needs, so it does not pull
// the full Gaussian3D and Projected2DGaussian into L2. Written by bin_count.
struct BlendData
{
    float2 mean;            // == Projected2DGaussian.mean
    float3 cov2d;           // == Projected2DGaussian.cov2d
    float  opacity;         // == Gaussian3D.opacity (pre-sigmoid logit)
    float3 sh0;             // == Gaussian3D.sh0
    float  opacity_compensation; // == Projected2DGaussian.opacity_compensation (mip)
};

#endif // TRAINER_TYPES_SLANG
