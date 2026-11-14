// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Gaussian conformance to the core IBlendablePrimitive contract: coverage()/color() delegate to
// evalGaussian2DFromCov and shDC, so the generic blend loop keeps the Gaussian math and gradients.

#ifndef TRAINER_GAUSSIAN_BLEND_SLANG
#define TRAINER_GAUSSIAN_BLEND_SLANG

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "eval.slang"
#include "../../core/interfaces.slang"

struct GaussianPrimitive : IBlendablePrimitive
{
    float2 mean;                    // screen-space center (pixels)
    float3 cov2d;                   // packed (cov_00, cov_01, cov_11)
    float  opacity;                 // pre-sigmoid logit
    float3 sh0;                     // SH band-0 (DC) coeff
    float  opacity_compensation;    // Mip-Splatting coef3d*coef2d (1.0 = off)

    [Differentiable]
    float coverage(no_diff float2 pixel)
    {
        return evalGaussian2DFromCov(pixel, mean, cov2d, opacity, opacity_compensation);
    }

    [Differentiable]
    float3 color()
    {
        return shDC(sh0);
    }

    // Non-differentiable constructor; gradients flow through coverage()/color() on the record.
    static GaussianPrimitive make(float2 mean, float3 cov2d, float opacity,
                                  float3 sh0, float opacity_compensation)
    {
        GaussianPrimitive g;
        g.mean = mean;
        g.cov2d = cov2d;
        g.opacity = opacity;
        g.sh0 = sh0;
        g.opacity_compensation = opacity_compensation;
        return g;
    }
}

// Gathers the compact 36 B BlendData record (one gather, better L2 hit rate in sorted order);
// matches backward_persplat's blend_data gather so forward and backward read identical primitives.
struct GaussianSource : IPrimitiveSource
{
    StructuredBuffer<BlendData> blend_data;

    typealias Primitive = GaussianPrimitive;

    GaussianPrimitive primitiveAt(uint idx)
    {
        BlendData bd = blend_data[idx];
        return GaussianPrimitive.make(bd.mean, bd.cov2d, bd.opacity,
                                      bd.sh0, bd.opacity_compensation);
    }
}

#endif // TRAINER_GAUSSIAN_BLEND_SLANG
