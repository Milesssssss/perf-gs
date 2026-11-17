// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Gaussian conformance to the core IBlendablePrimitive contract: coverage() is splatCoverage on the
// per-view record bin_count precomputes, so the generic blend loop keeps the Gaussian math and gradients.

#ifndef TRAINER_GAUSSIAN_BLEND_SLANG
#define TRAINER_GAUSSIAN_BLEND_SLANG

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "eval.slang"
#include "../../core/interfaces.slang"

struct GaussianPrimitive : IBlendablePrimitive
{
    float2 mean;                    // screen-space center (pixels)
    float3 conic;                   // packed inverse covariance (00, 01, 11)
    float  amplitude;               // splatAmplitude(opacity logit, Mip compensation)
    float  opacity;                 // pre-sigmoid logit
    float3 rgb;                     // this view's color

    [Differentiable]
    float coverage(no_diff float2 pixel)
    {
        return splatCoverage(pixel, mean, conic, amplitude, opacity);
    }

    [Differentiable]
    float3 color()
    {
        return rgb;
    }

    // Non-differentiable constructor; gradients flow through coverage()/color() on the record.
    static GaussianPrimitive make(BlendData bd)
    {
        GaussianPrimitive g;
        g.mean = bd.mean;
        g.conic = bd.conic;
        g.amplitude = bd.amplitude;
        g.opacity = bd.opacity;
        g.rgb = bd.color;
        return g;
    }

    // Zero coverage everywhere: masked backward lanes stay finite scan identities.
    static GaussianPrimitive none()
    {
        GaussianPrimitive g;
        g.mean = float2(0.0);
        g.conic = float3(0.0);
        g.amplitude = 0.0;
        g.opacity = 0.0;
        g.rgb = float3(0.0);
        return g;
    }
}

// Gathers the compact 40 B BlendData record (one gather, better L2 hit rate in sorted order);
// matches backward_persplat's blend_data gather so forward and backward read identical primitives.
struct GaussianSource : IPrimitiveSource
{
    StructuredBuffer<BlendData> blend_data;

    typealias Primitive = GaussianPrimitive;

    GaussianPrimitive primitiveAt(uint idx)
    {
        return GaussianPrimitive.make(blend_data[idx]);
    }
}

#endif // TRAINER_GAUSSIAN_BLEND_SLANG
