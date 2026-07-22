// Gaussian conformance to the core IBlendable contract.
//
// The compact per-pixel blend record — screen-space mean, packed 2D covariance,
// pre-sigmoid opacity logit, SH band-0 color, and the Mip-Splatting opacity
// compensation — is exactly the field set BlendData packs. coverage()/color()
// delegate to the existing differentiable Gaussian primitives (evalGaussian2D-
// FromCov, shDC), so the tile blend loop can be written generically over
// IBlendable with ZERO change to the Gaussian math or its gradients.

#ifndef TRAINER_GAUSSIAN_BLEND_SLANG
#define TRAINER_GAUSSIAN_BLEND_SLANG

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "eval.slang"
#include "../../core/interfaces.slang"

struct GaussianBlend : IBlendable
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

    // Assemble from the values the forward already has in hand (the same
    // fields BlendData packs). Non-differentiable constructor: the diff path
    // runs through coverage()/color() on the constructed record.
    static GaussianBlend make(float2 mean, float3 cov2d, float opacity,
                              float3 sh0, float opacity_compensation)
    {
        GaussianBlend g;
        g.mean = mean;
        g.cov2d = cov2d;
        g.opacity = opacity;
        g.sh0 = sh0;
        g.opacity_compensation = opacity_compensation;
        return g;
    }
}

#endif // TRAINER_GAUSSIAN_BLEND_SLANG
