// Gaussian conformance to the core IBlendablePrimitive contract.
//
// The compact per-pixel blend record — screen-space mean, packed 2D covariance,
// pre-sigmoid opacity logit, SH band-0 color, and the Mip-Splatting opacity
// compensation — is exactly the field set BlendData packs. coverage()/color()
// delegate to the existing differentiable Gaussian primitives (evalGaussian2D-
// FromCov, shDC), so the tile blend loop can be written generically over
// IBlendablePrimitive with ZERO change to the Gaussian math or its gradients.

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

    // Assemble from the values the forward already has in hand (the same
    // fields BlendData packs). Non-differentiable constructor: the diff path
    // runs through coverage()/color() on the constructed record.
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

// Gaussian source for the generic blend loop: yields a GaussianPrimitive per
// index. Gathers the compact 36 B BlendData record bin_count packs (mean,
// cov2d, opacity, sh0, opacity_compensation) — the SAME repacked fields as the
// fat gaussians[] + projected_2d[] pair, but ONE gather instead of two. The
// blend reads primitives in random (sorted) order, so gathering 36 B not 80 B
// lifts the L2 hit rate; this matches backward_persplat's blend_data gather
// exactly, so forward + backward read the primitive identically.
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
