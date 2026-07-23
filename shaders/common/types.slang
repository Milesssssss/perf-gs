// Shared type definitions for the v0 forward-only pipeline.
// Kept intentionally minimal: no Adam state, no gradients, no SH.

#ifndef TRAINER_TYPES_SLANG
#define TRAINER_TYPES_SLANG

#include "camera.slang"
#include "fixed_point.slang"

// Gaussian3D (packed storage) + AdamState (Adam moments) are GENERATED from the
// host<->shader schema in src/trainer/contracts.py — the single source of truth
// for the primitive layout. See that module for the field semantics (logit
// opacity, SH0 DC color, quaternion (w,x,y,z) order).
#include "generated/gaussian_types.slang"

struct Projected2DGaussian
{
    float2 mean;            // screen-space center (in pixels)
    float3 cov2d;           // 2x2 covariance packed as (cov_00, cov_01, cov_11)
    float depth_vs;         // view-space depth (positive-z is forward in our convention)
    float opacity_compensation; // Mip-Splatting: coef3d*coef2d, multiplies alpha in
                                // the blend (1.0 when --mip-splatting is off). Computed
                                // once in bin_count, reused by forward + backward.
};

// Compact per-gaussian record for the blend backward's HOT random gather.
// The blend backward reads splats in sorted (tile, depth) order, so g_idx is
// a random permutation and gaussians[g_idx] / projected_2d[g_idx] are scatter
// gathers. Those two fat structs total 80 B but the blend uses only these 4
// fields (36 B): gathering the whole 80 B pollutes L2 (~45% useful bytes).
// This packs exactly what the blend needs into one 36 B record so the gather
// is ~100% useful. Written by bin_count (which already has all four values in
// hand), consumed by backward_persplat. NOT read back to the CPU.
struct BlendData
{
    float2 mean;            // == Projected2DGaussian.mean
    float3 cov2d;           // == Projected2DGaussian.cov2d
    float  opacity;         // == Gaussian3D.opacity (pre-sigmoid logit)
    float3 sh0;             // == Gaussian3D.sh0
    float  opacity_compensation; // == Projected2DGaussian.opacity_compensation (mip)
};

#endif // TRAINER_TYPES_SLANG
