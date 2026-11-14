// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Core interfaces of the generic differentiable blendable-primitive rasterizer: primitives supply
// differentiable coverage() and color(); compositing is the Compositor's job (core/compositor.slang).

#ifndef TRAINER_CORE_INTERFACES_SLANG
#define TRAINER_CORE_INTERFACES_SLANG

#include "../common/camera.slang"   // CameraData (projection contract)

interface IBlendablePrimitive : IDifferentiable
{
    // Alpha before the blend loop's ALPHA_MAX clamp and ALPHA_MIN skip.
    [Differentiable] float coverage(no_diff float2 pixel);

    [Differentiable] float3 color();
}

[Differentiable]
float primitiveCoverage<P : IBlendablePrimitive>(P prim, no_diff float2 pixel)
{
    return prim.coverage(pixel);
}

interface IPrimitiveSource
{
    associatedtype Primitive : IBlendablePrimitive;

    // Not differentiable: gradients flow through the returned primitive's coverage()/color().
    Primitive primitiveAt(uint idx);
}

interface IProjectableModel : IDifferentiable
{
    [Differentiable]
    void projectInto(no_diff CameraData cam,
                     no_diff float filter_3D, no_diff float kernel_size, no_diff uint mip_enabled,
                     out float2 mean2d, out float3 cov2d, out float depth_vs,
                     out float cov2d_det, out float opacity_compensation);
}

[Differentiable]
void projectModel<M : IProjectableModel>(
    M model, no_diff CameraData cam,
    no_diff float filter_3D, no_diff float kernel_size, no_diff uint mip_enabled,
    out float2 mean2d, out float3 cov2d, out float depth_vs,
    out float cov2d_det, out float opacity_compensation)
{
    model.projectInto(cam, filter_3D, kernel_size, mip_enabled,
                      mean2d, cov2d, depth_vs, cov2d_det, opacity_compensation);
}

interface IScreenFootprint
{
    // Unclamped screen-space AABB; false (outputs undefined) when the primitive has no extent.
    bool bounds(out float2 out_min, out float2 out_max);

    // True walks rows (rowTiles), false columns (colTiles); the shorter axis minimizes crossing tests.
    bool iterateRows(uint2 tile_min, uint2 tile_max);

    // Inclusive column range [tx_lo, tx_hi] crossed in tile row `ty`; false if the row is untouched.
    bool rowTiles(float2 bbox_min, float2 bbox_max, uint ty,
                  uint tile_min_x, uint tile_max_x, out uint tx_lo, out uint tx_hi);

    // Column twin of rowTiles.
    bool colTiles(float2 bbox_min, float2 bbox_max, uint tx,
                  uint tile_min_y, uint tile_max_y, out uint ty_lo, out uint ty_hi);
}

#endif // TRAINER_CORE_INTERFACES_SLANG
