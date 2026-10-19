// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Core contracts for the generic differentiable blendable-primitive rasterizer.
//
// A "blend record" is the compact, per-pixel-evaluable form a primitive takes
// inside the tile blend loop. It is IDifferentiable so the backward rasterizer
// can flow per-pixel gradients into the primitive's stored fields via bwd_diff
// (validated on vulkan+d3d12 by the P0 spike, shaders/_spike/generic_spike.slang).
//
// Each primitive supplies only the two differentiable per-pixel quantities the
// blend loop needs: coverage() (its alpha / falloff) and color(). HOW those
// accumulate front-to-back — the extinction / compositing — is the Compositor's
// job (core/compositor.slang), kept orthogonal so either can vary on its own.

#ifndef TRAINER_CORE_INTERFACES_SLANG
#define TRAINER_CORE_INTERFACES_SLANG

#include "../common/camera.slang"   // CameraData (projection contract)

interface IBlendablePrimitive : IDifferentiable
{
    // Per-pixel coverage/alpha BEFORE the blend loop's ALPHA_MAX clamp and
    // ALPHA_MIN skip. For a 3D gaussian this is sigmoid(opacity) * comp *
    // exp(-0.5 * dᵀ Σ⁻¹ d); a 2D disk would return a smoothstep falloff.
    [Differentiable] float coverage(no_diff float2 pixel);

    // View-independent linear-RGB color the covered pixel receives. For a
    // gaussian this is the SH band-0 (DC) activation shDC(sh0).
    [Differentiable] float3 color();
}

[Differentiable]
float primitiveCoverage<P : IBlendablePrimitive>(P prim, no_diff float2 pixel)
{
    return prim.coverage(pixel);
}

interface IPrimitiveSource
{
    // The concrete primitive type this source yields (e.g. GaussianPrimitive).
    associatedtype Primitive : IBlendablePrimitive;

    // Gather + assemble the primitive record at `idx`. Not differentiable: the
    // storage load is a plain gather; the diff path runs through the returned
    // primitive's coverage()/color().
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
    // Unclamped screen-space AABB of the footprint. Returns false when the
    // primitive has no extent (fully culled); out params are then undefined.
    bool bounds(out float2 out_min, out float2 out_max);

    // Pick the walk axis: true iterates ROWS (per-row column spans via
    // rowTiles), false iterates COLUMNS (per-column row spans via colTiles).
    // Choosing the shorter axis keeps the per-strip crossing tests minimal.
    bool iterateRows(uint2 tile_min, uint2 tile_max);

    // For tile row `ty`, the inclusive column range [tx_lo, tx_hi] the footprint
    // actually crosses within [tile_min_x, tile_max_x]. False if the row is
    // untouched (skip it).
    bool rowTiles(float2 bbox_min, float2 bbox_max, uint ty,
                  uint tile_min_x, uint tile_max_x, out uint tx_lo, out uint tx_hi);

    // Column twin of rowTiles: the inclusive row range [ty_lo, ty_hi] crossed in
    // tile column `tx` within [tile_min_y, tile_max_y]. False if untouched.
    bool colTiles(float2 bbox_min, float2 bbox_max, uint tx,
                  uint tile_min_y, uint tile_max_y, out uint ty_lo, out uint ty_hi);
}

#endif // TRAINER_CORE_INTERFACES_SLANG
