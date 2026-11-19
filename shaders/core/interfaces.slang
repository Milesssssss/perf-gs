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

#endif // TRAINER_CORE_INTERFACES_SLANG
