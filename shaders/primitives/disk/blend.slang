// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_DISK_BLEND_SLANG
#define TRAINER_DISK_BLEND_SLANG

#include "types.slang"
#include "../../core/interfaces.slang"

struct DiskPrimitive : IBlendablePrimitive
{
    float2 center;
    float log_radius;
    float opacity;
    float3 color_value;

    [Differentiable]
    float coverage(no_diff float2 pixel)
    {
        float radius = exp(log_radius);
        float distance = length(pixel - center);
        float t = clamp(1.0 - distance / radius, 0.0, 1.0);
        float smooth = t * t * (3.0 - 2.0 * t);
        return (1.0 / (1.0 + exp(-opacity))) * smooth;
    }

    [Differentiable]
    float3 color()
    {
        return color_value;
    }

    static DiskPrimitive make(Disk2D disk)
    {
        DiskPrimitive primitive;
        primitive.center = disk.center;
        primitive.log_radius = disk.log_radius;
        primitive.opacity = disk.opacity;
        primitive.color_value = disk.color_value;
        return primitive;
    }
}

struct DiskSource : IPrimitiveSource
{
    StructuredBuffer<Disk2D> primitives;
    typealias Primitive = DiskPrimitive;

    DiskPrimitive primitiveAt(uint index)
    {
        return DiskPrimitive.make(primitives[index]);
    }
}

// Scalar objective that exercises both generic primitive methods in one reverse pass.
[Differentiable]
float primitiveLinearObjective<P : IBlendablePrimitive>(
    P primitive,
    no_diff float2 pixel,
    no_diff float coverage_seed,
    no_diff float3 color_seed)
{
    return primitive.coverage(pixel) * coverage_seed
         + dot(primitive.color(), color_seed);
}

#endif // TRAINER_DISK_BLEND_SLANG
