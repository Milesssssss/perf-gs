// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Gaussian storage and optimizer-state structs. Python reads their layout by reflection.

#ifndef TRAINER_GAUSSIAN_TYPES
#define TRAINER_GAUSSIAN_TYPES

struct Gaussian3D
{
    float3 mean;
    float3 log_scale;
    float4 rot_quat;   // (w, x, y, z)
    float opacity;     // pre-sigmoid logit
    float3 sh0;        // linear RGB = 0.5 + C0 * sh0
};

// Offsets of each field in the flat gradient/optimizer channel order.
namespace GaussianChannel
{
    static const uint Mean = 0u;
    static const uint LogScale = 3u;
    static const uint Rotation = 6u;
    static const uint Opacity = 10u;
    static const uint Color = 11u;
    static const int Count = 14;
}

void normalizeGaussian3D(inout Gaussian3D primitive)
{
    primitive.rot_quat = normalize(primitive.rot_quat);
}

// Adam (m, v) per optimized field, in Gaussian3D field order.
struct AdamMoments
{
    float3 mean_m;
    float3 mean_v;
    float3 log_scale_m;
    float3 log_scale_v;
    float4 rot_m;
    float4 rot_v;
    float opacity_m;
    float opacity_v;
    float3 color_m;
    float3 color_v;
};

#include "../../common/adam_precision.slang"

#if defined(ADAM_MOMENTS_BF16)
struct AdamState
{
    uint3 mean_mv;       // bf16 (m, v)
    uint3 log_scale_mv;
    uint4 rot_mv;
    uint opacity_mv;
    uint3 color_mv;
};

AdamMoments loadAdamMoments(AdamState s)
{
    AdamMoments m;
    m.mean_m = bf16Lo(s.mean_mv);
    m.mean_v = bf16Hi(s.mean_mv);
    m.log_scale_m = bf16Lo(s.log_scale_mv);
    m.log_scale_v = bf16Hi(s.log_scale_mv);
    m.rot_m = bf16Lo(s.rot_mv);
    m.rot_v = bf16Hi(s.rot_mv);
    m.opacity_m = bf16Lo(s.opacity_mv);
    m.opacity_v = bf16Hi(s.opacity_mv);
    m.color_m = bf16Lo(s.color_mv);
    m.color_v = bf16Hi(s.color_mv);
    return m;
}

AdamState storeAdamMoments(AdamMoments m, uint seed)
{
    AdamState s;
    uint rng = seed;
    s.mean_mv = packBf16(m.mean_m, m.mean_v, rng);
    s.log_scale_mv = packBf16(m.log_scale_m, m.log_scale_v, rng);
    s.rot_mv = packBf16(m.rot_m, m.rot_v, rng);
    s.opacity_mv = packBf16(m.opacity_m, m.opacity_v, rng);
    s.color_mv = packBf16(m.color_m, m.color_v, rng);
    return s;
}
#else
struct AdamState
{
    AdamMoments moments;
};

AdamMoments loadAdamMoments(AdamState s)
{
    return s.moments;
}

AdamState storeAdamMoments(AdamMoments m, uint seed)
{
    AdamState s;
    s.moments = m;
    return s;
}
#endif

#endif // TRAINER_GAUSSIAN_TYPES
