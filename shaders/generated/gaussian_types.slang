// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// DO NOT EDIT. Generated from src/trainer/contracts.py.
// Regenerate with:  python -m trainer.contracts
// Validated not-stale by tests/test_contracts.py.

#ifndef TRAINER_GAUSSIAN_TYPES
#define TRAINER_GAUSSIAN_TYPES

// Packed per-gaussian storage; uploaded verbatim by
// trainer.buffer_io.write_gaussians (GAUSSIAN_FLOAT_COUNT float32s).
struct Gaussian3D
{
    float3 mean;  // world-space position
    float3 log_scale;  // per-axis log scale; exp(log_scale) is the 3D gaussian scale
    float4 rot_quat;  // rotation quaternion (w, x, y, z)
    float opacity;  // pre-sigmoid opacity logit; sigmoid(opacity) is the probability
    float3 sh0;  // SH band-0 (DC) color coeff; linear RGB = 0.5 + C0*sh0
};

// Flat optimizer/gradient channel offsets.
namespace GaussianChannel
{
    static const uint Mean = 0u;  // 3 scalar(s)
    static const uint LogScale = 3u;  // 3 scalar(s)
    static const uint Rotation = 6u;  // 4 scalar(s)
    static const uint Opacity = 10u;  // 1 scalar(s)
    static const uint Color = 11u;  // 3 scalar(s)
    static const int Count = 14;
}

// Apply every schema-declared storage invariant after optimization.
void normalizeGaussian3D(inout Gaussian3D primitive)
{
    primitive.rot_quat = normalize(primitive.rot_quat);
}

// Adam first/second moments in float32, the precision the optimizer
// computes in. One (m, v) pair per optimized field, in Gaussian3D
// field order.
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

#include "../common/adam_precision.slang"

// Stored per-gaussian moments (AoS): the adam kernel loads all m/v
// for one gaussian, steps, writes back (no atomics).
#if defined(ADAM_MOMENTS_BF16)
struct AdamState
{
    uint3 mean_mv;  // bf16 (m, v)
    uint3 log_scale_mv;  // bf16 (m, v)
    uint4 rot_mv;  // bf16 (m, v)
    uint opacity_mv;  // bf16 (m, v)
    uint3 color_mv;  // bf16 (m, v)
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
