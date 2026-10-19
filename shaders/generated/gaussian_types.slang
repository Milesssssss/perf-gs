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

// Per-gaussian Adam first/second moments (AoS): the adam kernel loads
// all m/v for one gaussian, steps, writes back (no atomics). One
// (m, v) pair per optimized field, in Gaussian3D field order.
struct AdamState
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

#endif // TRAINER_GAUSSIAN_TYPES
