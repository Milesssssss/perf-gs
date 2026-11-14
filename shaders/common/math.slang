// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Differentiable math utilities shared by the gaussian primitive.
// Kept lean in v0: only what the forward pass actually calls.

#ifndef TRAINER_MATH_SLANG
#define TRAINER_MATH_SLANG

[Differentiable]
[PreferRecompute(SideEffectBehavior.Allow)]
float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

[Differentiable]
[PreferRecompute(SideEffectBehavior.Allow)]
float3 sigmoid3(float3 v)
{
    return float3(sigmoid(v.x), sigmoid(v.y), sigmoid(v.z));
}

// log(1 + e^x) without overflow. Note softplus(x) == -log(1 - sigmoid(x)).
[Differentiable]
[PreferRecompute(SideEffectBehavior.Allow)]
float softplus(float x)
{
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

// Spherical-harmonics band-0 (DC) color activation, matching 3DGS/INRIA:
//   color = 0.5 + C0 * sh0,   C0 = 0.5 * sqrt(1/pi).
// This is an AFFINE (unbounded) map, unlike the old sigmoid. Its Jacobian is
// the constant C0, so it does NOT attenuate the color gradient for bright
// values the way sigmoid'(x)=sigma*(1-sigma) collapses toward white — which
// is the whole reason for the switch. Band-0 is view-independent (constant
// over directions); higher bands would add view dependence later.
static const float SH_C0 = 0.28209479177387814;

[Differentiable]
[PreferRecompute(SideEffectBehavior.Allow)]
float3 shDC(float3 sh0)
{
    return 0.5 + SH_C0 * sh0;
}

// Rotation matrix from a (w, x, y, z) quaternion. The input is normalized
// inside the function so callers can pass an un-normalized quaternion during
// training — the normalization is part of the differentiable path.
[Differentiable]
[PreferRecompute(SideEffectBehavior.Allow)]
float3x3 matrixFromQuaternion(float4 q)
{
    q = normalize(q);
    float w = q.x, x = q.y, y = q.z, z = q.w;

    return float3x3(
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),       2.0 * (x * z + w * y),
        2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
        2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y)
    );
}

// 3D covariance Σ = R S Sᵀ Rᵀ, with S = diag(scale). `scale` here is the
// already-exponentiated per-axis scale (callers pass exp(log_scale)).
[Differentiable]
[PreferRecompute(SideEffectBehavior.Allow)]
float3x3 buildCovariance3D(float3 scale, float4 rot_quat)
{
    float3x3 R = matrixFromQuaternion(rot_quat);
    float3x3 S = float3x3(
        scale.x, 0.0,     0.0,
        0.0,     scale.y, 0.0,
        0.0,     0.0,     scale.z
    );
    float3x3 M = mul(R, S);
    return mul(M, transpose(M));
}

#endif // TRAINER_MATH_SLANG
