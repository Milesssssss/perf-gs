// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Differentiable math utilities shared by the gaussian primitive.

#ifndef TRAINER_MATH_SLANG
#define TRAINER_MATH_SLANG

[Differentiable]
[PreferRecompute(SideEffectBehavior.Allow)]
float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

// log(1 + e^x) without overflow. Note softplus(x) == -log(1 - sigmoid(x)).
[Differentiable]
[PreferRecompute(SideEffectBehavior.Allow)]
float softplus(float x)
{
    return max(x, 0.0) + log(1.0 + exp(-abs(x)));
}

// SH band-0 (DC) color activation as in 3DGS: color = 0.5 + C0 * sh0, C0 = 0.5 * sqrt(1/pi).
// Affine, so unlike a sigmoid it does not attenuate the color gradient of bright values.
static const float SH_C0 = 0.28209479177387814;

[Differentiable]
[PreferRecompute(SideEffectBehavior.Allow)]
float3 shDC(float3 sh0)
{
    return 0.5 + SH_C0 * sh0;
}

// (w, x, y, z) quaternion, normalized here so the normalization is on the differentiable path.
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

// Σ = R S Sᵀ Rᵀ with S = diag(scale); callers pass the exponentiated scale, exp(log_scale).
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
