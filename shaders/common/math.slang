// Differentiable math utilities shared by the gaussian primitive.
// Kept lean in v0: only what the forward pass actually calls.

#ifndef TRAINER_MATH_SLANG
#define TRAINER_MATH_SLANG

[Differentiable]
[PreferRecompute]
float sigmoid(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

[Differentiable]
[PreferRecompute]
float3 sigmoid3(float3 v)
{
    return float3(sigmoid(v.x), sigmoid(v.y), sigmoid(v.z));
}

// Rotation matrix from a (w, x, y, z) quaternion. The input is normalized
// inside the function so callers can pass an un-normalized quaternion during
// training — the normalization is part of the differentiable path.
[Differentiable]
[PreferRecompute]
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
[PreferRecompute]
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
