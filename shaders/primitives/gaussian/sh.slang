// View-dependent spherical-harmonics color (optional; classic 3DGS).
//
// The band-0 DC term is Gaussian3D.sh0 (the always-present SH0 color). These
// are the higher bands, evaluated against the per-gaussian view direction and
// stored in a SEPARATE features_rest buffer, so the default SH0 path never
// touches this file. Differentiable in the rest coefficients + view direction
// so the per-gaussian sh_backward can route dL/dRGB to the coefficients and,
// via the direction, to the mean (autodiff, matching INRIA's SH backward).

#ifndef TRAINER_GAUSSIAN_SH_SLANG
#define TRAINER_GAUSSIAN_SH_SLANG

#include "../../common/math.slang"      // SH_C0, shDC
#include "../../common/half_pack.slang"  // fp16x2 packing for features_rest
#include "generated/sh_types.slang"     // SH_REST_COEFFS, SH_REST_FLOATS, SH_MAX_DEGREE

// Compile-time upper bound on packed feature uints per gaussian (2 coeffs/uint
// at the max degree); the runtime count is halfFeatUints(rest_floats) <= this.
static const int SH_FEAT_UINTS_MAX = (SH_REST_FLOATS + 1) / 2;

// Real spherical-harmonics band constants (Kerbl et al. / INRIA 3DGS).
static const float SH_C1 = 0.4886025119029199;
static const float SH_C2_0 = 1.0925484305920792;
static const float SH_C2_1 = -1.0925484305920792;
static const float SH_C2_2 = 0.31539156525252005;
static const float SH_C2_3 = -1.0925484305920792;
static const float SH_C2_4 = 0.5462742152960396;
static const float SH_C3_0 = -0.5900435899266435;
static const float SH_C3_1 = 2.890611442640554;
static const float SH_C3_2 = -0.4570457994644658;
static const float SH_C3_3 = 0.3731763325901154;
static const float SH_C3_4 = -0.4570457994644658;
static const float SH_C3_5 = 1.445305721320277;
static const float SH_C3_6 = -0.5900435899266435;

// View-dependent SH color BEFORE the >=0 clamp: 0.5 + C0*sh0 plus every band up
// to active_deg. `dir` is the normalized view direction (gaussian - camera).
// rest[k] is SH coefficient (k+1) as an RGB triple. Differentiable in rest +
// dir; sh0 / active_deg are no_diff (the DC gradient is recovered from the
// accumulated blend-color gradient, not re-derived here).
[Differentiable]
float3 evalSHColor(no_diff float3 sh0, float3 rest[SH_REST_COEFFS],
                   float3 dir, no_diff int active_deg)
{
    float3 c = SH_C0 * sh0;
    if (active_deg >= 1)
    {
        float x = dir.x, y = dir.y, z = dir.z;
        c += -SH_C1 * y * rest[0] + SH_C1 * z * rest[1] - SH_C1 * x * rest[2];
        if (active_deg >= 2)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            c += SH_C2_0 * xy * rest[3]
               + SH_C2_1 * yz * rest[4]
               + SH_C2_2 * (2.0 * zz - xx - yy) * rest[5]
               + SH_C2_3 * xz * rest[6]
               + SH_C2_4 * (xx - yy) * rest[7];
            if (active_deg >= 3)
            {
                c += SH_C3_0 * y * (3.0 * xx - yy) * rest[8]
                   + SH_C3_1 * xy * z * rest[9]
                   + SH_C3_2 * y * (4.0 * zz - xx - yy) * rest[10]
                   + SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * rest[11]
                   + SH_C3_4 * x * (4.0 * zz - xx - yy) * rest[12]
                   + SH_C3_5 * z * (xx - yy) * rest[13]
                   + SH_C3_6 * x * (xx - 3.0 * yy) * rest[14];
            }
        }
    }
    return 0.5 + c;
}

// World-space camera center from the world->view matrix. view maps world->view
// as p_view = R*p_world + t (mul(view, [p,1]), matching project.slang), so the
// center (where p_view == 0) is -R^T t.
float3 cameraCenterWorld(float4x4 view_matrix)
{
    float3x3 R = float3x3(view_matrix);
    float3 t = float3(view_matrix[0][3], view_matrix[1][3], view_matrix[2][3]);
    return -mul(transpose(R), t);
}

// SH color as a function of the gaussian MEAN (the view direction depends on
// it): dir = normalize(mean - cam). Differentiable in mean + rest so the SH
// backward routes dL/dRGB to the coefficients AND, through the direction, to
// the mean (INRIA folds this dir->mean term). Returns the pre-clamp color.
[Differentiable]
float3 shColorFromMean(no_diff float3 sh0, float3 rest[SH_REST_COEFFS],
                       float3 mean, no_diff float3 cam_center, no_diff int active_deg)
{
    float3 dir = normalize(mean - cam_center);
    return evalSHColor(sh0, rest, dir, active_deg);
}

// Load a gaussian's SH rest coefficients (RGB-interleaved per coeff) from the
// fp16-packed features_rest buffer into the fixed-size array evalSHColor wants.
// `rest_floats` is the configured storage width (3 * per-channel rest coeffs);
// coefficients beyond it (unused at the active degree) are zero-filled.
float3[SH_REST_COEFFS] loadSHRest(StructuredBuffer<uint> features_rest, uint gid,
                                 uint rest_floats)
{
    float3 rest[SH_REST_COEFFS];
    uint base = gid * halfFeatUints(rest_floats);
    uint n_coeffs = rest_floats / 3u;
    [ForceUnroll]
    for (uint k = 0u; k < uint(SH_REST_COEFFS); ++k)
    {
        if (k < n_coeffs)
        {
            uint s = k * 3u;
            rest[k] = float3(readHalfScalar(features_rest, base, s),
                             readHalfScalar(features_rest, base, s + 1u),
                             readHalfScalar(features_rest, base, s + 2u));
        }
        else
        {
            rest[k] = float3(0.0);
        }
    }
    return rest;
}

#endif // TRAINER_GAUSSIAN_SH_SLANG
