// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_GAUSSIAN_PROJECT_SLANG
#define TRAINER_GAUSSIAN_PROJECT_SLANG

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../core/interfaces.slang"

// Mean world -> view -> clip -> ndc -> pixel; single source of truth shared with mip/compute_3d_filter.slang.
[Differentiable]
void projectMeanToPixel(
    float3 mean,
    no_diff float4x4 world_to_view,
    no_diff float4x4 view_to_clip,
    no_diff float2 frame_dim,
    out float3 mean_vs,
    out float2 mean_2d,
    out float clip_w)     // clip-space w; callers cull on clip_w <= 0
{
    mean_vs = mul(world_to_view, float4(mean, 1.0)).xyz;
    float4 mean_clip = mul(view_to_clip, float4(mean_vs, 1.0));
    clip_w = mean_clip.w;
    float inv_w = 1.0 / clip_w;
    float2 ndc = mean_clip.xy * inv_w;
    mean_2d = (ndc * 0.5 + 0.5) * frame_dim;
}

// Perspective Jacobian J = d(pixel)/d(view) at the mean with the standard 1.3*tan_fov clamp;
// shared with structure-aware densification (densify/score_structure_aware.slang).
[Differentiable]
float3x3 projectionJacobian(
    float3 mean_vs,
    no_diff float4x4 view_to_clip,
    no_diff float2 frame_dim)
{
    float fx = view_to_clip[0][0] * frame_dim.x * 0.5;
    float fy = view_to_clip[1][1] * frame_dim.y * 0.5;

    float tan_fovx = 0.5 * frame_dim.x / abs(fx);
    float tan_fovy = 0.5 * frame_dim.y / abs(fy);

    float limx = 1.3 * tan_fovx;
    float limy = 1.3 * tan_fovy;
    float txtz = clamp(mean_vs.x / mean_vs.z, -limx, limx);
    float tytz = clamp(mean_vs.y / mean_vs.z, -limy, limy);

    return float3x3(
        fx / mean_vs.z, 0.0, -fx * txtz / mean_vs.z,
        0.0, fy / mean_vs.z, -fy * tytz / mean_vs.z,
        0.0, 0.0, 0.0
    );
}

[Differentiable]
void projectGaussian3DTo2D(
    float3 mean,
    float3 log_scale,
    float4 rot_quat,
    no_diff float4x4 world_to_view,
    no_diff float4x4 view_to_clip,
    no_diff float2 frame_dim,
    no_diff float filter_3D,     // Mip-Splatting 3D smoothing size (0 = off)
    no_diff float kernel_size,   // 2D low-pass dilation (0.3 vanilla / 0.1 mip)
    no_diff uint mip_enabled,    // gate the (differentiable) compensation
    out float2 mean_2d,
    out float3 cov2d,
    out float depth_vs,
    out float cov2d_det,
    out float opacity_compensation)   // Mip coef3d*coef2d (1.0 when off)
{
    float3 mean_vs;
    float clip_w;
    projectMeanToPixel(mean, world_to_view, view_to_clip, frame_dim, mean_vs, mean_2d, clip_w);
    depth_vs = mean_vs.z;

    // Mip-Splatting 3D smoothing filter: isotropic add of the world-space low-pass size in squared-scale space.
    float3 scale0 = exp(log_scale);
    float3 scale = sqrt(scale0 * scale0 + filter_3D * filter_3D);
    float3x3 Sigma3D = buildCovariance3D(scale, rot_quat);
    float3x3 W = float3x3(world_to_view);
    float3x3 Sigma_view = mul(W, mul(Sigma3D, transpose(W)));

    float3x3 J = projectionJacobian(mean_vs, view_to_clip, frame_dim);

    float3x3 cov2d_full = mul(J, mul(Sigma_view, transpose(J)));

    float a = cov2d_full[0][0] + kernel_size;
    float b = cov2d_full[0][1];
    float c = cov2d_full[1][1] + kernel_size;

    float max_radius = min(frame_dim.x, frame_dim.y);
    float max_lambda = max_radius * max_radius * 0.5;
    float mid = 0.5 * (a + c);
    float disc = sqrt(max((mid - a) * (mid - a) + b * b, 0.0));
    float lambda1 = mid + disc;

    if (lambda1 > max_lambda)
    {
        float scale_factor = max_lambda / lambda1;
        a = (a - kernel_size) * scale_factor + kernel_size;
        b = b * scale_factor;
        c = (c - kernel_size) * scale_factor + kernel_size;
    }

    cov2d = float3(a, b, c);
    cov2d_det = a * c - b * b;

    // Differentiable Mip-Splatting compensation; the gate keeps vanilla exactly 1.0 with no spurious gradient.
    if (mip_enabled != 0u)
    {
        float3 s2 = scale0 * scale0;                 // pre-filter squared scale
        float3 s2f = s2 + filter_3D * filter_3D;
        float coef3d = sqrt((s2.x * s2.y * s2.z) / (s2f.x * s2f.y * s2f.z));
        // 2D compensation as in SADGS (the mip-splatting original differs slightly).
        float det1 = max(1e-6, cov2d_det);           // post-dilation det
        float det0 = max(1e-6, (a - kernel_size) * (c - kernel_size) - b * b);
        float coef2d = sqrt(det0 / (det1 + 1e-6));
        opacity_compensation = coef3d * coef2d;
    }
    else
    {
        opacity_compensation = 1.0;
    }
}

// The Gaussian's geometry as an IProjectableModel (opacity/sh0 don't affect projection).
struct GaussianModel : IProjectableModel
{
    float3 mean;
    float3 log_scale;
    float4 rot_quat;

    [Differentiable]
    void projectInto(no_diff CameraData cam,
                     no_diff float filter_3D, no_diff float kernel_size, no_diff uint mip_enabled,
                     out float2 mean2d, out float3 cov2d, out float depth_vs,
                     out float cov2d_det, out float opacity_compensation)
    {
        projectGaussian3DTo2D(mean, log_scale, rot_quat,
            cam.view_matrix, cam.proj_matrix, cam.image_size,
            filter_3D, kernel_size, mip_enabled,
            mean2d, cov2d, depth_vs, cov2d_det, opacity_compensation);
    }
}

#endif // TRAINER_GAUSSIAN_PROJECT_SLANG