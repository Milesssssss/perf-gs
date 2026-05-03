// 3D -> 2D gaussian projection.
//
// Produces the screen-space mean, the 2x2 screen-space covariance, and the
// view-space depth. Callers are expected to cull against `depth_vs <= 0`
// (behind-camera) BEFORE relying on the screen outputs. We do NOT clamp
// `mean_clip.w` to a floor — a behind-camera point produces undefined
// screen coordinates, and clamping only hides the bug.
//
// Implements the EWA splatting Jacobian of the pinhole projection and the
// standard 3DGS "low-pass" 0.3-diagonal dilation on the 2D covariance to
// keep sub-pixel gaussians visible (Kerbl et al., 3D Gaussian Splatting,
// 2023, §4.2 and supplemental on screen-space filtering).

#ifndef TRAINER_GAUSSIAN_PROJECT_SLANG
#define TRAINER_GAUSSIAN_PROJECT_SLANG

#include "../../common/math.slang"

[Differentiable]
[PreferRecompute]
void projectGaussian3DTo2D(
    float3 mean,
    float3 log_scale,
    float4 rot_quat,
    no_diff float4x4 world_to_view,
    no_diff float4x4 view_to_clip,
    no_diff float2 frame_dim,
    out float2 mean_2d,
    out float3 cov2d,
    out float depth_vs)
{
    // --- mean: world -> view -> clip -> NDC -> pixel ---
    float3 mean_vs = mul(world_to_view, float4(mean, 1.0)).xyz;
    depth_vs = mean_vs.z;

    float4 mean_clip = mul(view_to_clip, float4(mean_vs, 1.0));
    float inv_w = 1.0 / mean_clip.w;
    float2 ndc = mean_clip.xy * inv_w;
    float2 uv = ndc * 0.5 + 0.5;
    mean_2d = uv * frame_dim;

    // --- 3D covariance in view space ---
    float3 scale = exp(log_scale);
    float3x3 Sigma3D = buildCovariance3D(scale, rot_quat);

    float3x3 W = float3x3(
        world_to_view[0].xyz,
        world_to_view[1].xyz,
        world_to_view[2].xyz
    );
    float3x3 Sigma_view = mul(mul(W, Sigma3D), transpose(W));

    // --- Jacobian of perspective divide to screen pixels ---
    // For a pinhole with clip-space matrix entries P[0][0]=fx_ndc and
    // P[1][1]=fy_ndc (both typically positive, y possibly flipped), the
    // pixel-space focal lengths are:
    float focal_x = view_to_clip[0][0] * frame_dim.x * 0.5;
    float focal_y = view_to_clip[1][1] * frame_dim.y * 0.5;

    float x = mean_vs.x;
    float y = mean_vs.y;
    float z = mean_vs.z;
    float z2 = z * z;

    // J rows = d(screen_xy) / d(view_xyz) evaluated at the mean.
    float2 J0 = float2(-focal_x / z, 0.0);             // d/dX
    float2 J1 = float2(0.0, -focal_y / z);             // d/dY
    float2 J2 = float2(focal_x * x / z2, focal_y * y / z2); // d/dZ

    // Σ_2d = J Σ_view Jᵀ  (inlined 2x2 symmetric result)
    float cov_00 = J0.x * (Sigma_view[0][0] * J0.x + Sigma_view[0][1] * J1.x + Sigma_view[0][2] * J2.x)
                 + J1.x * (Sigma_view[1][0] * J0.x + Sigma_view[1][1] * J1.x + Sigma_view[1][2] * J2.x)
                 + J2.x * (Sigma_view[2][0] * J0.x + Sigma_view[2][1] * J1.x + Sigma_view[2][2] * J2.x);

    float cov_01 = J0.x * (Sigma_view[0][0] * J0.y + Sigma_view[0][1] * J1.y + Sigma_view[0][2] * J2.y)
                 + J1.x * (Sigma_view[1][0] * J0.y + Sigma_view[1][1] * J1.y + Sigma_view[1][2] * J2.y)
                 + J2.x * (Sigma_view[2][0] * J0.y + Sigma_view[2][1] * J1.y + Sigma_view[2][2] * J2.y);

    float cov_11 = J0.y * (Sigma_view[0][0] * J0.y + Sigma_view[0][1] * J1.y + Sigma_view[0][2] * J2.y)
                 + J1.y * (Sigma_view[1][0] * J0.y + Sigma_view[1][1] * J1.y + Sigma_view[1][2] * J2.y)
                 + J2.y * (Sigma_view[2][0] * J0.y + Sigma_view[2][1] * J1.y + Sigma_view[2][2] * J2.y);

    // Low-pass filter: adds half a pixel^2 of variance on each axis so
    // sub-pixel gaussians still rasterize as at least one pixel.
    // Not defensive — derived in the 3DGS paper (see header comment).
    cov2d = float3(cov_00 + 0.3, cov_01, cov_11 + 0.3);
}

#endif // TRAINER_GAUSSIAN_PROJECT_SLANG
