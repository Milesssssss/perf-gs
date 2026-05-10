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
    out float depth_vs,
    out float cov2d_det)
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
    float focal_x = view_to_clip[0][0] * frame_dim.x * 0.5;
    float focal_y = view_to_clip[1][1] * frame_dim.y * 0.5;

    float z = mean_vs.z;
    float z2 = z * z;
    // Clamp x/z and y/z to prevent Jacobian degeneracy at screen edges
    // This is the standard 3DGS fix (lim = tan(FOV/2) * 1.3).
    float lim_x = 1.3 * z / (view_to_clip[0][0]);
    float lim_y = 1.3 * z / (view_to_clip[1][1]);
    float x = clamp(mean_vs.x, -lim_x, lim_x);
    float y = clamp(mean_vs.y, -lim_y, lim_y);

    // J = d(screen_xy) / d(view_xyz), as a 2×3 matrix (2 rows, 3 cols).
    float2x3 J = float2x3(
        float3(focal_x / z, 0.0, -focal_x * x / z2), // d(sx)/d(vx,vy,vz)
        float3(0.0, focal_y / z, -focal_y * y / z2)  // d(sy)/d(vx,vy,vz)
    );

    float2x2 cov2d_full = mul(J, mul(Sigma_view, transpose(J)));

    // Pack upper triangle into float3 (xx, xy, yy).
    cov2d = float3(cov2d_full[0][0] + 0.3, cov2d_full[0][1], cov2d_full[1][1] + 0.3);
    cov2d_det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
}

#endif // TRAINER_GAUSSIAN_PROJECT_SLANG
