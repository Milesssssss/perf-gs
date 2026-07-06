//[PreferRecompute]
[Differentiable]
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
    // Transform mean to view space.
    float3 mean_vs = mul(world_to_view, float4(mean, 1.0)).xyz;
    depth_vs = mean_vs.z;

    // Project mean to pixel coordinates.
    float4 mean_clip = mul(view_to_clip, float4(mean_vs, 1.0));
    float inv_w = 1.0 / mean_clip.w;
    float2 ndc = mean_clip.xy * inv_w;
    mean_2d = (ndc * 0.5 + 0.5) * frame_dim;
    
    // 3D covariance in view space.
    float3 scale = exp(log_scale);
    float3x3 Sigma3D = buildCovariance3D(scale, rot_quat);
    float3x3 W = float3x3(world_to_view);
    float3x3 Sigma_view = mul(W, mul(Sigma3D, transpose(W)));

    float fx = view_to_clip[0][0] * frame_dim.x * 0.5;
    float fy = view_to_clip[1][1] * frame_dim.y * 0.5;

    float tan_fovx = 0.5 * frame_dim.x / abs(fx);
    float tan_fovy = 0.5 * frame_dim.y / abs(fy);

    float limx = 1.3 * tan_fovx;
    float limy = 1.3 * tan_fovy;
    float txtz = clamp(mean_vs.x / mean_vs.z, -limx, limx);
    float tytz = clamp(mean_vs.y / mean_vs.z, -limy, limy);

    float3x3 J = float3x3(
        fx / mean_vs.z, 0.0, -fx * txtz / mean_vs.z,
        0.0, fy / mean_vs.z, -fy * tytz / mean_vs.z,
        0.0, 0.0, 0.0
    );

    // 2D covariance
    // mat2 cov2d_full = mat2(J * Vrk * transpose(J));
    float3x3 cov2d_full = mul(J, mul(Sigma_view, transpose(J)));

    float a = cov2d_full[0][0] + 0.3;
    float b = cov2d_full[0][1];
    float c = cov2d_full[1][1] + 0.3;

    float max_radius = min(frame_dim.x, frame_dim.y);
    float max_lambda = max_radius * max_radius * 0.5;
    float mid = 0.5 * (a + c);
    float disc = sqrt(max((mid - a) * (mid - a) + b * b, 0.0));
    float lambda1 = mid + disc;

    if (lambda1 > max_lambda)
    {
        float scale_factor = max_lambda / lambda1;
        a = (a - 0.3) * scale_factor + 0.3;
        b = b * scale_factor;
        c = (c - 0.3) * scale_factor + 0.3;
    }

    cov2d = float3(a, b, c);
    cov2d_det = a * c - b * b;
}