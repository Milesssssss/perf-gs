// Backward rasterizer — direct int64 global atomics.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/eval.slang"

static const uint TILE_SIZE = 16;

static const float ALPHA_MAX = 0.99;
static const float ALPHA_MIN_TRACK = 1.0 / 255.0;

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void backwardRasterize(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<Projected2DGaussian> projected_2d,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,
    StructuredBuffer<CameraData> camera,
    RWTexture2D<float4> output_image,
    RWTexture2D<float> output_transmittance,
    RWTexture2D<uint> gaussian_count_per_pixel,
    RWTexture2D<float4> dL_dC_texture,
    RWStructuredBuffer<int64_t> grad_mean_x,
    RWStructuredBuffer<int64_t> grad_mean_y,
    RWStructuredBuffer<int64_t> grad_mean_z,
    RWStructuredBuffer<int64_t> grad_log_scale_x,
    RWStructuredBuffer<int64_t> grad_log_scale_y,
    RWStructuredBuffer<int64_t> grad_log_scale_z,
    RWStructuredBuffer<int64_t> grad_rot_w,
    RWStructuredBuffer<int64_t> grad_rot_x,
    RWStructuredBuffer<int64_t> grad_rot_y,
    RWStructuredBuffer<int64_t> grad_rot_z,
    RWStructuredBuffer<int64_t> grad_opacity,
    RWStructuredBuffer<int64_t> grad_color_r,
    RWStructuredBuffer<int64_t> grad_color_g,
    RWStructuredBuffer<int64_t> grad_color_b,
    RWStructuredBuffer<int64_t> grad_2d_x_iter,
    RWStructuredBuffer<int64_t> grad_2d_y_iter)
{
    uint2 pixel = dtid.xy;
    uint2 frame_dim_px = uint2(camera[0].image_size);
    if (pixel.x >= frame_dim_px.x || pixel.y >= frame_dim_px.y) return;

    uint2 num_tiles = (frame_dim_px + TILE_SIZE - 1) / TILE_SIZE;
    uint tile_idx = gid.y * num_tiles.x + gid.x;
    uint t_start = tile_start[tile_idx];

    uint processed = gaussian_count_per_pixel[pixel];
    if (processed == 0) return;

    float2 pixel_center = float2(pixel) + 0.5;
    float4 dL_dC_full = dL_dC_texture[pixel];
    float3 dL_dC = dL_dC_full.rgb;
    float dL_dA = dL_dC_full.a;

    float T_post = output_transmittance[pixel];
    float3 C_behind = float3(0.0, 0.0, 0.0);

    float4x4 world_to_view = camera[0].view_matrix;
    float4x4 view_to_clip  = camera[0].proj_matrix;
    float2 frame_dim = camera[0].image_size;

    for (int i = int(processed) - 1; i >= 0; --i)
    {
        uint g_idx = sorted_gaussian_ids[t_start + uint(i)];
        Gaussian3D g = gaussians[g_idx];
        Projected2DGaussian proj = projected_2d[g_idx];

        float alpha_raw = evalGaussian2DFromCov(pixel_center, proj.mean, proj.cov2d, g.opacity);
        float alpha = min(alpha_raw, ALPHA_MAX);
        float alpha_grad_scale = (alpha_raw <= ALPHA_MAX) ? 1.0 : 0.0;

        float denom = 1.0 - alpha;
        float T_i = T_post / denom;

        var dp_color_logit = diffPair(g.color_logit);
        float3 c_i = sigmoid3(dp_color_logit.p);

        float3 dL_dc_i = dL_dC * (alpha * T_i);
        bwd_diff(sigmoid3)(dp_color_logit, dL_dc_i);

        float3 dC_dalpha = c_i * T_i - C_behind / denom;
        float dL_dalpha = dot(dL_dC, dC_dalpha) + dL_dA * T_i;
        float dL_d_alpha_raw = dL_dalpha * alpha_grad_scale;

        var dp_mean_2d = diffPair(proj.mean);
        var dp_cov2d   = diffPair(proj.cov2d);
        var dp_opacity = diffPair(g.opacity);
        bwd_diff(evalGaussian2DFromCov)(
            pixel_center, dp_mean_2d, dp_cov2d, dp_opacity, dL_d_alpha_raw);

        var dp_mean      = diffPair(g.mean);
        var dp_log_scale = diffPair(g.log_scale);
        var dp_rot_quat  = diffPair(g.rot_quat);
        bwd_diff(projectGaussian3DTo2D)(
            dp_mean, dp_log_scale, dp_rot_quat,
            world_to_view, view_to_clip, frame_dim,
            dp_mean_2d.d, dp_cov2d.d, 0.0);

        InterlockedAdd(grad_mean_x[g_idx], toFixed(dp_mean.d.x));
        InterlockedAdd(grad_mean_y[g_idx], toFixed(dp_mean.d.y));
        InterlockedAdd(grad_mean_z[g_idx], toFixed(dp_mean.d.z));
        InterlockedAdd(grad_log_scale_x[g_idx], toFixed(dp_log_scale.d.x));
        InterlockedAdd(grad_log_scale_y[g_idx], toFixed(dp_log_scale.d.y));
        InterlockedAdd(grad_log_scale_z[g_idx], toFixed(dp_log_scale.d.z));
        InterlockedAdd(grad_rot_w[g_idx], toFixed(dp_rot_quat.d.x));
        InterlockedAdd(grad_rot_x[g_idx], toFixed(dp_rot_quat.d.y));
        InterlockedAdd(grad_rot_y[g_idx], toFixed(dp_rot_quat.d.z));
        InterlockedAdd(grad_rot_z[g_idx], toFixed(dp_rot_quat.d.w));
        InterlockedAdd(grad_opacity[g_idx], toFixed(dp_opacity.d));
        InterlockedAdd(grad_color_r[g_idx], toFixed(dp_color_logit.d.x));
        InterlockedAdd(grad_color_g[g_idx], toFixed(dp_color_logit.d.y));
        InterlockedAdd(grad_color_b[g_idx], toFixed(dp_color_logit.d.z));

        if (alpha >= ALPHA_MIN_TRACK)
        {
            InterlockedAdd(grad_2d_x_iter[g_idx], toFixed(abs(dp_mean_2d.d.x)));
            InterlockedAdd(grad_2d_y_iter[g_idx], toFixed(abs(dp_mean_2d.d.y)));
        }

        C_behind += c_i * alpha * T_i;
        T_post = T_i;
    }
}
