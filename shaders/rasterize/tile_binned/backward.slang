// Backward rasterizer with shared-memory gradient reduction.
//
// One workgroup per tile (16x16 pixels). For each gaussian in the tile's
// sorted list (walked in reverse), all 256 threads compute their per-pixel
// gradient contribution, then reduce into groupshared accumulators. A single
// thread flushes the tile-local sum to the global atomic buffers. This
// reduces global atomic contention from 256 threads per gaussian to 1.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/eval.slang"

static const uint TILE_SIZE = 16;
static const uint TILE_THREADS = TILE_SIZE * TILE_SIZE;  // 256

static const float ALPHA_MAX = 0.99;
static const float ALPHA_MIN_TRACK = 1.0 / 255.0;

// Shared memory for tile-local gradient accumulation.
// One slot per gradient component — all threads in the tile reduce into these.
groupshared int s_grad_mean_x;
groupshared int s_grad_mean_y;
groupshared int s_grad_mean_z;
groupshared int s_grad_log_scale_x;
groupshared int s_grad_log_scale_y;
groupshared int s_grad_log_scale_z;
groupshared int s_grad_rot_w;
groupshared int s_grad_rot_x;
groupshared int s_grad_rot_y;
groupshared int s_grad_rot_z;
groupshared int s_grad_opacity;
groupshared int s_grad_color_r;
groupshared int s_grad_color_g;
groupshared int s_grad_color_b;
groupshared int s_grad_2d_x;
groupshared int s_grad_2d_y;

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void backwardRasterize(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
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
    RWStructuredBuffer<int> grad_mean_x,
    RWStructuredBuffer<int> grad_mean_y,
    RWStructuredBuffer<int> grad_mean_z,
    RWStructuredBuffer<int> grad_log_scale_x,
    RWStructuredBuffer<int> grad_log_scale_y,
    RWStructuredBuffer<int> grad_log_scale_z,
    RWStructuredBuffer<int> grad_rot_w,
    RWStructuredBuffer<int> grad_rot_x,
    RWStructuredBuffer<int> grad_rot_y,
    RWStructuredBuffer<int> grad_rot_z,
    RWStructuredBuffer<int> grad_opacity,
    RWStructuredBuffer<int> grad_color_r,
    RWStructuredBuffer<int> grad_color_g,
    RWStructuredBuffer<int> grad_color_b,
    RWStructuredBuffer<int> grad_2d_x_iter,
    RWStructuredBuffer<int> grad_2d_y_iter)
{
    uint2 pixel = dtid.xy;
    uint2 frame_dim_px = uint2(camera[0].image_size);
    uint local_idx = gtid.y * TILE_SIZE + gtid.x;
    bool pixel_valid = (pixel.x < frame_dim_px.x && pixel.y < frame_dim_px.y);

    uint2 num_tiles = (frame_dim_px + TILE_SIZE - 1) / TILE_SIZE;
    uint tile_idx = gid.y * num_tiles.x + gid.x;
    uint t_start = tile_start[tile_idx];
    uint t_end = tile_end[tile_idx];

    // Find max processed count across tile (all threads walk the same range).
    uint processed = pixel_valid ? gaussian_count_per_pixel[pixel] : 0;

    float2 pixel_center = float2(pixel) + 0.5;
    float4 dL_dC_full = pixel_valid ? dL_dC_texture[pixel] : float4(0, 0, 0, 0);
    float3 dL_dC = dL_dC_full.rgb;
    float dL_dA = dL_dC_full.a;

    float T_post = pixel_valid ? output_transmittance[pixel] : 1.0;
    float3 C_behind = float3(0.0, 0.0, 0.0);

    float4x4 world_to_view = camera[0].view_matrix;
    float4x4 view_to_clip  = camera[0].proj_matrix;
    float2 frame_dim = camera[0].image_size;

    // Walk gaussians in reverse. Each iteration: all threads compute their
    // contribution, reduce via shared memory, one thread flushes to global.
    uint tile_count = (t_end > t_start) ? (t_end - t_start) : 0;

    for (int i = int(tile_count) - 1; i >= 0; --i)
    {
        uint g_idx = sorted_gaussian_ids[t_start + uint(i)];
        Gaussian3D g = gaussians[g_idx];
        Projected2DGaussian proj = projected_2d[g_idx];

        // Clear shared accumulators.
        if (local_idx == 0)
        {
            s_grad_mean_x = 0; s_grad_mean_y = 0; s_grad_mean_z = 0;
            s_grad_log_scale_x = 0; s_grad_log_scale_y = 0; s_grad_log_scale_z = 0;
            s_grad_rot_w = 0; s_grad_rot_x = 0; s_grad_rot_y = 0; s_grad_rot_z = 0;
            s_grad_opacity = 0;
            s_grad_color_r = 0; s_grad_color_g = 0; s_grad_color_b = 0;
            s_grad_2d_x = 0; s_grad_2d_y = 0;
        }
        GroupMemoryBarrierWithGroupSync();

        // Only pixels that actually processed this gaussian contribute.
        bool contributes = pixel_valid && (uint(i) < processed);

        if (contributes)
        {
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
            float2 d_mean_2d = dp_mean_2d.d;
            float3 d_cov2d   = dp_cov2d.d;
            float  d_depth   = 0.0;
            bwd_diff(projectGaussian3DTo2D)(
                dp_mean, dp_log_scale, dp_rot_quat,
                world_to_view, view_to_clip, frame_dim,
                d_mean_2d, d_cov2d, d_depth);

            // Accumulate into shared memory (tile-local reduction).
            InterlockedAdd(s_grad_mean_x, toFixed(dp_mean.d.x));
            InterlockedAdd(s_grad_mean_y, toFixed(dp_mean.d.y));
            InterlockedAdd(s_grad_mean_z, toFixed(dp_mean.d.z));
            InterlockedAdd(s_grad_log_scale_x, toFixed(dp_log_scale.d.x));
            InterlockedAdd(s_grad_log_scale_y, toFixed(dp_log_scale.d.y));
            InterlockedAdd(s_grad_log_scale_z, toFixed(dp_log_scale.d.z));
            InterlockedAdd(s_grad_rot_w, toFixed(dp_rot_quat.d.x));
            InterlockedAdd(s_grad_rot_x, toFixed(dp_rot_quat.d.y));
            InterlockedAdd(s_grad_rot_y, toFixed(dp_rot_quat.d.z));
            InterlockedAdd(s_grad_rot_z, toFixed(dp_rot_quat.d.w));
            InterlockedAdd(s_grad_opacity, toFixed(dp_opacity.d));
            InterlockedAdd(s_grad_color_r, toFixed(dp_color_logit.d.x));
            InterlockedAdd(s_grad_color_g, toFixed(dp_color_logit.d.y));
            InterlockedAdd(s_grad_color_b, toFixed(dp_color_logit.d.z));

            if (alpha >= ALPHA_MIN_TRACK)
            {
                InterlockedAdd(s_grad_2d_x, toFixed(abs(dp_mean_2d.d.x)));
                InterlockedAdd(s_grad_2d_y, toFixed(abs(dp_mean_2d.d.y)));
            }

            C_behind += c_i * alpha * T_i;
            T_post = T_i;
        }

        // Barrier: all threads done accumulating into shared.
        GroupMemoryBarrierWithGroupSync();

        // One thread flushes the tile sum to global memory.
        if (local_idx == 0)
        {
            if (s_grad_mean_x != 0) InterlockedAdd(grad_mean_x[g_idx], s_grad_mean_x);
            if (s_grad_mean_y != 0) InterlockedAdd(grad_mean_y[g_idx], s_grad_mean_y);
            if (s_grad_mean_z != 0) InterlockedAdd(grad_mean_z[g_idx], s_grad_mean_z);
            if (s_grad_log_scale_x != 0) InterlockedAdd(grad_log_scale_x[g_idx], s_grad_log_scale_x);
            if (s_grad_log_scale_y != 0) InterlockedAdd(grad_log_scale_y[g_idx], s_grad_log_scale_y);
            if (s_grad_log_scale_z != 0) InterlockedAdd(grad_log_scale_z[g_idx], s_grad_log_scale_z);
            if (s_grad_rot_w != 0) InterlockedAdd(grad_rot_w[g_idx], s_grad_rot_w);
            if (s_grad_rot_x != 0) InterlockedAdd(grad_rot_x[g_idx], s_grad_rot_x);
            if (s_grad_rot_y != 0) InterlockedAdd(grad_rot_y[g_idx], s_grad_rot_y);
            if (s_grad_rot_z != 0) InterlockedAdd(grad_rot_z[g_idx], s_grad_rot_z);
            if (s_grad_opacity != 0) InterlockedAdd(grad_opacity[g_idx], s_grad_opacity);
            if (s_grad_color_r != 0) InterlockedAdd(grad_color_r[g_idx], s_grad_color_r);
            if (s_grad_color_g != 0) InterlockedAdd(grad_color_g[g_idx], s_grad_color_g);
            if (s_grad_color_b != 0) InterlockedAdd(grad_color_b[g_idx], s_grad_color_b);
            if (s_grad_2d_x != 0) InterlockedAdd(grad_2d_x_iter[g_idx], s_grad_2d_x);
            if (s_grad_2d_y != 0) InterlockedAdd(grad_2d_y_iter[g_idx], s_grad_2d_y);
        }
    }
}
