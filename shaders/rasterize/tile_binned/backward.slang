// Backward rasterizer — tile-cooperative reduction + single projection backward.
//
// Per gaussian per tile: all 256 threads compute the cheap eval backward
// independently, then a parallel reduction sums the per-pixel gradients.
// Thread 0 runs the expensive projection backward ONCE with the summed
// gradient and issues 16 global atomics — eliminating 255/256 of the
// projection compute and 4080/4096 of the global atomic traffic.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/eval.slang"

static const uint TILE_SIZE = 16;
static const uint TILE_AREA = TILE_SIZE * TILE_SIZE;

static const float ALPHA_MAX = 0.99;
static const float ALPHA_MIN_TRACK = 1.0 / 255.0;

static const uint CH_MEAN2D_X  = 0;
static const uint CH_MEAN2D_Y  = 1;
static const uint CH_COV2D_0   = 2;
static const uint CH_COV2D_1   = 3;
static const uint CH_COV2D_2   = 4;
static const uint CH_OPACITY   = 5;
static const uint CH_COLOR_R   = 6;
static const uint CH_COLOR_G   = 7;
static const uint CH_COLOR_B   = 8;
static const uint CH_ITER_X    = 9;
static const uint CH_ITER_Y    = 10;
static const uint NUM_CHANNELS = 11;

groupshared float gs_reduce[NUM_CHANNELS * TILE_AREA];
groupshared uint gs_max_processed;
groupshared Gaussian3D gs_gauss;
groupshared Projected2DGaussian gs_proj;
groupshared uint gs_g_idx;

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
    uint tid = gtid.y * TILE_SIZE + gtid.x;
    uint2 frame_dim_px = uint2(camera[0].image_size);

    bool valid_pixel = (pixel.x < frame_dim_px.x && pixel.y < frame_dim_px.y);

    uint2 num_tiles = (frame_dim_px + TILE_SIZE - 1) / TILE_SIZE;
    uint tile_idx = gid.y * num_tiles.x + gid.x;
    uint t_start = tile_start[tile_idx];

    uint processed = valid_pixel ? gaussian_count_per_pixel[pixel] : 0;

    if (tid == 0) gs_max_processed = 0;
    GroupMemoryBarrierWithGroupSync();
    InterlockedMax(gs_max_processed, processed);
    GroupMemoryBarrierWithGroupSync();

    uint max_processed = gs_max_processed;
    if (max_processed == 0) return;

    float2 pixel_center = float2(pixel) + 0.5;
    float4 dL_dC_full = valid_pixel ? dL_dC_texture[pixel] : float4(0.0);
    float3 dL_dC = dL_dC_full.rgb;
    float dL_dA = dL_dC_full.a;

    float T_post = valid_pixel ? output_transmittance[pixel] : 1.0;
    float T_final = T_post;
    float3 C_behind = float3(0.0, 0.0, 0.0);

    float4x4 world_to_view = camera[0].view_matrix;
    float4x4 view_to_clip  = camera[0].proj_matrix;
    float2 frame_dim = camera[0].image_size;

    for (int i = int(max_processed) - 1; i >= 0; --i)
    {
        // --- Phase 1: Cooperative load ---, we could try moving from 16x16 tiles with all in GS mem to 32x8 with 8 in GS mem and 8 in registers via wave-intr
        if (tid == 0)
        {
            uint idx = sorted_gaussian_ids[t_start + uint(i)];
            gs_g_idx = idx;
            gs_gauss = gaussians[idx];
            gs_proj = projected_2d[idx];
        }
        GroupMemoryBarrierWithGroupSync();

        uint g_idx = gs_g_idx;
        Gaussian3D g = gs_gauss;
        Projected2DGaussian proj = gs_proj;

        // --- Phase 2: Per-pixel eval backward ---
        bool contributes = valid_pixel && (uint(i) < processed);

        float2 local_dp_mean_2d = float2(0.0);
        float3 local_dp_cov2d = float3(0.0);
        float local_dp_opacity = 0.0;
        float3 local_dp_color = float3(0.0);
        float local_abs_mean2d_x = 0.0;
        float local_abs_mean2d_y = 0.0;

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

            float dL_dalpha = dot(dL_dC, dC_dalpha) + dL_dA * (T_final / denom);
            
            float dL_d_alpha_raw = dL_dalpha * alpha_grad_scale;

            var dp_mean_2d = diffPair(proj.mean);
            var dp_cov2d   = diffPair(proj.cov2d);
            var dp_opacity = diffPair(g.opacity);
            bwd_diff(evalGaussian2DFromCov)(
                pixel_center, dp_mean_2d, dp_cov2d, dp_opacity, dL_d_alpha_raw);

            local_dp_mean_2d = dp_mean_2d.d;
            local_dp_cov2d = dp_cov2d.d;
            local_dp_opacity = dp_opacity.d;
            local_dp_color = dp_color_logit.d;

            if (alpha >= ALPHA_MIN_TRACK)
            {
                local_abs_mean2d_x = abs(dp_mean_2d.d.x);
                local_abs_mean2d_y = abs(dp_mean_2d.d.y);
            }

            C_behind += c_i * alpha * T_i;
            T_post = T_i;
        }

        // --- Phase 3: Parallel reduction ---
        gs_reduce[CH_MEAN2D_X * TILE_AREA + tid] = local_dp_mean_2d.x;
        gs_reduce[CH_MEAN2D_Y * TILE_AREA + tid] = local_dp_mean_2d.y;
        gs_reduce[CH_COV2D_0  * TILE_AREA + tid] = local_dp_cov2d.x;
        gs_reduce[CH_COV2D_1  * TILE_AREA + tid] = local_dp_cov2d.y;
        gs_reduce[CH_COV2D_2  * TILE_AREA + tid] = local_dp_cov2d.z;
        gs_reduce[CH_OPACITY  * TILE_AREA + tid] = local_dp_opacity;
        gs_reduce[CH_COLOR_R  * TILE_AREA + tid] = local_dp_color.x;
        gs_reduce[CH_COLOR_G  * TILE_AREA + tid] = local_dp_color.y;
        gs_reduce[CH_COLOR_B  * TILE_AREA + tid] = local_dp_color.z;
        gs_reduce[CH_ITER_X   * TILE_AREA + tid] = local_abs_mean2d_x;
        gs_reduce[CH_ITER_Y   * TILE_AREA + tid] = local_abs_mean2d_y;
        GroupMemoryBarrierWithGroupSync();

        //These two next phases are probably causing quite a bit of divergence, but
        //it may still be a win overall compared to doing the projection backward in every thread
        //Ideally while some threads are doing the reduction and the first one the proj backward
        // could we have one of the threads that finishes the reduction early be the one issuing
        // the mem read to load the gaussian in the GS mem? Rn thread 0 has to do the single proj + atomics and issue the mem read for next splat.
        
        [ForceUnroll]
        for (uint stride = TILE_AREA >> 1; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                [ForceUnroll]
                for (uint ch = 0; ch < NUM_CHANNELS; ++ch)
                    gs_reduce[ch * TILE_AREA + tid] += gs_reduce[ch * TILE_AREA + tid + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        // --- Phase 4: Single projection backward + atomics (thread 0) ---
        if (tid == 0)
        {
            float2 sum_dp_mean_2d = float2(
                gs_reduce[CH_MEAN2D_X * TILE_AREA],
                gs_reduce[CH_MEAN2D_Y * TILE_AREA]);
            float3 sum_dp_cov2d = float3(
                gs_reduce[CH_COV2D_0 * TILE_AREA],
                gs_reduce[CH_COV2D_1 * TILE_AREA],
                gs_reduce[CH_COV2D_2 * TILE_AREA]);

            var dp_mean      = diffPair(g.mean);
            var dp_log_scale = diffPair(g.log_scale);
            var dp_rot_quat  = diffPair(g.rot_quat);
            bwd_diff(projectGaussian3DTo2D)(
                dp_mean, dp_log_scale, dp_rot_quat,
                world_to_view, view_to_clip, frame_dim,
                sum_dp_mean_2d, sum_dp_cov2d, 0.0, 0.0);

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
            InterlockedAdd(grad_opacity[g_idx], toFixed(gs_reduce[CH_OPACITY * TILE_AREA]));
            InterlockedAdd(grad_color_r[g_idx], toFixed(gs_reduce[CH_COLOR_R * TILE_AREA]));
            InterlockedAdd(grad_color_g[g_idx], toFixed(gs_reduce[CH_COLOR_G * TILE_AREA]));
            InterlockedAdd(grad_color_b[g_idx], toFixed(gs_reduce[CH_COLOR_B * TILE_AREA]));
            InterlockedAdd(grad_2d_x_iter[g_idx], toFixed(gs_reduce[CH_ITER_X * TILE_AREA]));
            InterlockedAdd(grad_2d_y_iter[g_idx], toFixed(gs_reduce[CH_ITER_Y * TILE_AREA]));
        }

        GroupMemoryBarrierWithGroupSync();
    }
}
