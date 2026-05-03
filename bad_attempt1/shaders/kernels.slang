// 3DGS Training Pipeline - All Compute Kernels
// This file contains all entry points for the training pipeline with explicit parameters

#include "types.slang"
#include "utils.slang"

// ============================================================================
// CLEAR / INITIALIZATION KERNELS
// ============================================================================

[shader("compute")]
[numthreads(256, 1, 1)]
void clearTileBins(
    uint3 DTid : SV_DispatchThreadID,
    uniform uint total_tiles,
    RWStructuredBuffer<uint> screen_tiles_count,
    RWStructuredBuffer<uint> screen_tiles_indices
)
{
    uint tileIdx = DTid.x;
    if (tileIdx >= total_tiles)
        return;

    screen_tiles_count[tileIdx] = 0;
    
    uint baseIdx = tileIdx * MAX_GAUSSIANS_PER_TILE;
    for (uint i = 0; i < MAX_GAUSSIANS_PER_TILE; i++)
    {
        screen_tiles_indices[baseIdx + i] = 0xFFFFFFFF;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clearProjectedGaussians(
    uint3 DTid : SV_DispatchThreadID,
    uniform uint max_splats,
    RWStructuredBuffer<Struct_Projected2DGaussian> projected_gaussians_2d
)
{
    uint idx = DTid.x;
    if (idx >= max_splats)
        return;

    Struct_Projected2DGaussian proj;
    proj.mean = float2(0, 0);
    proj.cov2d = float3(0, 0, 0);
    proj.depth_vs = 0;
    projected_gaussians_2d[idx] = proj;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clearGradientBuffers(
    uint3 DTid : SV_DispatchThreadID,
    uniform uint max_splats,
    RWStructuredBuffer<int> mean_gradient_accum_x,
    RWStructuredBuffer<int> mean_gradient_accum_y,
    RWStructuredBuffer<int> mean_gradient_accum_z,
    RWStructuredBuffer<int> mean2d_gradient_accum_magnitude,
    RWStructuredBuffer<int> scale_gradient_accum_x,
    RWStructuredBuffer<int> scale_gradient_accum_y,
    RWStructuredBuffer<int> scale_gradient_accum_z,
    RWStructuredBuffer<int> rotation_gradient_accum_x,
    RWStructuredBuffer<int> rotation_gradient_accum_y,
    RWStructuredBuffer<int> rotation_gradient_accum_z,
    RWStructuredBuffer<int> rotation_gradient_accum_w,
    RWStructuredBuffer<int> color_gradient_accum_r,
    RWStructuredBuffer<int> color_gradient_accum_g,
    RWStructuredBuffer<int> color_gradient_accum_b,
    RWStructuredBuffer<int> opacity_gradient_accum
)
{
    uint idx = DTid.x;
    if (idx >= max_splats)
        return;

    mean_gradient_accum_x[idx] = 0;
    mean_gradient_accum_y[idx] = 0;
    mean_gradient_accum_z[idx] = 0;
    mean2d_gradient_accum_magnitude[idx] = 0;
    scale_gradient_accum_x[idx] = 0;
    scale_gradient_accum_y[idx] = 0;
    scale_gradient_accum_z[idx] = 0;
    rotation_gradient_accum_x[idx] = 0;
    rotation_gradient_accum_y[idx] = 0;
    rotation_gradient_accum_z[idx] = 0;
    rotation_gradient_accum_w[idx] = 0;
    color_gradient_accum_r[idx] = 0;
    color_gradient_accum_g[idx] = 0;
    color_gradient_accum_b[idx] = 0;
    opacity_gradient_accum[idx] = 0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clearAdamState(
    uint3 DTid : SV_DispatchThreadID,
    uniform uint max_splats,
    RWStructuredBuffer<Struct_AdamState> adam_state
)
{
    uint idx = DTid.x;
    if (idx >= max_splats)
        return;

    adam_state[idx] = zero_adam_state();
}

[shader("compute")]
[numthreads(256, 1, 1)]
void clearPixelCoverage(
    uint3 DTid : SV_DispatchThreadID,
    uniform uint max_splats,
    RWStructuredBuffer<uint> pixel_coverage_count_per_gaussian
)
{
    uint idx = DTid.x;
    if (idx >= max_splats)
        return;

    pixel_coverage_count_per_gaussian[idx] = 0;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void clearTextures(
    uint3 DTid : SV_DispatchThreadID,
    uniform float2 frame_dim,
    RWTexture2D<float4> output_image,
    RWTexture2D<float> output_transmittance,
    RWTexture2D<uint> gaussian_count_per_pixel
)
{
    uint2 pixel = DTid.xy;
    if (pixel.x >= (uint)frame_dim.x || pixel.y >= (uint)frame_dim.y)
        return;

    output_image[pixel] = float4(0, 0, 0, 0);
    output_transmittance[pixel] = 1.0;
    gaussian_count_per_pixel[pixel] = 0;
}

// ============================================================================
// BINNING KERNEL
// ============================================================================

void compute2DBoundingBox(float2 mean_2d, float3 cov2d, out float2 bbox_min, out float2 bbox_max)
{
    float a = cov2d.x;
    float b = cov2d.y;
    float c = cov2d.z;

    float trace = a + c;
    float det = a * c - b * b;

    float discriminant = max(0.0, trace * trace * 0.25 - det);
    float lambda1 = trace * 0.5 + sqrt(discriminant);
    float lambda2 = trace * 0.5 - sqrt(discriminant);

    float radius_x = 3.1 * sqrt(max(lambda1, 0.0));
    float radius_y = 3.1 * sqrt(max(lambda2, 0.0));
    float radius = max(radius_x, radius_y);

    bbox_min = mean_2d - float2(radius, radius);
    bbox_max = mean_2d + float2(radius, radius);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void binGaussians(
    uint3 DTid : SV_DispatchThreadID,
    uniform float2 frame_dim,
    uniform uint num_gaussians,
    StructuredBuffer<Struct_Gaussian3D> input_gaussians,
    StructuredBuffer<CameraData> cameras,
    uniform uint camera_index,
    RWStructuredBuffer<Struct_Projected2DGaussian> projected_gaussians_2d,
    RWStructuredBuffer<uint> screen_tiles_count,
    RWStructuredBuffer<uint> screen_tiles_indices,
    RWStructuredBuffer<uint> training_step_counter
)
{
    if (DTid.x == 0)
        InterlockedAdd(training_step_counter[0], 1);

    uint gid = DTid.x;
    if (gid >= num_gaussians)
        return;

    Struct_Gaussian3D gaussian = input_gaussians[gid];
    CameraData camera = cameras[camera_index];

    const uint2 numTiles = ((uint2)frame_dim + TILE_SIZE - 1) / TILE_SIZE;

    float4x4 world_to_view = camera.view_matrix;
    float4x4 view_to_screen = camera.projection_matrix;

    Struct_Projected2DGaussian proj_2d;

    projectGaussian3DTo2D_diff(
        gaussian.mean,
        gaussian.log_scale,
        gaussian.rotation_quat,
        world_to_view,
        view_to_screen,
        frame_dim,
        proj_2d.mean,
        proj_2d.cov2d,
        proj_2d.depth_vs
    );
    proj_2d.depth_vs = -proj_2d.depth_vs;
    projected_gaussians_2d[gid] = proj_2d;

    float2 screen_min, screen_max;
    compute2DBoundingBox(proj_2d.mean, proj_2d.cov2d, screen_min, screen_max);

    screen_min = max(screen_min, float2(0, 0));
    screen_max = min(screen_max, frame_dim);

    if (screen_min.x >= screen_max.x || screen_min.y >= screen_max.y)
        return;

    uint2 tile_min = uint2(screen_min) / TILE_SIZE;
    uint2 tile_max = uint2(screen_max) / TILE_SIZE;
    tile_max = min(tile_max, numTiles - 1);

    for (uint ty = tile_min.y; ty <= tile_max.y; ty++)
    {
        for (uint tx = tile_min.x; tx <= tile_max.x; tx++)
        {
            uint tileIdx = ty * numTiles.x + tx;

            uint slot;
            InterlockedAdd(screen_tiles_count[tileIdx], 1, slot);

            if (slot < MAX_GAUSSIANS_PER_TILE)
            {
                screen_tiles_indices[tileIdx * MAX_GAUSSIANS_PER_TILE + slot] = gid;
            }
        }
    }
}

// ============================================================================
// SORTING KERNEL (Bitonic Sort per Tile)
// ============================================================================

groupshared uint shared_ids[MAX_GAUSSIANS_PER_TILE];
groupshared float shared_depths[MAX_GAUSSIANS_PER_TILE];

[shader("compute")]
[numthreads(1024, 1, 1)]
void sortGaussiansPerTile(
    uint3 GTid : SV_GroupThreadID,
    uint3 Gid : SV_GroupID,
    uniform float2 frame_dim,
    RWStructuredBuffer<uint> screen_tiles_count,
    RWStructuredBuffer<uint> screen_tiles_indices,
    StructuredBuffer<Struct_Projected2DGaussian> projected_gaussians_2d
)
{
    uint2 numTiles = ((uint2)frame_dim + TILE_SIZE - 1) / TILE_SIZE;

    uint tileIdx = Gid.x;
    uint tid = GTid.x;

    bool validTile = (tileIdx < numTiles.x * numTiles.y);

    uint gaussian_count = 0;
    uint start_idx = 0;

    if (validTile)
    {
        gaussian_count = min(screen_tiles_count[tileIdx], MAX_GAUSSIANS_PER_TILE);
        start_idx = tileIdx * MAX_GAUSSIANS_PER_TILE;
    }

    // Load into shared memory
    if (validTile && tid < gaussian_count)
    {
        uint gid = screen_tiles_indices[start_idx + tid];
        shared_ids[tid] = gid;
        shared_depths[tid] = projected_gaussians_2d[gid].depth_vs;
    }
    else
    {
        shared_ids[tid] = 0xFFFFFFFF;
        shared_depths[tid] = 3.402823466e+38;
    }

    GroupMemoryBarrierWithGroupSync();

    // Bitonic sort front-to-back: smaller depth first
    for (uint k = 2; k <= MAX_GAUSSIANS_PER_TILE; k <<= 1)
    {
        for (uint j = k >> 1; j > 0; j >>= 1)
        {
            uint ixj = tid ^ j;

            if (ixj > tid)
            {
                float depth_a = shared_depths[tid];
                float depth_b = shared_depths[ixj];

                bool ascending = ((tid & k) == 0);
                bool shouldSwap = ascending ? (depth_a > depth_b) : (depth_a < depth_b);

                if (shouldSwap)
                {
                    uint temp_id = shared_ids[tid];
                    float temp_depth = shared_depths[tid];
                    
                    shared_ids[tid] = shared_ids[ixj];
                    shared_depths[tid] = shared_depths[ixj];
                    
                    shared_ids[ixj] = temp_id;
                    shared_depths[ixj] = temp_depth;
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }

    // Write back only valid entries
    if (validTile && tid < gaussian_count)
    {
        screen_tiles_indices[start_idx + tid] = shared_ids[tid];
    }
}

// ============================================================================
// FORWARD RASTERIZATION KERNEL
// ============================================================================

groupshared uint pixel_count_per_splat_gs;

[shader("compute")]
[numthreads(16, 16, 1)]
void forwardRasterize(
    uint3 DTid : SV_DispatchThreadID,
    uint3 Gid : SV_GroupID,
    uint Gidx : SV_GroupIndex,
    uniform float2 frame_dim,
    StructuredBuffer<Struct_Gaussian3D> input_gaussians,
    StructuredBuffer<Struct_Projected2DGaussian> projected_gaussians_2d,
    StructuredBuffer<uint> screen_tiles_count,
    StructuredBuffer<uint> screen_tiles_indices,
    RWTexture2D<float4> output_image,
    RWTexture2D<float> output_transmittance,
    RWTexture2D<uint> gaussian_count_per_pixel,
    RWStructuredBuffer<uint> pixel_coverage_count_per_gaussian
)
{
    uint2 pixel = DTid.xy;

    if (pixel.x >= (uint)frame_dim.x || pixel.y >= (uint)frame_dim.y)
        return;

    uint2 tile_coord = DTid.xy / TILE_SIZE;
    uint2 num_tiles = ((uint2)frame_dim + TILE_SIZE - 1) / TILE_SIZE;

    uint tileIdx = tile_coord.y * num_tiles.x + tile_coord.x;

    uint tile_gaussian_count = min(screen_tiles_count[tileIdx], MAX_GAUSSIANS_PER_TILE);
    uint start_idx = tileIdx * MAX_GAUSSIANS_PER_TILE;

    float3 rendered_color = float3(0, 0, 0);
    float transmittance = 1.0;
    uint pixel_accum_count = 0;

    // Front-to-back blending
    for (uint i = 0; i < tile_gaussian_count; i++)
    {
        if (Gidx == 0)
            pixel_count_per_splat_gs = 0;

        GroupMemoryBarrierWithGroupSync();

        uint gaussian_idx = screen_tiles_indices[start_idx + i];

        Struct_Gaussian3D g3d = input_gaussians[gaussian_idx];
        Struct_Projected2DGaussian proj = projected_gaussians_2d[gaussian_idx];

        float gauss_i_alpha = evalGaussian2DFromCov(
            pixel,
            proj.mean,
            proj.cov2d,
            g3d.opacity
        );

        pixel_accum_count++;
        InterlockedAdd(pixel_count_per_splat_gs, 1);

        GroupMemoryBarrierWithGroupSync();
        
        if (Gidx == 0)
            InterlockedAdd(pixel_coverage_count_per_gaussian[gaussian_idx], pixel_count_per_splat_gs);
        
        AllMemoryBarrierWithGroupSync();

        float3 color_rgb = sigmoid3(g3d.color_logit);
        rendered_color += color_rgb * gauss_i_alpha * transmittance;

        transmittance *= (1.0 - gauss_i_alpha);

        if (transmittance < 0.001f)
            break;
    }

    float4 output_color = float4(pow(rendered_color, 1.0 / 2.2), 1.0f);

    gaussian_count_per_pixel[pixel] = pixel_accum_count;
    output_transmittance[pixel] = transmittance;
    output_image[pixel] = output_color;
}

// ============================================================================
// BACKWARD RASTERIZATION KERNEL
// ============================================================================

groupshared GradientAccumulator gs_gradientAccumulator;

void accumulate_gradients_groupshared(
    float3 mean_d, 
    float2 mean2d_d, 
    float3 scale_d, 
    float3 col_d, 
    float opacity_d, 
    float4 rotation_d,
    bool descend_mean,
    bool descend_scale,
    bool descend_color,
    bool descend_opacity,
    bool descend_rotation
)
{
    if (descend_mean)
    {
        InterlockedAdd(gs_gradientAccumulator.grad_mean_x, toFixed(mean_d.x));
        InterlockedAdd(gs_gradientAccumulator.grad_mean_y, toFixed(mean_d.y));
        InterlockedAdd(gs_gradientAccumulator.grad_mean_z, toFixed(mean_d.z));
        InterlockedAdd(gs_gradientAccumulator.grad_mean2d_magnitude, toFixed(length(mean2d_d)));
    }

    if (descend_scale)
    {
        InterlockedAdd(gs_gradientAccumulator.grad_log_scale_x, toFixed(scale_d.x));
        InterlockedAdd(gs_gradientAccumulator.grad_log_scale_y, toFixed(scale_d.y));
        InterlockedAdd(gs_gradientAccumulator.grad_log_scale_z, toFixed(scale_d.z));
    }

    if (descend_color)
    {
        InterlockedAdd(gs_gradientAccumulator.grad_color_r, toFixed(col_d.r));
        InterlockedAdd(gs_gradientAccumulator.grad_color_g, toFixed(col_d.g));
        InterlockedAdd(gs_gradientAccumulator.grad_color_b, toFixed(col_d.b));
    }

    if (descend_opacity)
    {
        InterlockedAdd(gs_gradientAccumulator.grad_opacity, toFixed(opacity_d));
    }

    if (descend_rotation)
    {
        InterlockedAdd(gs_gradientAccumulator.grad_quat_x, toFixed(rotation_d.x));
        InterlockedAdd(gs_gradientAccumulator.grad_quat_y, toFixed(rotation_d.y));
        InterlockedAdd(gs_gradientAccumulator.grad_quat_z, toFixed(rotation_d.z));
        InterlockedAdd(gs_gradientAccumulator.grad_quat_w, toFixed(rotation_d.w));
    }
}

void accumulate_gradients_global(
    uint gaussian_idx,
    bool descend_mean,
    bool descend_scale,
    bool descend_color,
    bool descend_opacity,
    bool descend_rotation,
    RWStructuredBuffer<int> mean_gradient_accum_x,
    RWStructuredBuffer<int> mean_gradient_accum_y,
    RWStructuredBuffer<int> mean_gradient_accum_z,
    RWStructuredBuffer<int> mean2d_gradient_accum_magnitude,
    RWStructuredBuffer<int> scale_gradient_accum_x,
    RWStructuredBuffer<int> scale_gradient_accum_y,
    RWStructuredBuffer<int> scale_gradient_accum_z,
    RWStructuredBuffer<int> rotation_gradient_accum_x,
    RWStructuredBuffer<int> rotation_gradient_accum_y,
    RWStructuredBuffer<int> rotation_gradient_accum_z,
    RWStructuredBuffer<int> rotation_gradient_accum_w,
    RWStructuredBuffer<int> color_gradient_accum_r,
    RWStructuredBuffer<int> color_gradient_accum_g,
    RWStructuredBuffer<int> color_gradient_accum_b,
    RWStructuredBuffer<int> opacity_gradient_accum
)
{
    if (descend_mean)
    {
        InterlockedAdd(mean_gradient_accum_x[gaussian_idx], gs_gradientAccumulator.grad_mean_x);
        InterlockedAdd(mean_gradient_accum_y[gaussian_idx], gs_gradientAccumulator.grad_mean_y);
        InterlockedAdd(mean_gradient_accum_z[gaussian_idx], gs_gradientAccumulator.grad_mean_z);
        InterlockedAdd(mean2d_gradient_accum_magnitude[gaussian_idx], gs_gradientAccumulator.grad_mean2d_magnitude);
    }

    if (descend_scale)
    {
        InterlockedAdd(scale_gradient_accum_x[gaussian_idx], gs_gradientAccumulator.grad_log_scale_x);
        InterlockedAdd(scale_gradient_accum_y[gaussian_idx], gs_gradientAccumulator.grad_log_scale_y);
        InterlockedAdd(scale_gradient_accum_z[gaussian_idx], gs_gradientAccumulator.grad_log_scale_z);
    }

    if (descend_color)
    {
        InterlockedAdd(color_gradient_accum_r[gaussian_idx], gs_gradientAccumulator.grad_color_r);
        InterlockedAdd(color_gradient_accum_g[gaussian_idx], gs_gradientAccumulator.grad_color_g);
        InterlockedAdd(color_gradient_accum_b[gaussian_idx], gs_gradientAccumulator.grad_color_b);
    }

    if (descend_opacity)
    {
        InterlockedAdd(opacity_gradient_accum[gaussian_idx], gs_gradientAccumulator.grad_opacity);
    }

    if (descend_rotation)
    {
        InterlockedAdd(rotation_gradient_accum_x[gaussian_idx], gs_gradientAccumulator.grad_quat_x);
        InterlockedAdd(rotation_gradient_accum_y[gaussian_idx], gs_gradientAccumulator.grad_quat_y);
        InterlockedAdd(rotation_gradient_accum_z[gaussian_idx], gs_gradientAccumulator.grad_quat_z);
        InterlockedAdd(rotation_gradient_accum_w[gaussian_idx], gs_gradientAccumulator.grad_quat_w);
    }
}

[shader("compute")]
[numthreads(16, 16, 1)]
void backwardRasterize(
    uint3 DTid : SV_DispatchThreadID,
    uint3 Gid : SV_GroupID,
    uint Gid_flat : SV_GroupIndex,
    uniform float2 frame_dim,
    uniform bool descend_mean,
    uniform bool descend_scale,
    uniform bool descend_rotation,
    uniform bool descend_color,
    uniform bool descend_opacity,
    StructuredBuffer<Struct_Gaussian3D> input_gaussians,
    StructuredBuffer<Struct_Projected2DGaussian> projected_gaussians_2d,
    StructuredBuffer<CameraData> cameras,
    uniform uint camera_index,
    Texture2DArray<float4> target_images,
    SamplerState target_sampler,
    Texture2D<float4> output_image,
    Texture2D<float> output_transmittance,
    StructuredBuffer<uint> screen_tiles_count,
    StructuredBuffer<uint> screen_tiles_indices,
    Texture2D<uint> gaussian_count_per_pixel,
    // Gradient accumulator buffers
    RWStructuredBuffer<int> mean_gradient_accum_x,
    RWStructuredBuffer<int> mean_gradient_accum_y,
    RWStructuredBuffer<int> mean_gradient_accum_z,
    RWStructuredBuffer<int> mean2d_gradient_accum_magnitude,
    RWStructuredBuffer<int> scale_gradient_accum_x,
    RWStructuredBuffer<int> scale_gradient_accum_y,
    RWStructuredBuffer<int> scale_gradient_accum_z,
    RWStructuredBuffer<int> rotation_gradient_accum_x,
    RWStructuredBuffer<int> rotation_gradient_accum_y,
    RWStructuredBuffer<int> rotation_gradient_accum_z,
    RWStructuredBuffer<int> rotation_gradient_accum_w,
    RWStructuredBuffer<int> color_gradient_accum_r,
    RWStructuredBuffer<int> color_gradient_accum_g,
    RWStructuredBuffer<int> color_gradient_accum_b,
    RWStructuredBuffer<int> opacity_gradient_accum
)
{
    uint2 pixel = DTid.xy;
    
    if (pixel.x >= (uint)frame_dim.x || pixel.y >= (uint)frame_dim.y)
        return;

    uint2 numTiles = ((uint2)frame_dim + TILE_SIZE - 1) / TILE_SIZE;
    uint tileIdx = Gid.y * numTiles.x + Gid.x;
    uint tile_gaussian_count = min(screen_tiles_count[tileIdx], MAX_GAUSSIANS_PER_TILE);
    uint start_idx = tileIdx * MAX_GAUSSIANS_PER_TILE;

    CameraData camera = cameras[camera_index];
    float4x4 world_to_view = camera.view_matrix;
    float4x4 view_to_screen = camera.projection_matrix;

    // Sample target image from texture array
    float2 uv = (float2(pixel) + 0.5) / frame_dim;
    float4 target_rgba = target_images.SampleLevel(target_sampler, float3(uv, camera.image_index), 0);
    
    float target_alpha = target_rgba.a;
    float T = output_transmittance[pixel];
    float rendered_opacity = 1.0f - T;

    // L2 loss derivative in sRGB space with gamma chain rule
    float3 dL_dC_srgb = 2.0 * (output_image[pixel].rgb - target_rgba.rgb);
    float3 C_linear = pow(output_image[pixel].rgb, 2.2);
    float3 C_linear_clamped = max(C_linear, 1e-4);
    float3 dC_srgb_dC_linear = (1.0 / 2.2) * pow(C_linear_clamped, (1.0 / 2.2) - 1.0);
    float3 dL_dC = dL_dC_srgb * dC_srgb_dC_linear;

    float dL_dA_pixel = 2.0 * (rendered_opacity - target_alpha);

    float3 color_behind_accum = float3(0, 0, 0);
    uint gaussian_count = gaussian_count_per_pixel[pixel];
    float T_i_plus1 = T;

    // Back-to-front gradient computation
    for (int i = tile_gaussian_count - 1; i >= 0; i--)
    {
        if (Gid_flat == 0)
        {
            gs_gradientAccumulator.grad_mean_x = 0;
            gs_gradientAccumulator.grad_mean_y = 0;
            gs_gradientAccumulator.grad_mean_z = 0;
            gs_gradientAccumulator.grad_mean2d_magnitude = 0;
            gs_gradientAccumulator.grad_log_scale_x = 0;
            gs_gradientAccumulator.grad_log_scale_y = 0;
            gs_gradientAccumulator.grad_log_scale_z = 0;
            gs_gradientAccumulator.grad_quat_x = 0;
            gs_gradientAccumulator.grad_quat_y = 0;
            gs_gradientAccumulator.grad_quat_z = 0;
            gs_gradientAccumulator.grad_quat_w = 0;
            gs_gradientAccumulator.grad_color_r = 0;
            gs_gradientAccumulator.grad_color_g = 0;
            gs_gradientAccumulator.grad_color_b = 0;
            gs_gradientAccumulator.grad_opacity = 0;
        }

        GroupMemoryBarrierWithGroupSync();

        uint gaussian_idx = screen_tiles_indices[start_idx + i];
        bool pixel_processed_this = (i < gaussian_count);

        if (pixel_processed_this)
        {
            Struct_Gaussian3D gaussian = input_gaussians[gaussian_idx];
            Struct_Projected2DGaussian proj = projected_gaussians_2d[gaussian_idx];

            float alpha_i = evalGaussian2DFromCov(pixel, proj.mean, proj.cov2d, gaussian.opacity);

            float denom = max(1.0 - alpha_i, 1e-6);
            float T_i = T_i_plus1 / denom;

            var dp_color_logit = diffPair(gaussian.color_logit);
            float3 color_rgb = sigmoid3(dp_color_logit.p);
            float3 color_i = color_rgb;

            float dC_dColor_i = alpha_i * T_i;
            float3 dL_dColor_i = dL_dC * dC_dColor_i;

            bwd_diff(sigmoid3)(dp_color_logit, dL_dColor_i);

            float3 dC_dAlpha_i = (color_i * T_i - color_behind_accum / denom);
            float3 dL_dAlpha_vec = dL_dC * dC_dAlpha_i;
            float dL_dAlpha_color = dot(dL_dAlpha_vec, float3(1, 1, 1));
            float dL_dAlpha_opacity = dL_dA_pixel * T_i;
            float opacity_loss_weight = 0.1f;
            float dL_dAlpha = dL_dAlpha_color + dL_dAlpha_opacity * opacity_loss_weight;

            var dp_mean = diffPair(gaussian.mean);
            var dp_log_scale = diffPair(gaussian.log_scale);
            var dp_rotation = diffPair(gaussian.rotation_quat);
            var dp_opacity = diffPair(gaussian.opacity);

            bwd_diff(evalGaussian3DAtPixel)(
                pixel,
                dp_mean, dp_log_scale, dp_rotation, dp_opacity,
                world_to_view, view_to_screen, frame_dim,
                dL_dAlpha
            );

            var dp_mean_2d = diffPair(proj.mean);
            var dp_cov2d = diffPair(proj.cov2d);
            var dp_opacity_dummy = diffPair(gaussian.opacity);

            bwd_diff(evalGaussian2DFromCov)(
                pixel,
                dp_mean_2d,
                dp_cov2d,
                dp_opacity_dummy,
                dL_dAlpha
            );

            color_behind_accum += color_i * alpha_i * T_i;
            T_i_plus1 = T_i;

            accumulate_gradients_groupshared(
                dp_mean.d,
                dp_mean_2d.d,
                dp_log_scale.d,
                dp_color_logit.d,
                dp_opacity.d,
                dp_rotation.d,
                descend_mean,
                descend_scale,
                descend_color,
                descend_opacity,
                descend_rotation
            );
        }

        GroupMemoryBarrierWithGroupSync();

        if (Gid_flat == 0)
        {
            accumulate_gradients_global(
                gaussian_idx,
                descend_mean,
                descend_scale,
                descend_color,
                descend_opacity,
                descend_rotation,
                mean_gradient_accum_x,
                mean_gradient_accum_y,
                mean_gradient_accum_z,
                mean2d_gradient_accum_magnitude,
                scale_gradient_accum_x,
                scale_gradient_accum_y,
                scale_gradient_accum_z,
                rotation_gradient_accum_x,
                rotation_gradient_accum_y,
                rotation_gradient_accum_z,
                rotation_gradient_accum_w,
                color_gradient_accum_r,
                color_gradient_accum_g,
                color_gradient_accum_b,
                opacity_gradient_accum
            );
        }

        AllMemoryBarrierWithGroupSync();
    }
}

// ============================================================================
// ADAM OPTIMIZER KERNEL
// ============================================================================

float adam_step_1d(
    float param,
    float grad,
    inout float m,
    inout float v,
    float lr,
    float bias_correction1,
    float bias_correction2
)
{
    m = ADAM_BETA1 * m + (1.0f - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1.0f - ADAM_BETA2) * grad * grad;

    float m_hat = m / bias_correction1;
    float v_hat = v / bias_correction2;

    return param - lr * m_hat / (sqrt(v_hat) + ADAM_EPSILON);
}

float3 adam_step_3d(
    float3 param,
    float3 grad,
    inout float3 m,
    inout float3 v,
    float lr,
    float bias_correction1,
    float bias_correction2
)
{
    m = ADAM_BETA1 * m + (1.0f - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1.0f - ADAM_BETA2) * grad * grad;

    float3 m_hat = m / bias_correction1;
    float3 v_hat = v / bias_correction2;

    return param - lr * m_hat / (sqrt(v_hat) + ADAM_EPSILON);
}

float4 adam_step_4d(
    float4 param,
    float4 grad,
    inout float4 m,
    inout float4 v,
    float lr,
    float bias_correction1,
    float bias_correction2
)
{
    m = ADAM_BETA1 * m + (1.0f - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1.0f - ADAM_BETA2) * grad * grad;

    float4 m_hat = m / bias_correction1;
    float4 v_hat = v / bias_correction2;

    return param - lr * m_hat / (sqrt(v_hat) + ADAM_EPSILON);
}

GaussianGradients load_gradients(
    uint g_idx,
    float inv_num_views,
    RWStructuredBuffer<int> mean_gradient_accum_x,
    RWStructuredBuffer<int> mean_gradient_accum_y,
    RWStructuredBuffer<int> mean_gradient_accum_z,
    RWStructuredBuffer<int> mean2d_gradient_accum_magnitude,
    RWStructuredBuffer<int> scale_gradient_accum_x,
    RWStructuredBuffer<int> scale_gradient_accum_y,
    RWStructuredBuffer<int> scale_gradient_accum_z,
    RWStructuredBuffer<int> rotation_gradient_accum_x,
    RWStructuredBuffer<int> rotation_gradient_accum_y,
    RWStructuredBuffer<int> rotation_gradient_accum_z,
    RWStructuredBuffer<int> rotation_gradient_accum_w,
    RWStructuredBuffer<int> color_gradient_accum_r,
    RWStructuredBuffer<int> color_gradient_accum_g,
    RWStructuredBuffer<int> color_gradient_accum_b,
    RWStructuredBuffer<int> opacity_gradient_accum
)
{
    GaussianGradients grad;

    grad.mean = float3(
        fromFixed(mean_gradient_accum_x[g_idx]),
        fromFixed(mean_gradient_accum_y[g_idx]),
        fromFixed(mean_gradient_accum_z[g_idx])
    ) * inv_num_views;

    grad.mean2d_mag = fromFixed(mean2d_gradient_accum_magnitude[g_idx]);

    grad.scale = float3(
        fromFixed(scale_gradient_accum_x[g_idx]),
        fromFixed(scale_gradient_accum_y[g_idx]),
        fromFixed(scale_gradient_accum_z[g_idx])
    ) * inv_num_views;

    grad.rotation = float4(
        fromFixed(rotation_gradient_accum_x[g_idx]),
        fromFixed(rotation_gradient_accum_y[g_idx]),
        fromFixed(rotation_gradient_accum_z[g_idx]),
        fromFixed(rotation_gradient_accum_w[g_idx])
    ) * inv_num_views;

    grad.color = float3(
        fromFixed(color_gradient_accum_r[g_idx]),
        fromFixed(color_gradient_accum_g[g_idx]),
        fromFixed(color_gradient_accum_b[g_idx])
    ) * inv_num_views;

    grad.opacity = fromFixed(opacity_gradient_accum[g_idx]) * inv_num_views;

    return grad;
}

void adam_update_gaussian(
    inout Struct_Gaussian3D gaussian,
    in GaussianGradients grad,
    inout Struct_AdamState state,
    in AdamConfig config,
    bool descend_mean,
    bool descend_scale,
    bool descend_rotation,
    bool descend_color,
    bool descend_opacity
)
{
    if (descend_mean)
    {
        gaussian.mean = adam_step_3d(
            gaussian.mean, grad.mean,
            state.mean_m, state.mean_v,
            config.lr_mean,
            config.bias_correction1, config.bias_correction2
        );
    }

    if (descend_scale)
    {
        gaussian.log_scale = adam_step_3d(
            gaussian.log_scale, grad.scale,
            state.scale_m, state.scale_v,
            config.lr_scale,
            config.bias_correction1, config.bias_correction2
        );
        gaussian.log_scale = clamp(gaussian.log_scale, -6.0f, 2.0f);
    }

    if (descend_rotation)
    {
        gaussian.rotation_quat = adam_step_4d(
            gaussian.rotation_quat, grad.rotation,
            state.rotation_m, state.rotation_v,
            config.lr_rotation,
            config.bias_correction1, config.bias_correction2
        );
    }

    if (descend_color)
    {
        gaussian.color_logit = adam_step_3d(
            gaussian.color_logit, grad.color,
            state.color_m, state.color_v,
            config.lr_color,
            config.bias_correction1, config.bias_correction2
        );
    }

    if (descend_opacity)
    {
        gaussian.opacity = adam_step_1d(
            gaussian.opacity, grad.opacity,
            state.opacity_m, state.opacity_v,
            config.lr_opacity,
            config.bias_correction1, config.bias_correction2
        );
        gaussian.opacity = clamp(gaussian.opacity, 0.0f, 1.0f);
    }
}

bool is_large_gaussian(Struct_Gaussian3D g)
{
    float3 splat_linear_scale = exp(g.log_scale);
    float max_scale = max(max(splat_linear_scale.x, splat_linear_scale.y), splat_linear_scale.z);
    return max_scale > 0.01f;
}

bool should_densify(float grad_magnitude, uint batch_iter, uint start_iter, uint end_iter)
{
    if (batch_iter < start_iter || batch_iter % 100 != 0 || batch_iter > end_iter)
        return false;
    return grad_magnitude > 0.0002f;
}

bool should_prune(Struct_Gaussian3D g, uint batch_iter, uint start_iter)
{
    if (!(batch_iter % 100 == 0 && batch_iter >= start_iter))
        return false;

    if (g.opacity < 0.005f)
        return true;

    float3 scale = exp(g.log_scale);
    if (max(max(scale.x, scale.y), scale.z) > 0.15f)
        return true;

    return false;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void updateGaussiansAdam(
    uint3 DTid : SV_DispatchThreadID,
    uniform uint num_gaussians,
    uniform uint num_views,
    uniform uint iteration,
    uniform bool descend_mean,
    uniform bool descend_scale,
    uniform bool descend_rotation,
    uniform bool descend_color,
    uniform bool descend_opacity,
    uniform bool enable_densification,
    uniform uint start_densification_iter,
    uniform uint end_densification_iter,
    uniform uint max_splats,
    RWStructuredBuffer<Struct_Gaussian3D> input_gaussians,
    RWStructuredBuffer<Struct_AdamState> adam_state,
    RWStructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<uint> pixel_coverage_count_per_gaussian,
    // Gradient buffers (read then clear)
    RWStructuredBuffer<int> mean_gradient_accum_x,
    RWStructuredBuffer<int> mean_gradient_accum_y,
    RWStructuredBuffer<int> mean_gradient_accum_z,
    RWStructuredBuffer<int> mean2d_gradient_accum_magnitude,
    RWStructuredBuffer<int> scale_gradient_accum_x,
    RWStructuredBuffer<int> scale_gradient_accum_y,
    RWStructuredBuffer<int> scale_gradient_accum_z,
    RWStructuredBuffer<int> rotation_gradient_accum_x,
    RWStructuredBuffer<int> rotation_gradient_accum_y,
    RWStructuredBuffer<int> rotation_gradient_accum_z,
    RWStructuredBuffer<int> rotation_gradient_accum_w,
    RWStructuredBuffer<int> color_gradient_accum_r,
    RWStructuredBuffer<int> color_gradient_accum_g,
    RWStructuredBuffer<int> color_gradient_accum_b,
    RWStructuredBuffer<int> opacity_gradient_accum
)
{
    uint g_idx = DTid.x;
    if (g_idx >= num_gaussians)
        return;

    uint batch_iter = iteration / num_views;
    
    // Only update after processing all views in batch
    if (iteration % num_views != (num_views - 1))
        return;

    Struct_Gaussian3D in_g = input_gaussians[g_idx];
    float inv_num_views = 1.0f / float(num_views);

    GaussianGradients grad = load_gradients(
        g_idx, inv_num_views,
        mean_gradient_accum_x, mean_gradient_accum_y, mean_gradient_accum_z,
        mean2d_gradient_accum_magnitude,
        scale_gradient_accum_x, scale_gradient_accum_y, scale_gradient_accum_z,
        rotation_gradient_accum_x, rotation_gradient_accum_y, rotation_gradient_accum_z, rotation_gradient_accum_w,
        color_gradient_accum_r, color_gradient_accum_g, color_gradient_accum_b,
        opacity_gradient_accum
    );

    // Densification (split/clone)
    if (enable_densification && should_densify(grad.mean2d_mag / float(num_views), batch_iter, start_densification_iter, end_densification_iter))
    {
        if (is_large_gaussian(in_g))
        {
            // Split - handled separately or mark for split
            // For now, mark by reducing opacity (will be processed in compaction)
        }
        // Clone logic would go here
    }

    // Pruning
    if (enable_densification && should_prune(in_g, batch_iter, start_densification_iter))
    {
        in_g.opacity = 0.0f;
        input_gaussians[g_idx] = in_g;
        adam_state[g_idx] = zero_adam_state();
        return;
    }

    // Opacity reset every 3000 batches
    if (enable_densification && batch_iter % 3000 == 0 && batch_iter > 0)
    {
        in_g.opacity = min(in_g.opacity, 0.01f);
        Struct_AdamState state_reset = adam_state[g_idx];
        state_reset.opacity_m = 0.0f;
        state_reset.opacity_v = 0.0f;
        adam_state[g_idx] = state_reset;
    }

    // Adam update
    Struct_AdamState state = adam_state[g_idx];

    AdamConfig config = make_adam_config(
        1.6e-4f,  // lr_mean
        2e-3f,    // lr_scale
        1e-3f,    // lr_rotation
        2.5e-3f,  // lr_color
        5e-2f,    // lr_opacity
        batch_iter + 1
    );

    adam_update_gaussian(
        in_g, grad, state, config,
        descend_mean, descend_scale, descend_rotation, descend_color, descend_opacity
    );

    input_gaussians[g_idx] = in_g;
    adam_state[g_idx] = state;

    // Clear gradients for this gaussian
    mean_gradient_accum_x[g_idx] = 0;
    mean_gradient_accum_y[g_idx] = 0;
    mean_gradient_accum_z[g_idx] = 0;
    mean2d_gradient_accum_magnitude[g_idx] = 0;
    scale_gradient_accum_x[g_idx] = 0;
    scale_gradient_accum_y[g_idx] = 0;
    scale_gradient_accum_z[g_idx] = 0;
    rotation_gradient_accum_x[g_idx] = 0;
    rotation_gradient_accum_y[g_idx] = 0;
    rotation_gradient_accum_z[g_idx] = 0;
    rotation_gradient_accum_w[g_idx] = 0;
    color_gradient_accum_r[g_idx] = 0;
    color_gradient_accum_g[g_idx] = 0;
    color_gradient_accum_b[g_idx] = 0;
    opacity_gradient_accum[g_idx] = 0;
}

// ============================================================================
// GAUSSIAN INITIALIZATION KERNEL
// ============================================================================

[shader("compute")]
[numthreads(256, 1, 1)]
void initializeGaussians(
    uint3 DTid : SV_DispatchThreadID,
    uniform uint num_gaussians,
    uniform float3 scene_min,
    uniform float3 scene_max,
    uniform float initial_scale,
    uniform float initial_opacity,
    RWStructuredBuffer<Struct_Gaussian3D> input_gaussians,
    RWStructuredBuffer<Struct_AdamState> adam_state,
    RWStructuredBuffer<uint> num_gaussians_counter
)
{
    uint idx = DTid.x;
    if (idx >= num_gaussians)
        return;

    if (idx == 0)
        num_gaussians_counter[0] = num_gaussians;

    // Initialize with random positions in scene bounds
    float3 scene_size = scene_max - scene_min;
    
    // Use index-based pseudo-random positioning
    uint seed = idx;
    float rx = random(seed);
    float ry = random(seed + 1000);
    float rz = random(seed + 2000);

    Struct_Gaussian3D g;
    g.mean = scene_min + float3(rx, ry, rz) * scene_size;
    g.log_scale = float3(log(initial_scale), log(initial_scale), log(initial_scale));
    g.rotation_quat = float4(1, 0, 0, 0);  // Identity quaternion
    g.opacity = initial_opacity;
    g.color_logit = float3(0, 0, 0);  // Mid-gray in logit space

    input_gaussians[idx] = g;
    adam_state[idx] = zero_adam_state();
}
