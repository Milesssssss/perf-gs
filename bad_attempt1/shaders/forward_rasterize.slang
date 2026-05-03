// 2D Gaussian Gradient Descent - Image-based Loss
// Compares rendered input Gaussian against a target IMAGE (not target Gaussian params)
/*$(ShaderResources)*/

#define MAX_GAUSSIANS_PER_PIXEL 10
#define MAX_GAUSSIANS_PER_TILE 1024
#define TILE_SIZE 16

static float2 camera_pos = /*$(Variable:cameraPos)*/;
static float2 scale = /*$(Variable:scale)*/;
static float2 frame_dim = /*$(Variable:frameDim)*/;
//static float4x4 world_to_view = transpose(/*$(Variable:ViewMtx)*/);
//static float4x4 view_to_screen = transpose(/*$(Variable:ProjMtx)*/);
static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

static uint display_mode = /*$(Variable:display_mode)*/;

#include "utils.slang"

groupshared uint pixel_count_per_splat_gs;

/*$(_compute:forwardPass)*/ (
    uint3 DTid: SV_DispatchThreadID,
    uint3 Gid: SV_GroupID,
    uint Gidx: SV_GroupIndex
)
{
    uint2 pixel = DTid.xy;

    if (pixel.x >= frame_dim.x || pixel.y >= frame_dim.y)
        return;

    uint2 tile_coord = DTid.xy / TILE_SIZE;
    uint2 num_tiles = ((uint2)frame_dim + TILE_SIZE - 1) / TILE_SIZE;

    uint tileIdx = tile_coord.y * num_tiles.x + tile_coord.x;

    uint tile_gaussian_count = min(screen_bin_tiles_count[tileIdx], MAX_GAUSSIANS_PER_TILE);
    uint start_idx = tileIdx * MAX_GAUSSIANS_PER_TILE;

    float3 rendered_color = float3(0, 0, 0);
    float transmittance = 1.0;
    uint pixel_accum_count = 0;
    float3 simil_vbuffer_color_visualized = 0;

    // Front-to-back
    for (uint i = 0; i < tile_gaussian_count; i++)
    {
        if (Gidx == 0)
            pixel_count_per_splat_gs = 0;

        GroupMemoryBarrierWithGroupSync();

        uint gaussian_idx = screen_bin_tiles_gaussian_ids[start_idx + i];

        Struct_Gaussian3D g3d           = input_gaussians[gaussian_idx];
        Struct_Projected2DGaussian proj = projected_gaussians_2d[gaussian_idx];

        float gauss_i_alpha = evalGaussian2DFromCov(
            pixel,
            proj.mean, 
            proj.cov2d,   
            g3d.opacity   
        );

        gauss_i_alpha = gauss_i_alpha;
        //TODO: probably some gradient issue if I stop early (?). It caused issues.
        
        pixel_accum_count++;
        /// else
        //    break;
        //
        InterlockedAdd(pixel_count_per_splat_gs, 1);

        GroupMemoryBarrierWithGroupSync();
        if (Gidx == 0) //We basically know that for now in reality this will be always 256 (16x16), as there are no early returns or density clamps, but should be added for perf
            InterlockedAdd(pixel_coverage_count_per_gaussian[gaussian_idx], pixel_count_per_splat_gs);
        
        AllMemoryBarrierWithGroupSync();

        float3 color_rgb = sigmoid3(g3d.color_logit);
        //color_rgb = float3(1, 0, 0);
        rendered_color += color_rgb * gauss_i_alpha * transmittance;

        InterlockedMax(output_debug_info[pixel], toFixed(proj.depth_vs));
        
        if (display_mode == 2)
            simil_vbuffer_color_visualized += index_to_random_color_rgb(gaussian_idx) * gauss_i_alpha * transmittance;

        transmittance *= (1.0 - gauss_i_alpha);

        if (transmittance < 0.001f)
            break;
    }

    //if (tile_gaussian_count > 0)
    //    float4 output_color = float4(pow(float3(1, 0, 0), 1.0 / 2.2), 1.0f);

    float4 output_color = float4(pow(rendered_color, 1.0 / 2.2), 1.0f);

    //if(display_mode == 1)
    //    output_debug_info[pixel] = target_image[pixel]; // Red = unsorted tile
    //if (display_mode == 2)
    //    output_debug_info[pixel] = float4(simil_vbuffer_color_visualized, 1);
    
    gaussian_count_per_pixel[pixel] = pixel_accum_count;
    output_transmittance[pixel]     = transmittance;
    output_image[pixel]             = output_color;

}