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

/*$(_compute:forwardPassDebug)*/ (
    uint3 DTid: SV_DispatchThreadID,
    uint3 Gid: SV_GroupID
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
    float3 simil_vbuffer_color_visualized = 0;

    // Front-to-back
    for (uint i = 0; i < tile_gaussian_count; i++)
    {
        uint gaussian_idx = screen_bin_tiles_gaussian_ids[start_idx + i];

        Struct_Gaussian3D g3d           = input_gaussians[gaussian_idx];
        Struct_Projected2DGaussian proj = projected_gaussians_2d[gaussian_idx];

        float gauss_i_alpha = evalGaussian2DFromCov(
            pixel,
            proj.mean, 
            proj.cov2d,   
            g3d.opacity   
        );

        gauss_i_alpha = saturate(gauss_i_alpha);

        float3 color_rgb = sigmoid3(g3d.color_logit);
        rendered_color += color_rgb * gauss_i_alpha * transmittance;

        if (display_mode == 2)
            simil_vbuffer_color_visualized += index_to_random_color_rgb(gaussian_idx) * gauss_i_alpha * transmittance;
        
        transmittance *= (1.0 - gauss_i_alpha);
    }

    //if (tile_gaussian_count > 0)
    //    float4 output_color = float4(pow(float3(1, 0, 0), 1.0 / 2.2), 1.0f);

    float4 output_color = float4(pow(rendered_color, 1.0 / 2.2), 1.0f);

    output_debug_view[pixel] = output_color;
    
}