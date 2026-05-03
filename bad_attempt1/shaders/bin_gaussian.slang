// Unnamed technique, shader fillBuffer
/*$(ShaderResources)*/

static float2 cameraPos = /*$(Variable:cameraPos)*/;
static float2 scale = /*$(Variable:scale)*/;
static float2 frame_dim = /*$(Variable:frameDim)*/;
static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

#define MAX_GAUSSIANS_PER_TILE 1024
#define TILE_SIZE 16

#include "utils.slang"

void compute2DBoundingBox(float2 mean_2d, float3 cov2d, out float2 bbox_min, out float2 bbox_max)
{
    // eigenvalues of 2x2 covariance matrix to get major/minor axes
    float a = cov2d.x; // cov_00
    float b = cov2d.y; // cov_01
    float c = cov2d.z; // cov_11

    float trace = a + c;
    float det = a * c - b * b;

    float discriminant = max(0.0, trace * trace * 0.25 - det);
    float lambda1 = trace * 0.5 + sqrt(discriminant);
    float lambda2 = trace * 0.5 - sqrt(discriminant);

    // usula 3-sigma thing
    float radius_x = 3.1 * sqrt(max(lambda1, 0.0));
    float radius_y = 3.1 * sqrt(max(lambda2, 0.0));

    // Conservative
    float radius = max(radius_x, radius_y);

    bbox_min = mean_2d - float2(radius, radius);
    bbox_max = mean_2d + float2(radius, radius);
}

/*$(_compute:binGaussians)*/ (uint3 DTid: SV_DispatchThreadID)
{
    if (DTid.x == 0)
        training_step_counter[0] = training_step_counter[0] + 1;

    uint gid = DTid.x;
    if (gid >= num_gaussians_counter[0]) return;

    Struct_Gaussian3D gaussian = input_gaussians[gid]; // Now using 3D gaussians

    const uint2 numTiles = ((uint2)frame_dim + TILE_SIZE - 1) / TILE_SIZE;

    //Should these be transposed?

    float4x4 world_to_view_t = float4x4(
        target_camera_view_matrix[0],
        target_camera_view_matrix[1],
        target_camera_view_matrix[2],
        target_camera_view_matrix[3]
    );

    float4x4 view_to_screen_t = float4x4(
        target_camera_projection_matrix[0],
        target_camera_projection_matrix[1],
        target_camera_projection_matrix[2],
        target_camera_projection_matrix[3],
    );

    Struct_Projected2DGaussian proj_2d;

    projectGaussian3DTo2D_diff(
        gaussian.mean,
        gaussian.log_scale,
        gaussian.rotation_quat,
        world_to_view_t,
        view_to_screen_t,
        frame_dim,
        proj_2d.mean,
        proj_2d.cov2d,
        proj_2d.depth_vs
    );
    proj_2d.depth_vs = -proj_2d.depth_vs; 
    // We add the projected gaussian in the same index as the original 3d gaussian buffer
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
    //screen_tiles_indices[0] = gid;

    // Bin into tiles
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