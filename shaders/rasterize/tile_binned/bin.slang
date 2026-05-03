// Binning pass: project each gaussian, compute its pixel-space bounding
// box, and append its index to the per-tile list for every tile the bbox
// touches. Also writes the Projected2DGaussian record so the forward pass
// doesn't re-project.
//
// One thread per gaussian. Tile capacity is hard-capped at
// MAX_GAUSSIANS_PER_TILE; overflow is silently dropped in v0 (see
// PLAN.md — global radix sort without a cap is a v1 upgrade).

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"

static const uint TILE_SIZE = 16;
static const uint MAX_GAUSSIANS_PER_TILE = 1024;

void compute2DBoundingBox(float2 mean_2d, float3 cov2d,
                          out float2 bbox_min, out float2 bbox_max)
{
    // Eigenvalues of the 2x2 covariance give the principal-axis radii.
    float a = cov2d.x;
    float b = cov2d.y;
    float c = cov2d.z;
    float trace = a + c;
    float det = a * c - b * b;
    float disc = max(0.0, trace * trace * 0.25 - det);
    float lambda1 = trace * 0.5 + sqrt(disc);
    float lambda2 = trace * 0.5 - sqrt(disc);

    // 3-sigma radius, scaled slightly conservative. Matches the canonical
    // 3DGS binning choice. The 3.1 factor is historical from the paper's
    // reference implementation; we could tune it later if we see fringe
    // gaussians clipped against tile edges.
    float radius_x = 3.1 * sqrt(max(lambda1, 0.0));
    float radius_y = 3.1 * sqrt(max(lambda2, 0.0));
    float radius = max(radius_x, radius_y);

    bbox_min = mean_2d - float2(radius, radius);
    bbox_max = mean_2d + float2(radius, radius);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void binGaussians(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    RWStructuredBuffer<Projected2DGaussian> projected_2d,
    RWStructuredBuffer<uint> screen_tiles_count,
    RWStructuredBuffer<uint> screen_tiles_indices)
{
    uint gid = dtid.x;
    if (gid >= num_gaussians_counter[0])
        return;

    Gaussian3D g = gaussians[gid];

    float2 mean_2d;
    float3 cov2d;
    float depth_vs;
    projectGaussian3DTo2D(
        g.mean, g.log_scale, g.rot_quat,
        camera[0].view_matrix, camera[0].proj_matrix, camera[0].image_size,
        mean_2d, cov2d, depth_vs);

    // Record projection result for the forward pass, regardless of cull
    // so indices line up (forward also checks depth).
    Projected2DGaussian proj;
    proj.mean = mean_2d;
    proj.cov2d = cov2d;
    proj.depth_vs = depth_vs;
    projected_2d[gid] = proj;

    if (depth_vs <= 0.0)
        return;

    float2 frame_dim = camera[0].image_size;
    float2 bbox_min, bbox_max;
    compute2DBoundingBox(mean_2d, cov2d, bbox_min, bbox_max);

    bbox_min = max(bbox_min, float2(0.0, 0.0));
    bbox_max = min(bbox_max, frame_dim);
    if (bbox_min.x >= bbox_max.x || bbox_min.y >= bbox_max.y)
        return;

    uint2 num_tiles = ((uint2)frame_dim + TILE_SIZE - 1) / TILE_SIZE;
    uint2 tile_min = uint2(bbox_min) / TILE_SIZE;
    uint2 tile_max = uint2(bbox_max) / TILE_SIZE;
    tile_max = min(tile_max, num_tiles - 1);

    for (uint ty = tile_min.y; ty <= tile_max.y; ++ty)
    {
        for (uint tx = tile_min.x; tx <= tile_max.x; ++tx)
        {
            uint tile_idx = ty * num_tiles.x + tx;
            uint slot;
            InterlockedAdd(screen_tiles_count[tile_idx], 1, slot);
            if (slot < MAX_GAUSSIANS_PER_TILE)
            {
                screen_tiles_indices[tile_idx * MAX_GAUSSIANS_PER_TILE + slot] = gid;
            }
        }
    }
}
