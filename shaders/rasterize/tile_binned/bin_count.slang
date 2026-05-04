// Phase 1 of the uncapped binning pipeline: per-gaussian tile-touch count.
//
// For each live gaussian:
//   1. Project to 2D (same as old bin.slang).
//   2. Near-plane cull, SnugBox, clamp to frame.
//   3. Use AccuTile to count the exact number of tiles the ellipse touches.
//   4. Write projected_2d (so forward/backward can reuse it) and
//      tile_count_per_gaussian[gid].
//
// The caller then prefix-scans tile_count_per_gaussian to get write
// offsets, and dispatches bin_emit to actually write (key, value) pairs.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/snugbox.slang"
#include "../../primitives/gaussian/accutile.slang"

static const uint TILE_SIZE = 16;

[shader("compute")]
[numthreads(256, 1, 1)]
void binCount(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    RWStructuredBuffer<Projected2DGaussian> projected_2d,
    RWStructuredBuffer<uint> tile_count_per_gaussian,  // [max_splats]
    uniform float z_near,
    uniform uint max_splats)
{
    uint gid = dtid.x;
    if (gid >= max_splats) return;

    // Default: zero tiles (culled). Write now so the scan sees a clean
    // value even if we early-return.
    tile_count_per_gaussian[gid] = 0;

    if (gid >= num_gaussians_counter[0]) return;

    Gaussian3D g = gaussians[gid];

    float2 mean_2d;
    float3 cov2d;
    float depth_vs;
    projectGaussian3DTo2D(
        g.mean, g.log_scale, g.rot_quat,
        camera[0].view_matrix, camera[0].proj_matrix, camera[0].image_size,
        mean_2d, cov2d, depth_vs);

    Projected2DGaussian proj;
    proj.mean = mean_2d;
    proj.cov2d = cov2d;
    proj.depth_vs = depth_vs;
    projected_2d[gid] = proj;

    if (depth_vs < z_near) return;

    float2 frame_dim = camera[0].image_size;
    float2 bbox_min, bbox_max;
    if (!snugBox(mean_2d, cov2d, g.opacity, bbox_min, bbox_max))
        return;

    bbox_min = max(bbox_min, float2(0.0, 0.0));
    bbox_max = min(bbox_max, frame_dim);
    if (bbox_min.x >= bbox_max.x || bbox_min.y >= bbox_max.y)
        return;

    uint2 num_tiles = ((uint2)frame_dim + TILE_SIZE - 1) / TILE_SIZE;
    uint2 tile_min = uint2(bbox_min) / TILE_SIZE;
    uint2 tile_max = uint2(bbox_max) / TILE_SIZE;
    tile_max = min(tile_max, num_tiles - 1);

    EllipseParams e;
    if (!computeEllipseParams(cov2d, g.opacity, e)) return;
    float x_d_args, y_d_args;
    computeExtremumOffsets(e, x_d_args, y_d_args);

    uint tiles_y_span = tile_max.y - tile_min.y + 1;
    uint tiles_x_span = tile_max.x - tile_min.x + 1;

    uint count = 0;
    if (tiles_y_span <= tiles_x_span)
    {
        for (uint ty = tile_min.y; ty <= tile_max.y; ++ty)
        {
            float row_y_bot = float(ty * TILE_SIZE);
            float row_y_top = float((ty + 1) * TILE_SIZE);
            float2 xr = accuTileRowXRange(
                e, mean_2d.y, bbox_min.x, bbox_max.x,
                y_d_args, mean_2d, row_y_bot, row_y_top);
            if (xr.y < xr.x) continue;
            uint tx_min = uint(max(xr.x, 0.0)) / TILE_SIZE;
            uint tx_max = uint(max(xr.y, 0.0)) / TILE_SIZE;
            tx_min = max(tx_min, tile_min.x);
            tx_max = min(tx_max, tile_max.x);
            count += (tx_max - tx_min + 1);
        }
    }
    else
    {
        for (uint tx = tile_min.x; tx <= tile_max.x; ++tx)
        {
            float col_x_left  = float(tx * TILE_SIZE);
            float col_x_right = float((tx + 1) * TILE_SIZE);
            float2 yr = accuTileColYRange(
                e, mean_2d.x, bbox_min.y, bbox_max.y,
                x_d_args, mean_2d, col_x_left, col_x_right);
            if (yr.y < yr.x) continue;
            uint ty_min = uint(max(yr.x, 0.0)) / TILE_SIZE;
            uint ty_max = uint(max(yr.y, 0.0)) / TILE_SIZE;
            ty_min = max(ty_min, tile_min.y);
            ty_max = min(ty_max, tile_max.y);
            count += (ty_max - ty_min + 1);
        }
    }
    tile_count_per_gaussian[gid] = count;
}
