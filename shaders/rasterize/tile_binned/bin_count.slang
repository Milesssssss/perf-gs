// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Binning phase 1: project each live gaussian and count the tiles its ellipse touches (AccuTile).
// The caller prefix-scans tile_count_per_gaussian into bin_emit's write offsets.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/snugbox.slang"
#include "../../primitives/gaussian/accutile.slang"

// TODO: try merging this and bin_emit.slang into a single kernel.

#include "common/execution.slang"
static const uint2 TILE_DIM = uint2(TILE_WIDTH, TILE_HEIGHT);

[shader("compute")]
[numthreads(256, 1, 1)]
void binCount(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    RWStructuredBuffer<Projected2DGaussian> projected_2d,
    RWStructuredBuffer<BlendData> blend_data,          // [max_splats], compact blend gather record
    RWStructuredBuffer<uint> tile_count_per_gaussian,  // [max_splats]
    StructuredBuffer<uint> cluster_visible,            // [ceil(max_splats/128)]
    StructuredBuffer<float> filter_3D,                 // [max_splats], mip 3D size (0=off)
    uniform uint mip_enabled,
    uniform float kernel_size,                         // 2D dilation (0.3 vanilla / 0.1 mip)
    uniform float z_near,
    uniform uint max_splats)
{
    uint gid = dtid.x;
    if (gid >= max_splats) return;

    // Written before any early return so the scan never sees a stale count.
    tile_count_per_gaussian[gid] = 0;

    if (gid >= num_gaussians_counter[0]) return;

    // Culled slots keep stale projected_2d/blend_data, which no downstream kernel reads.
    if (cluster_visible[gid / CLUSTER_SIZE] == 0u) return;

    Gaussian3D g = gaussians[gid];

    float2 mean_2d;
    float3 cov2d;
    float depth_vs;
    float cov2d_det;

    // Mip-Splatting 3D filter and opacity compensation (1.0 when off); f3d = 0, kernel_size = 0.3 is vanilla.
    float f3d = (mip_enabled != 0u) ? filter_3D[gid] : 0.0;
    float comp;

    GaussianModel model;
    model.mean = g.mean;
    model.log_scale = g.log_scale;
    model.rot_quat = g.rot_quat;
    model.projectInto(camera[0], f3d, kernel_size, mip_enabled,
                      mean_2d, cov2d, depth_vs, cov2d_det, comp);

    Projected2DGaussian proj;
    proj.mean = mean_2d;
    proj.cov2d = cov2d;
    proj.depth_vs = depth_vs;
    proj.opacity_compensation = comp;
    projected_2d[gid] = proj;

    // Compact record the blend passes gather instead of gaussians + projected_2d.
    BlendData bd;
    bd.mean = mean_2d;
    bd.cov2d = cov2d;
    bd.opacity = g.opacity;
    bd.sh0 = g.sh0;
    bd.opacity_compensation = comp;
    blend_data[gid] = bd;

    if (depth_vs < z_near) return;

    float2 frame_dim = camera[0].image_size;

    float2 bbox_min, bbox_max;
    if (!snugBox(mean_2d, cov2d, g.opacity, bbox_min, bbox_max))
        return;

    bbox_min = max(bbox_min, float2(0.0, 0.0));
    bbox_max = min(bbox_max, frame_dim);
    if (bbox_min.x >= bbox_max.x || bbox_min.y >= bbox_max.y)
        return;

    uint2 num_tiles = ((uint2)frame_dim + TILE_DIM - 1u) / TILE_DIM;
    uint2 tile_min = uint2(bbox_min) / TILE_DIM;
    uint2 tile_max = uint2(bbox_max) / TILE_DIM;
    tile_max = min(tile_max, num_tiles - 1);

    // AccuTile count; bin_emit replays this exact walk, so its writes match the scanned offsets.
    EllipseParams e;
    if (!computeEllipseParams(cov2d, g.opacity, e))
        return;
    float x_d_args, y_d_args;
    computeExtremumOffsets(e, x_d_args, y_d_args);

    uint count = 0u;
    if (accuTileIterateRows(tile_min, tile_max))
    {
        for (uint ty = tile_min.y; ty <= tile_max.y; ++ty)
        {
            uint tx_lo, tx_hi;
            if (accuTileRowTiles(e, mean_2d, bbox_min, bbox_max, y_d_args,
                                 ty, tile_min.x, tile_max.x, tx_lo, tx_hi))
                count += tx_hi - tx_lo + 1u;
        }
    }
    else
    {
        for (uint tx = tile_min.x; tx <= tile_max.x; ++tx)
        {
            uint ty_lo, ty_hi;
            if (accuTileColTiles(e, mean_2d, bbox_min, bbox_max, x_d_args,
                                 tx, tile_min.y, tile_max.y, ty_lo, ty_hi))
                count += ty_hi - ty_lo + 1u;
        }
    }
    tile_count_per_gaussian[gid] = count;
}
