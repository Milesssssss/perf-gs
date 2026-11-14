// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Binning phase 2: per touched tile, emit key_hi = tile_id, key_lo = sortable depth, value = gaussian
// at the scanned offset; sorting by hi:lo groups tiles and orders each front-to-back.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/snugbox.slang"
#include "../../primitives/gaussian/accutile.slang"

#include "common/execution.slang"
static const uint2 TILE_DIM = uint2(TILE_WIDTH, TILE_HEIGHT);

// Order-preserving uint key; valid because the bin cull guarantees depth > z_near > 0.
uint depthToSortable(float depth)
{
    uint bits = asuint(depth);
    return bits ^ 0x80000000u;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void binEmit(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    StructuredBuffer<Projected2DGaussian> projected_2d,
    StructuredBuffer<uint> tile_offset_per_gaussian,  // exclusive scan
    RWStructuredBuffer<uint> keys_lo,
    RWStructuredBuffer<uint> keys_hi,
    RWStructuredBuffer<uint> values,
    // Cluster-cull mask: seeded all-ones when disabled.
    StructuredBuffer<uint> cluster_visible,
    uniform float z_near,
    // Sort-pair capacity; offsets are unbounded, so overflow is dropped (the host also clamps scan_total).
    uniform uint max_sort_pairs)
{
    uint gid = dtid.x;
    if (gid >= num_gaussians_counter[0]) return;
    // bin_count gave culled clusters 0 tiles; skip them before loading their data.
    if (cluster_visible[gid / CLUSTER_SIZE] == 0u) return;

    Gaussian3D g = gaussians[gid];
    Projected2DGaussian proj = projected_2d[gid];
    if (proj.depth_vs < z_near) return;

    float2 mean_2d = proj.mean;
    float3 cov2d = proj.cov2d;

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

    uint write_slot = tile_offset_per_gaussian[gid];
    uint key_lo = depthToSortable(proj.depth_vs);

    // Must visit exactly the tiles bin_count counted, or write_slot desyncs from the scanned offsets.
    EllipseParams e;
    if (!computeEllipseParams(cov2d, g.opacity, e))
        return;
    float x_d_args, y_d_args;
    computeExtremumOffsets(e, x_d_args, y_d_args);

    if (accuTileIterateRows(tile_min, tile_max))
    {
        for (uint ty = tile_min.y; ty <= tile_max.y; ++ty)
        {
            uint tx_lo, tx_hi;
            if (!accuTileRowTiles(e, mean_2d, bbox_min, bbox_max, y_d_args,
                                  ty, tile_min.x, tile_max.x, tx_lo, tx_hi))
                continue;
            for (uint tx = tx_lo; tx <= tx_hi; ++tx)
            {
                if (write_slot >= max_sort_pairs) return;  // overflow: drop, don't corrupt
                keys_hi[write_slot] = ty * num_tiles.x + tx;
                keys_lo[write_slot] = key_lo;
                values[write_slot] = gid;
                write_slot++;
            }
        }
    }
    else
    {
        for (uint tx = tile_min.x; tx <= tile_max.x; ++tx)
        {
            uint ty_lo, ty_hi;
            if (!accuTileColTiles(e, mean_2d, bbox_min, bbox_max, x_d_args,
                                  tx, tile_min.y, tile_max.y, ty_lo, ty_hi))
                continue;
            for (uint ty = ty_lo; ty <= ty_hi; ++ty)
            {
                if (write_slot >= max_sort_pairs) return;  // overflow: drop, don't corrupt
                keys_hi[write_slot] = ty * num_tiles.x + tx;
                keys_lo[write_slot] = key_lo;
                values[write_slot] = gid;
                write_slot++;
            }
        }
    }
}
