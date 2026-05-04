// Phase 2 of the uncapped binning pipeline: emit (key, value) pairs.
//
// For each live gaussian, re-run AccuTile and for every tile it touches
// write one pair at tile_offset_per_gaussian[gid] + local_tile_index:
//   key_hi = tile_id                         (32 bits)
//   key_lo = float_to_sortable_u32(depth_vs)  (32 bits)
//   value  = gaussian_idx
//
// The radix sort then sorts by the composite 64-bit key (hi:lo), which
// groups entries by tile_id and within each tile orders them by ascending
// depth (front-to-back).

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/snugbox.slang"
#include "../../primitives/gaussian/accutile.slang"

static const uint TILE_SIZE = 16;

// IEEE float32 → sortable uint32. For positive finite floats, flipping
// the sign bit makes the uint32 bit-pattern ascend with the float value.
// All our inputs have depth_vs > z_near > 0 by construction (bin cull).
uint depthToSortable(float depth)
{
    // asuint reinterprets without conversion.
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
    uniform float z_near)
{
    uint gid = dtid.x;
    if (gid >= num_gaussians_counter[0]) return;

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

    uint write_slot = tile_offset_per_gaussian[gid];
    uint key_lo = depthToSortable(proj.depth_vs);

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
            for (uint tx = tx_min; tx <= tx_max; ++tx)
            {
                uint tile_id = ty * num_tiles.x + tx;
                keys_hi[write_slot] = tile_id;
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
            for (uint ty = ty_min; ty <= ty_max; ++ty)
            {
                uint tile_id = ty * num_tiles.x + tx;
                keys_hi[write_slot] = tile_id;
                keys_lo[write_slot] = key_lo;
                values[write_slot] = gid;
                write_slot++;
            }
        }
    }
}
