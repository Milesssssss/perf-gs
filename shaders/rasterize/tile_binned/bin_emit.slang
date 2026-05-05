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

    uint write_slot = tile_offset_per_gaussian[gid];
    uint key_lo = depthToSortable(proj.depth_vs);

    for (uint ty = tile_min.y; ty <= tile_max.y; ++ty)
    {
        for (uint tx = tile_min.x; tx <= tile_max.x; ++tx)
        {
            uint tile_id = ty * num_tiles.x + tx;
            keys_hi[write_slot] = tile_id;
            keys_lo[write_slot] = key_lo;
            values[write_slot] = gid;
            write_slot++;
        }
    }
}
