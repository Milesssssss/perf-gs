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

#include "generated/execution_contract.slang"
static const uint2 TILE_DIM = uint2(CONTRACT_TILE_WIDTH, CONTRACT_TILE_HEIGHT);
static const uint CLUSTER_SIZE = CONTRACT_CLUSTER_SIZE;

// IEEE float32 -> sortable uint32. For positive finite floats, flipping
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
    // Cluster-cull mask: seeded all-ones when disabled.
    StructuredBuffer<uint> cluster_visible,
    uniform float z_near,
    // Capacity of the sort-pair buffers (keys_lo/hi/values). The exclusive
    // scan that produced tile_offset_per_gaussian is UNBOUNDED — for a
    // viewpoint whose total (tile,depth,gaussian) pairs exceed this capacity,
    // write_slot runs past the buffers and scribbles into whatever is
    // allocated next (tile_start/tile_end), producing corrupted per-tile
    // ranges → alternating black tiles for that view. Guarding each write by
    // this cap turns overflow into graceful degradation (excess pairs are
    // dropped: some splats go missing from that frame) instead of memory
    // corruption. The pipeline also clamps scan_total to this same capacity so
    // the sort/tile-range stages never READ past the written region.
    uniform uint max_sort_pairs)
{
    uint gid = dtid.x;
    if (gid >= num_gaussians_counter[0]) return;
    // Cluster-frustum cull: skip culled clusters BEFORE loading gaussians[gid]
    // and projected_2d[gid]. bin_count assigned tile_count=0 for these slots
    // so tile_offset_per_gaussian[gid] would reserve 0 write space anyway; the
    // benefit here is eliminating the two 32-byte-sector reads (56 B + 24 B)
    // per culled slot. Wave-uniform under Morton clustering.
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

    // AccuTile: emit only the tiles the ellipse actually crosses. This walk
    // MUST visit the same tiles bin_count summed (same stored projection →
    // bit-identical ellipse), or write_slot overruns/underruns the scanned
    // offset and the sort reads garbage. Shared helpers keep the two in lock
    // step; see accutile.slang.
    EllipseParams e;
    if (!computeEllipseParams(cov2d, g.opacity, e))
        return;
    float x_d_args, y_d_args;
    computeExtremumOffsets(e, x_d_args, y_d_args);

    if (accuTileIterateRows(tile_min, tile_max)) //We iterate: for each row, for each col
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
    else //her instead we iterate: for each col, for each row
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
