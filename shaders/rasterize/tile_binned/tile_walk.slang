// Generic tile-enumeration walks over an IScreenFootprint, shared by the two
// binning kernels so they stay in structural lock-step:
//   * bin_count calls countTiles() to SUM the touched tiles (sizes the scan).
//   * bin_emit  calls emitTiles()  to WRITE one (tile,depth,primitive) pair
//     per touched tile at the scanned offset.
// Both drive the SAME footprint methods with the SAME inputs, so they visit an
// identical tile set — a mismatch would desync the prefix-scan offsets from the
// emitted pairs and the radix sort would read garbage. Keeping the walk here,
// once, generic over the footprint, makes that agreement structural rather than
// two hand-kept-in-sync copies (and lets any primitive reuse both kernels).

#ifndef TRAINER_TILE_WALK_SLANG
#define TRAINER_TILE_WALK_SLANG

#include "../../core/interfaces.slang"

// Count the tiles the footprint crosses within the tile rectangle
// [tile_min, tile_max]. `bbox_min/bbox_max` are the frame-clamped AABB edges
// the per-strip crossing tests refine against.
uint countTiles<F : IScreenFootprint>(
    F fp, uint2 tile_min, uint2 tile_max, float2 bbox_min, float2 bbox_max)
{
    uint count = 0u;
    if (fp.iterateRows(tile_min, tile_max))
    {
        for (uint ty = tile_min.y; ty <= tile_max.y; ++ty)
        {
            uint tx_lo, tx_hi;
            if (fp.rowTiles(bbox_min, bbox_max, ty, tile_min.x, tile_max.x, tx_lo, tx_hi))
                count += tx_hi - tx_lo + 1u;
        }
    }
    else
    {
        for (uint tx = tile_min.x; tx <= tile_max.x; ++tx)
        {
            uint ty_lo, ty_hi;
            if (fp.colTiles(bbox_min, bbox_max, tx, tile_min.y, tile_max.y, ty_lo, ty_hi))
                count += ty_hi - ty_lo + 1u;
        }
    }
    return count;
}

// Emit one sort pair per touched tile, starting at `write_slot`. The composite
// key is (tile_id : depth) — keys_hi holds tile_id = ty*num_tiles_x + tx, keys_lo
// the depth-sortable bits — and `value` is the primitive index. Writes past
// `max_sort_pairs` are dropped (graceful overflow, never memory corruption),
// mirroring the exclusive-scan capacity clamp on the read side.
void emitTiles<F : IScreenFootprint>(
    F fp, uint2 tile_min, uint2 tile_max, float2 bbox_min, float2 bbox_max,
    uint num_tiles_x, uint key_lo, uint value, uint write_slot, uint max_sort_pairs,
    RWStructuredBuffer<uint> keys_hi, RWStructuredBuffer<uint> keys_lo,
    RWStructuredBuffer<uint> values)
{
    if (fp.iterateRows(tile_min, tile_max))
    {
        for (uint ty = tile_min.y; ty <= tile_max.y; ++ty)
        {
            uint tx_lo, tx_hi;
            if (!fp.rowTiles(bbox_min, bbox_max, ty, tile_min.x, tile_max.x, tx_lo, tx_hi))
                continue;
            for (uint tx = tx_lo; tx <= tx_hi; ++tx)
            {
                if (write_slot >= max_sort_pairs) return;  // overflow: drop, don't corrupt
                keys_hi[write_slot] = ty * num_tiles_x + tx;
                keys_lo[write_slot] = key_lo;
                values[write_slot] = value;
                write_slot++;
            }
        }
    }
    else
    {
        for (uint tx = tile_min.x; tx <= tile_max.x; ++tx)
        {
            uint ty_lo, ty_hi;
            if (!fp.colTiles(bbox_min, bbox_max, tx, tile_min.y, tile_max.y, ty_lo, ty_hi))
                continue;
            for (uint ty = ty_lo; ty <= ty_hi; ++ty)
            {
                if (write_slot >= max_sort_pairs) return;  // overflow: drop, don't corrupt
                keys_hi[write_slot] = ty * num_tiles_x + tx;
                keys_lo[write_slot] = key_lo;
                values[write_slot] = value;
                write_slot++;
            }
        }
    }
}

#endif // TRAINER_TILE_WALK_SLANG
