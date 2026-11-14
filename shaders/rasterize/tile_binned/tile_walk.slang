// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Tile walks over an IScreenFootprint shared by the count and emit binning kernels, so both visit
// an identical tile set (a mismatch desyncs the scanned offsets from the emitted pairs).

#ifndef TRAINER_TILE_WALK_SLANG
#define TRAINER_TILE_WALK_SLANG

#include "../../core/interfaces.slang"

// Tiles the footprint crosses in [tile_min, tile_max]; bbox_min/bbox_max is the frame-clamped AABB.
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

// One pair per touched tile from write_slot: keys_hi = tile_id, keys_lo = sortable depth,
// value = primitive index. Writes past max_sort_pairs are dropped rather than corrupting memory.
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
