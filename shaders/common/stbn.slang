// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Tile-stratified spatiotemporal blue-noise (STBN) pixel selection.

#ifndef TRAINER_COMMON_STBN_SLANG
#define TRAINER_COMMON_STBN_SLANG

#include "generated/execution_contract.slang"

#ifndef STBN_ENABLED
#define STBN_ENABLED 0
#endif

#ifndef STBN_MAX_TILE_PIXELS
#define STBN_MAX_TILE_PIXELS CONTRACT_TILE_AREA
#endif

static const uint STBN_TILE_WIDTH  = CONTRACT_TILE_WIDTH;
static const uint STBN_TILE_HEIGHT = CONTRACT_TILE_HEIGHT;
static const uint STBN_TILE_AREA   = CONTRACT_TILE_AREA;

// Packed table layout, one uint32 per voxel (see stbn.py:pack_selection):
//   bits  0..7   slot of this pixel in its tile's selected list, 0xFF = none
//   bits  8..15  pixel-in-tile index of the slot-i selected pixel (i < count)
//   bits 16..23  selected pixels in this tile, replicated across the block
//
// The table carries the SELECTED SET, not a ranking, so nothing here knows or
// cares whether that set came from a global slice threshold or a per-tile
// argsort. Per-tile counts therefore VARY under a global threshold, which is why
// the backward reads stbnTileCount instead of assuming the nominal budget.
static const uint STBN_SLOT_MASK = 0xFFu;
static const uint STBN_SLOT_NONE = 0xFFu;
static const uint STBN_COMPACT_SHIFT = 8u;
static const uint STBN_COMPACT_MASK = 0xFFu;
static const uint STBN_COUNT_SHIFT = 16u;
static const uint STBN_COUNT_MASK = 0xFFu;

bool stbnActive(uint budget)
{
    return budget > 0u && budget < STBN_TILE_AREA;
}

// Unbiasing weight for a budget of k pixels per tile: we summed k of TILE_AREA
// terms, so scale by TILE_AREA/k. Exactly 1.0 when the mask is off, which is
// what keeps the disabled path bit-identical.
float stbnWeight(uint budget)
{
    return stbnActive(budget) ? (float(STBN_TILE_AREA) / float(budget)) : 1.0;
}

// Base index of the block covering `pixel` on `slice`. `dims` is the table's
// (width, height) in pixels; the mask tiles the screen by wrapping, and the
// table's own tile grid is 8-aligned so a screen tile maps onto exactly one
// block (never straddles two).
uint stbnBlockBase(uint2 pixel, uint2 dims, uint slice)
{
    uint x = pixel.x % dims.x;
    uint y = pixel.y % dims.y;
    uint blocks_x = dims.x / STBN_TILE_WIDTH;
    uint blocks_y = dims.y / STBN_TILE_HEIGHT;
    uint bx = x / STBN_TILE_WIDTH;
    uint by = y / STBN_TILE_HEIGHT;
    return ((slice * blocks_y + by) * blocks_x + bx) * STBN_TILE_AREA;
}

// Pixel-in-tile index of a screen pixel, in the rasterizer's convention
// (p = row * TILE_WIDTH + col). The table's last axis uses the same order, so
// this indexes the block directly.
uint stbnPixelInBlock(uint2 pixel)
{
    return (pixel.y % STBN_TILE_HEIGHT) * STBN_TILE_WIDTH + (pixel.x % STBN_TILE_WIDTH);
}

// Raw packed word for a pixel.
uint stbnWord(StructuredBuffer<uint> table, uint2 pixel, uint2 dims, uint slice)
{
    return table[stbnBlockBase(pixel, dims, slice) + stbnPixelInBlock(pixel)];
}

// This pixel's slot in its tile's selected list, or STBN_SLOT_NONE.
uint stbnSlot(StructuredBuffer<uint> table, uint2 pixel, uint2 dims, uint slice)
{
    return stbnWord(table, pixel, dims, slice) & STBN_SLOT_MASK;
}

// Does this pixel feed the gradient on this slice? True everywhere when the
// mask is off (and then never touches the table).
bool stbnSelected(StructuredBuffer<uint> table, uint2 pixel, uint2 dims,
                  uint slice, uint budget)
{
    if (!stbnActive(budget))
        return true;
    return stbnSlot(table, pixel, dims, slice) != STBN_SLOT_NONE;
}

// How many of this tile's pixels are selected on this slice. Under a global
// threshold this is NOT the nominal budget — it varies tile to tile (measured
// 11..21 at a nominal 16 on the NVIDIA asset) — so the backward's compacted loop
// must take its bound from here rather than from the uniform.
uint stbnTileCount(StructuredBuffer<uint> table, uint block_base)
{
    return (table[block_base] >> STBN_COUNT_SHIFT) & STBN_COUNT_MASK;
}

// The `slot`-th selected pixel of a block, as a pixel-in-tile index. Slots
// [0, stbnTileCount) are exactly the selected pixels.
uint stbnPixelAtSlot(StructuredBuffer<uint> table, uint block_base, uint slot)
{
    return (table[block_base + slot] >> STBN_COMPACT_SHIFT) & STBN_COMPACT_MASK;
}

#endif // TRAINER_COMMON_STBN_SLANG
