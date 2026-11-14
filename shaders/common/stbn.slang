// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Tile-stratified spatiotemporal blue-noise (STBN) pixel selection.

#ifndef TRAINER_COMMON_STBN_SLANG
#define TRAINER_COMMON_STBN_SLANG

#include "common/execution.slang"

#ifndef STBN_ENABLED
#define STBN_ENABLED 0
#endif

#ifndef STBN_MAX_TILE_PIXELS
#define STBN_MAX_TILE_PIXELS TILE_AREA
#endif

static const uint STBN_TILE_WIDTH  = TILE_WIDTH;
static const uint STBN_TILE_HEIGHT = TILE_HEIGHT;
static const uint STBN_TILE_AREA   = TILE_AREA;

// Packed table layout, one uint32 per voxel (packed by stbn_table.pack_selection):
//   bits  0..7   slot of this pixel in its tile's selected list, 0xFF = none
//   bits  8..15  pixel-in-tile index of the slot-i selected pixel (i < count)
//   bits 16..23  selected pixels in this tile, replicated across the block
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

// Unbiasing weight TILE_AREA / k; exactly 1.0 when the mask is off, so that path stays bit-identical.
float stbnWeight(uint budget)
{
    return stbnActive(budget) ? (float(STBN_TILE_AREA) / float(budget)) : 1.0;
}

// `dims` is the table size in pixels, wrapped over the screen; it is 8-aligned, so a screen
// tile maps onto exactly one block.
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

// Rasterizer pixel-in-tile order (row * TILE_WIDTH + col), shared by the table's last axis.
uint stbnPixelInBlock(uint2 pixel)
{
    return (pixel.y % STBN_TILE_HEIGHT) * STBN_TILE_WIDTH + (pixel.x % STBN_TILE_WIDTH);
}

uint stbnWord(StructuredBuffer<uint> table, uint2 pixel, uint2 dims, uint slice)
{
    return table[stbnBlockBase(pixel, dims, slice) + stbnPixelInBlock(pixel)];
}

// This pixel's slot in its tile's selected list, or STBN_SLOT_NONE.
uint stbnSlot(StructuredBuffer<uint> table, uint2 pixel, uint2 dims, uint slice)
{
    return stbnWord(table, pixel, dims, slice) & STBN_SLOT_MASK;
}

// True everywhere when the mask is off, without reading the table.
bool stbnSelected(StructuredBuffer<uint> table, uint2 pixel, uint2 dims,
                  uint slice, uint budget)
{
    if (!stbnActive(budget))
        return true;
    return stbnSlot(table, pixel, dims, slice) != STBN_SLOT_NONE;
}

// Varies per tile under a global threshold, so compacted loops must bound on this, not the budget.
uint stbnTileCount(StructuredBuffer<uint> table, uint block_base)
{
    return (table[block_base] >> STBN_COUNT_SHIFT) & STBN_COUNT_MASK;
}

// Pixel-in-tile index of the slot-th selected pixel; slots [0, stbnTileCount) are valid.
uint stbnPixelAtSlot(StructuredBuffer<uint> table, uint block_base, uint slot)
{
    return (table[block_base + slot] >> STBN_COMPACT_SHIFT) & STBN_COMPACT_MASK;
}

#endif // TRAINER_COMMON_STBN_SLANG
