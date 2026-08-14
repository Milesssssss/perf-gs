// Tile-stratified spatiotemporal blue-noise (STBN) pixel selection.
//
// OPT-IN sparse supervision: instead of every pixel of the sampled view feeding
// the backward, only k of each tile's TILE_AREA pixels do, and the surviving
// terms are scaled by TILE_AREA/k. The chain rule over pixels,
//     dL/dtheta = sum_q (dL/dx_q) (dx_q/dtheta),
// is a SUM over pixels, so keeping a random subset S and dividing by its
// expected density is an unbiased estimator of the full-image gradient — for
// any image loss, including the windowed SSIM term (we subsample the terms of
// that sum, not the loss itself). See src/trainer/backend/render/stbn.py for
// where the mask comes from (NVIDIA's published scalar STBN, re-ranked per tile)
// and docs/stbn_pixel_sampling.md for the reasoning.
//
// TABLE LAYOUT (built by stbn.py: pack_table)
//   packed[block_base + i] = rank(pixel i) | (pixel of rank i) << 16
//   block_base = ((slice * blocks_y + by) * blocks_x + bx) * TILE_AREA
// Every (slice, tile) block holds a PERMUTATION of 0..TILE_AREA-1, so
// "rank < k" selects exactly k pixels per tile — no per-tile count jitter, and
// the compensation weight TILE_AREA/k is exact rather than an expectation.
//
// The two halves serve the two consumers:
//   * low  16 bits, indexed BY PIXEL -> "is this pixel selected?" (loss kernels)
//   * high 16 bits, indexed BY SLOT  -> "which pixel is the j-th selected one?"
//     so the backward walks a COMPACTED list of k pixels instead of testing all
//     TILE_AREA of them. That compaction is where the speedup comes from; the
//     loop trip count drops with k rather than staying at TILE_AREA.
//
// Selection depends only on (pixel, slice) — both warp-uniform across the lanes
// of a bucket, which all own splats of the SAME tile — so every lane skips the
// same pixels together. No wave divergence, and the wave scans in the backward
// stay in lockstep.

#ifndef TRAINER_COMMON_STBN_SLANG
#define TRAINER_COMMON_STBN_SLANG

#include "generated/execution_contract.slang"

// Compile-time: does the backward take the compacted-visit path at all? 0
// (default) keeps the exact full-pixel loop, bit-for-bit, with no added
// register or LDS cost. The loss kernels need no define — they mask at runtime
// (see stbnSelected), which is what makes STBN_ENABLED=0 a usable REFERENCE
// path: the same sparse gradient, just without the pixel-loop savings.
#ifndef STBN_ENABLED
#define STBN_ENABLED 0
#endif

// Compile-time ceiling on the per-tile budget k. It sizes the backward's staged
// LDS arrays, so a run that never exceeds (say) 16 pixels/tile also gets the
// occupancy win of a 4x smaller checkpoint stage — not just the shorter loop.
// The host validates the runtime budget against it (StbnConfig.max_tile_pixels).
#ifndef STBN_MAX_TILE_PIXELS
#define STBN_MAX_TILE_PIXELS CONTRACT_TILE_AREA
#endif

// Prefixed to avoid colliding with the TILE_* constants the rasterizer shaders
// declare for themselves (textual #include => one file scope).
static const uint STBN_TILE_WIDTH  = CONTRACT_TILE_WIDTH;
static const uint STBN_TILE_HEIGHT = CONTRACT_TILE_HEIGHT;
static const uint STBN_TILE_AREA   = CONTRACT_TILE_AREA;

static const uint STBN_RANK_MASK = 0xFFFFu;
static const uint STBN_INVERSE_SHIFT = 16u;

// Is the mask doing anything? The host passes budget == TILE_AREA (select
// everything) to disable it, and binds a dummy 1-element table in that case, so
// every reader MUST take this early-out before touching the table.
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
    return (pixel.y % STBN_TILE_HEIGHT) * STBN_TILE_WIDTH
           + (pixel.x % STBN_TILE_WIDTH);
}

// Rank of a pixel within its (slice, tile) block, in [0, TILE_AREA).
uint stbnRank(StructuredBuffer<uint> table, uint2 pixel, uint2 dims, uint slice)
{
    return table[stbnBlockBase(pixel, dims, slice) + stbnPixelInBlock(pixel)]
           & STBN_RANK_MASK;
}

// Does this pixel feed the gradient on this slice? True everywhere when the
// mask is off (and then never touches the table).
bool stbnSelected(StructuredBuffer<uint> table, uint2 pixel, uint2 dims,
                  uint slice, uint budget)
{
    if (!stbnActive(budget))
        return true;
    return stbnRank(table, pixel, dims, slice) < budget;
}

// The `slot`-th selected pixel of a block, as a pixel-in-tile index. Slots
// [0, budget) are exactly the selected pixels, in rank order.
uint stbnPixelAtSlot(StructuredBuffer<uint> table, uint block_base, uint slot)
{
    return table[block_base + slot] >> STBN_INVERSE_SHIFT;
}

#endif // TRAINER_COMMON_STBN_SLANG
