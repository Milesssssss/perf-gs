// Live-bucket compaction for the flat (warp-per-bucket) backward.
//
// The per-tile backward (backwardRasterizePerSplat) launches one 256-thread
// workgroup per tile and, inside, skips buckets that no pixel reached (via a
// workgroup max reduction of gaussian_count_per_pixel). The flat backward
// (backwardRasterizePerBucket) launches one WARP per bucket over a flat grid,
// so it cannot do that per-tile skip from inside the kernel. Instead we
// PRE-compute, after the forward, a compact list of only the LIVE buckets
// (those some pixel actually reached) and dispatch exactly those.
//
// Chain (host order, see pipeline.py):
//   tileMaxProcessed          : live_bucket_count[t] = ceil(max_processed_t / P)
//   _scan(live_bucket_count)   : live_bucket_offsets[t] (exclusive) + live_bucket_total
//   scatterLiveBuckets         : live_buckets[off+k] = uint2(tile, local_bucket)
//   prepareBucketDispatchArgs  : groups = ceil(live_bucket_total / WARPS_PER_WG)
//
// live_buckets carries (tile, local_bucket) directly, so the backward needs no
// binary search AND never launches a saturated (empty) bucket.

#include "checkpoint.slang"   // RASTER_BUCKET_SIZE
#include "../../common/stbn.slang"

#include "generated/execution_contract.slang"
// Non-square tiles: WIDTH across, HEIGHT down; pixel-in-tile row stride = WIDTH.
static const uint TILE_WIDTH  = CONTRACT_TILE_WIDTH;
static const uint TILE_HEIGHT = CONTRACT_TILE_HEIGHT;
static const uint TILE_AREA   = CONTRACT_TILE_AREA;
// Warps (== buckets) per flat-backward workgroup = TILE_AREA/32 (one 256-thread
// WG at 16x16 -> 8 warps; 8x8 -> 2; 8x4 (area 32) -> 1 warp). MUST match the
// per-bucket kernel's numthreads(TILE_AREA) and its slot = gid.x*NUM_WARPS+warp
// mapping, or the indirect dispatch under/over-covers the live-bucket list.
static const uint FLAT_WARPS_PER_WG = TILE_AREA / CONTRACT_RASTER_BUCKET_SIZE;

groupshared uint gs_tile_max;

// Pass 1: per tile, reduce the max processed-depth over its TILE_AREA pixels and
// turn it into a LIVE bucket count (ceil(max / P)). One TILE_AREA-thread WG/tile
// (tid indexes pixel-in-tile via tid%TILE_WIDTH / tid/TILE_WIDTH, so the launch
// MUST be TILE_AREA threads — host dispatches num_tiles*TILE_AREA to match).
[shader("compute")]
[numthreads(TILE_AREA, 1, 1)]
void tileMaxProcessed(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    RWTexture2D<uint> gaussian_count_per_pixel,
    RWStructuredBuffer<uint> live_bucket_count,
    // STBN sparse supervision (runtime; see common/stbn.slang). Reducing over
    // only the SAMPLED pixels is not just an optimization: at a low budget a
    // tile's deepest ray is usually an unsampled pixel, so counting it would
    // queue buckets whose every contribution is zero. Runtime-gated (not behind
    // STBN_ENABLED) so the compile-time-disabled backward still sees the same
    // live-bucket set — that equivalence is what makes STBN_ENABLED=0 a usable
    // reference path.
    StructuredBuffer<uint> stbn_table,
    uniform uint2 stbn_dims,
    uniform uint stbn_slice,
    uniform uint stbn_tile_pixels,
    uniform uint2 frame_dim_px,
    uniform uint num_tiles_x,
    uniform uint num_tiles)
{
    uint tile_idx = gid.x;
    if (tile_idx >= num_tiles) return;

    uint tid = gtid.x;
    if (tid == 0u) gs_tile_max = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint tile_x = tile_idx % num_tiles_x;
    uint tile_y = tile_idx / num_tiles_x;
    // pixel-in-tile: col = tid % WIDTH (row stride WIDTH), row = tid / WIDTH.
    uint2 pixel = uint2(tile_x * TILE_WIDTH  + (tid % TILE_WIDTH),
                        tile_y * TILE_HEIGHT + (tid / TILE_WIDTH));
    bool contributes = pixel.x < frame_dim_px.x && pixel.y < frame_dim_px.y
                       && stbnSelected(stbn_table, pixel, stbn_dims, stbn_slice,
                                       stbn_tile_pixels);
    uint processed = contributes ? gaussian_count_per_pixel[pixel] : 0u;
    InterlockedMax(gs_tile_max, processed);
    GroupMemoryBarrierWithGroupSync();

    if (tid == 0u)
    {
        uint m = gs_tile_max;
        live_bucket_count[tile_idx] = (m + RASTER_BUCKET_SIZE - 1u) / RASTER_BUCKET_SIZE;
    }
}

// Pass 3 (after the exclusive scan of live_bucket_count): expand each tile's
// live bucket range into the compact list. One thread per tile writes its
// `count` entries; count is small (a few buckets) for almost all tiles.
[shader("compute")]
[numthreads(256, 1, 1)]
void scatterLiveBuckets(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> live_bucket_count,
    StructuredBuffer<uint> live_bucket_offsets,
    RWStructuredBuffer<uint2> live_buckets,
    uniform uint num_tiles)
{
    uint t = dtid.x;
    if (t >= num_tiles) return;
    uint cnt = live_bucket_count[t];
    uint off = live_bucket_offsets[t];
    for (uint k = 0u; k < cnt; ++k)
        live_buckets[off + k] = uint2(t, k);
}

// Pass 4: single-thread indirect-dispatch args for the flat backward.
// groups_x = ceil(live_bucket_total / FLAT_WARPS_PER_WG); each WG owns 8 buckets.
[shader("compute")]
[numthreads(1, 1, 1)]
void prepareBucketDispatchArgs(
    RWStructuredBuffer<uint> live_bucket_total,
    RWStructuredBuffer<uint> dispatch_args,
    uniform uint max_buckets)   // defensive cap: never queue more than we sized
{
    uint total = min(live_bucket_total[0], max_buckets);
    uint groups = (total + FLAT_WARPS_PER_WG - 1u) / FLAT_WARPS_PER_WG;
    // Vulkan/D3D12 require at least (1,1,1); the in-kernel slot >= total guard
    // makes the extra group a no-op.
    if (groups == 0u) groups = 1u;
    dispatch_args[0] = groups;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}
