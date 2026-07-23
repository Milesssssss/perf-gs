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

#include "generated/execution_contract.slang"
static const uint TILE_SIZE = CONTRACT_TILE_SIZE;
static const uint TILE_AREA = TILE_SIZE * TILE_SIZE;   // 256
static const uint FLAT_WARPS_PER_WG = 8u;              // 256 threads / 32 lanes

groupshared uint gs_tile_max;

// Pass 1: per tile, reduce the max processed-depth over its 256 pixels and
// turn it into a LIVE bucket count (ceil(max / P)). One 256-thread WG/tile.
[shader("compute")]
[numthreads(256, 1, 1)]
void tileMaxProcessed(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    RWTexture2D<uint> gaussian_count_per_pixel,
    RWStructuredBuffer<uint> live_bucket_count,
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
    uint2 pixel = uint2(tile_x * TILE_SIZE + (tid % TILE_SIZE),
                        tile_y * TILE_SIZE + (tid / TILE_SIZE));
    uint processed = (pixel.x < frame_dim_px.x && pixel.y < frame_dim_px.y)
                     ? gaussian_count_per_pixel[pixel] : 0u;
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
