// Per-tile bucket count for the ragged (CSR) checkpoint buffer.
//
// bucket_count[t] = ceil( (tile_end[t] - tile_start[t]) / RASTER_BUCKET_SIZE )
//
// An exclusive prefix scan of this array yields bucket_offsets[t] (the base
// bucket index of tile t in the flat checkpoint buffer), and the scan total
// is the number of buckets actually used this frame. Runs after
// identifyTileRanges (needs tile_start/end).

#include "checkpoint.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void bucketCount(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    RWStructuredBuffer<uint> bucket_count,
    uniform uint num_tiles)
{
    uint t = dtid.x;
    if (t >= num_tiles) return;

    uint s = tile_start[t];
    uint e = tile_end[t];
    uint len = (e > s) ? (e - s) : 0u;
    bucket_count[t] = (len + RASTER_BUCKET_SIZE - 1u) / RASTER_BUCKET_SIZE;
}
