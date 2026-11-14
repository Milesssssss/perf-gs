// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Per-tile bucket count for the ragged (CSR) checkpoint buffer; its exclusive scan gives
// bucket_offsets[t] and the frame's bucket total. Runs after identifyTileRanges.

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
