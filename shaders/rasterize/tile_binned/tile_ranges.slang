// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Post-sort per-tile [tile_start, tile_end) ranges in the sorted pair array; empty tiles have
// start == end, so the caller must clear both before identifyTileRanges.

[shader("compute")]
[numthreads(256, 1, 1)]
void clearTileRanges(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<uint> tile_start,
    RWStructuredBuffer<uint> tile_end,
    uniform uint num_tiles)
{
    uint i = dtid.x;
    if (i >= num_tiles) return;
    tile_start[i] = 0;
    tile_end[i] = 0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void identifyTileRanges(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> tile_keys,    // sorted
    RWStructuredBuffer<uint> tile_start,
    RWStructuredBuffer<uint> tile_end,
    // n = scan_total[0], read on the GPU to avoid a per-frame readback.
    StructuredBuffer<uint> n_buf,
    uniform uint num_tiles)
{
    uint i = dtid.x;
    uint n = n_buf[0];
    if (i >= n) return;

    uint t = tile_keys[i];
    if (t >= num_tiles) return;

    if (i == 0)
    {
        tile_start[t] = 0;
    }
    else
    {
        uint t_prev = tile_keys[i - 1];
        if (t_prev != t)
        {
            tile_start[t] = i;
            if (t_prev < num_tiles)
                tile_end[t_prev] = i;
        }
    }
    if (i == n - 1)
    {
        tile_end[t] = n;
    }
}
