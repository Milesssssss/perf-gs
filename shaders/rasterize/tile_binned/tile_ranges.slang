// Post-sort: identify per-tile [start, end) ranges in the sorted pair
// array. One thread per sorted entry; each thread writes tile_start and
// tile_end at the boundaries where the neighbour's tile_id differs.
//
// Convention: tile_start[t] = inclusive first index of tile t in the
// sorted array; tile_end[t] = exclusive last. For tiles that didn't
// receive any pairs, tile_start == tile_end (empty range).
//
// The caller must clear tile_start and tile_end to 0 (or to n, whichever
// represents "empty") before dispatch.

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
    StructuredBuffer<uint> keys_hi,    // sorted
    RWStructuredBuffer<uint> tile_start,
    RWStructuredBuffer<uint> tile_end,
    // ``n`` lives on the GPU (scan_total[0]) — no per-frame readback.
    StructuredBuffer<uint> n_buf,
    uniform uint num_tiles)
{
    uint i = dtid.x;
    uint n = n_buf[0];
    if (i >= n) return;

    uint t = keys_hi[i];
    if (t >= num_tiles) return;  // robustness

    // First entry, or previous entry is a different tile → start of t.
    if (i == 0)
    {
        tile_start[t] = 0;
    }
    else
    {
        uint t_prev = keys_hi[i - 1];
        if (t_prev != t)
        {
            tile_start[t] = i;
            if (t_prev < num_tiles)
                tile_end[t_prev] = i;
        }
    }
    // Last entry of the whole array → end of its tile.
    if (i == n - 1)
    {
        tile_end[t] = n;
    }
}
