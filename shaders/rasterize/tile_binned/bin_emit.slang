// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Binning phase 2, one thread per (tile, splat) pair: pair k belongs to the depth rank whose scanned
// offset range holds k, so the pairs land front to back and a stable sort on tile_id alone then
// leaves every tile's list front-to-back. Per-pair threads keep the stores coalesced and the load
// balanced however many tiles a splat covers.

#include "../../common/types.slang"
#include "splat_tiles.slang"

#include "common/execution.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void binEmit(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<BlendData> blend_data,
    StructuredBuffer<CameraData> camera,
    StructuredBuffer<uint> depth_sorted_ids,
    StructuredBuffer<uint> depth_count,
    StructuredBuffer<uint> pair_offset_by_depth,   // exclusive scan of tile counts in depth order
    StructuredBuffer<uint> pair_total,             // [1] the scan's total
    RWStructuredBuffer<uint> tile_keys,
    RWStructuredBuffer<uint> values,
    // Sort-pair capacity; pairs past it are dropped (the host also clamps the sort count).
    uniform uint max_sort_pairs)
{
    uint pair = linearThreadIndex256(dtid);
    if (pair >= min(pair_total[0], max_sort_pairs)) return;

    // Last rank whose offset is <= pair; offset[0] == 0, and empty ranks never own a pair.
    uint lo = 1u, hi = depth_count[0];
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2u;
        if (pair_offset_by_depth[mid] <= pair)
            lo = mid + 1u;
        else
            hi = mid;
    }
    uint rank = lo - 1u;
    uint gid = depth_sorted_ids[rank];

    SplatTiles tiles;
    SplatTiles.make(blend_data[gid], camera[0].image_size, tiles);
    uint2 tile;
    tiles.walk(pair - pair_offset_by_depth[rank], tile);

    uint num_tiles_x = (uint(camera[0].image_size.x) + TILE_WIDTH - 1u) / TILE_WIDTH;
    tile_keys[pair] = tile.y * num_tiles_x + tile.x;
    values[pair] = gid;
}
