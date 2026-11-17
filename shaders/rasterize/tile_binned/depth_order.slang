// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Each splat's tile count at its front-to-back rank, so the exclusive scan yields bin_emit's
// depth-ordered write offsets.

[shader("compute")]
[numthreads(256, 1, 1)]
void gatherTileCountsByDepth(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> depth_sorted_ids,
    StructuredBuffer<uint> depth_count,
    StructuredBuffer<uint> tile_count_per_gaussian,
    RWStructuredBuffer<uint> tile_count_by_depth,
    uniform uint max_splats)
{
    uint rank = dtid.x;
    if (rank >= max_splats) return;
    // The scan covers max_splats entries, so ranks past the sorted count must read zero.
    tile_count_by_depth[rank] = (rank < depth_count[0])
        ? tile_count_per_gaussian[depth_sorted_ids[rank]] : 0u;
}
