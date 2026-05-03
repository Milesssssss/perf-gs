// Per-tile depth sort.
//
// One workgroup per tile. Each workgroup loads the tile's gaussian indices
// and depths into LDS, sorts by ascending depth_vs (closest first for the
// front-to-back rasterizer), and writes sorted indices back.
//
// This is an LDS insertion sort — simple, stable, O(n^2) worst case. For
// MAX_GAUSSIANS_PER_TILE=1024 that's 1M comparisons per full tile, fine
// for v0. A proper global radix sort replaces this in v1 (see the
// radix_sort/ placeholder dir).

#include "../../common/types.slang"

static const uint TILE_SIZE = 16;
static const uint MAX_GAUSSIANS_PER_TILE = 1024;

groupshared uint gs_indices[MAX_GAUSSIANS_PER_TILE];
groupshared float gs_depths[MAX_GAUSSIANS_PER_TILE];
groupshared uint gs_count;

[shader("compute")]
[numthreads(256, 1, 1)]
void sortTile(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint lid: SV_GroupIndex,
    StructuredBuffer<Projected2DGaussian> projected_2d,
    RWStructuredBuffer<uint> screen_tiles_count,
    RWStructuredBuffer<uint> screen_tiles_indices)
{
    uint tile_idx = gid.x;

    if (lid == 0)
    {
        gs_count = min(screen_tiles_count[tile_idx], MAX_GAUSSIANS_PER_TILE);
        // Also clamp the stored count so later passes agree with what got
        // sorted. Any overflow drops are now invisible downstream.
        screen_tiles_count[tile_idx] = gs_count;
    }
    GroupMemoryBarrierWithGroupSync();

    uint count = gs_count;
    uint base = tile_idx * MAX_GAUSSIANS_PER_TILE;

    // Cooperatively load indices + depths into LDS.
    for (uint i = lid; i < count; i += 256)
    {
        uint gaussian_idx = screen_tiles_indices[base + i];
        gs_indices[i] = gaussian_idx;
        gs_depths[i] = projected_2d[gaussian_idx].depth_vs;
    }
    GroupMemoryBarrierWithGroupSync();

    // Insertion sort run by a single thread. LDS is groupshared, so the
    // full sort happens here before any other thread writes back.
    if (lid == 0)
    {
        for (uint i = 1; i < count; ++i)
        {
            uint key_idx = gs_indices[i];
            float key_depth = gs_depths[i];
            int j = int(i) - 1;
            while (j >= 0 && gs_depths[j] > key_depth)
            {
                gs_indices[j + 1] = gs_indices[j];
                gs_depths[j + 1]  = gs_depths[j];
                --j;
            }
            gs_indices[j + 1] = key_idx;
            gs_depths[j + 1]  = key_depth;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Cooperatively write sorted indices back to global.
    for (uint i = lid; i < count; i += 256)
    {
        screen_tiles_indices[base + i] = gs_indices[i];
    }
}
