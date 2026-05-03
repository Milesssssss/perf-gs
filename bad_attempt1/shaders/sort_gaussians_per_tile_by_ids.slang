// Soft Rasterizer technique, shader sort_gaussians_per_tile
/*$(ShaderResources)*/

#define TILE_SIZE 16
#define MAX_GAUSSIANS_PER_TILE 1024

static float2 frameDim = /*$(Variable:frameDim)*/;

groupshared uint shared_ids[MAX_GAUSSIANS_PER_TILE];
groupshared float shared_depths[MAX_GAUSSIANS_PER_TILE];

/*$(_compute:sortTiles)*/(uint3 GTid : SV_GroupThreadID, uint3 Gid : SV_GroupID)
{
    uint2 numTiles = ((uint2)frameDim + TILE_SIZE - 1) / TILE_SIZE;

    uint tileIdx = Gid.x;
    uint tid = GTid.x;

    bool validTile = (tileIdx < numTiles.x * numTiles.y);

    uint gaussian_count = 0;
    uint start_idx = 0;

    if (validTile)
    {
        gaussian_count = min(screen_tiles_count[tileIdx], MAX_GAUSSIANS_PER_TILE);
        start_idx = tileIdx * MAX_GAUSSIANS_PER_TILE;
    }

    // Load into shared memory
    if (validTile && tid < gaussian_count)
    {
        uint gid = screen_tiles_indices[start_idx + tid];
        shared_ids[tid] = gid;
        shared_depths[tid] = projected_gaussians_2d[gid].depth_vs;
    }
    else
    {
        // Invalid entries get LARGE depth so they sort to the END (ascending sort)
        shared_ids[tid] = 0xFFFFFFFF;
        shared_depths[tid] = 3.402823466e+38; // FLT_MAX
    }

    GroupMemoryBarrierWithGroupSync();

    // Bitonic sort front-to-back: smaller depth first
    for (uint k = 2; k <= MAX_GAUSSIANS_PER_TILE; k <<= 1)
    {
        for (uint j = k >> 1; j > 0; j >>= 1)
        {
            uint ixj = tid ^ j;

            if (ixj > tid)
            {
                float depth_a = shared_depths[tid];
                float depth_b = shared_depths[ixj];

                bool ascending = ((tid & k) == 0);
                
                bool shouldSwap = ascending ? (depth_a > depth_b) : (depth_a < depth_b);

                if (shouldSwap)
                {
                    // Swap both ID and depth
                    uint temp_id = shared_ids[tid];
                    float temp_depth = shared_depths[tid];
                    
                    shared_ids[tid] = shared_ids[ixj];
                    shared_depths[tid] = shared_depths[ixj];
                    
                    shared_ids[ixj] = temp_id;
                    shared_depths[ixj] = temp_depth;
                }
            }

            GroupMemoryBarrierWithGroupSync();
        }
    }

    // Write back only valid entries
    if (validTile && tid < gaussian_count)
    {
        screen_tiles_indices[start_idx + tid] = shared_ids[tid];
    }
}