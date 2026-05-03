/*$(ShaderResources)*/

static float2 frame_dim = /*$(Variable:frameDim)*/;
#define TILE_SIZE 16

#define MAX_GAUSSIANS_PER_TILE 1024

/*$(_compute:clearTileBuffers)*/(uint3 DTid : SV_DispatchThreadID)
{
    uint tileIdx = DTid.x;
    
    uint2 numTiles = ((uint2)frame_dim + TILE_SIZE - 1) / TILE_SIZE;
    uint totalTiles = numTiles.x * numTiles.y;
    
    if (tileIdx >= totalTiles)
        return;

    screen_bin_tiles_count[tileIdx] = 0;
    
    uint baseIdx = tileIdx * MAX_GAUSSIANS_PER_TILE;
    for (uint i = 0; i < MAX_GAUSSIANS_PER_TILE; i++)
    {
        screen_bin_tiles_gaussian_ids[baseIdx + i] = 0xFFFFFFFF;
    }
}