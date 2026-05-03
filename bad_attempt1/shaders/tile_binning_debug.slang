// Binning Debug Visualization
// Shows heatmap of Gaussian count per tile (0-50 range)
// Draws warning indicator at center for overflowing tiles (>MAX_GAUSSIANS_PER_TILE)
/*$(ShaderResources)*/

#define TILE_SIZE 16
#define MAX_GAUSSIANS_PER_TILE 1024
#define HEATMAP_MAX 1024

static float2 frameDim = /*$(Variable:frameDim)*/;

// Heatmap: 0 = blue, 50+ = red
float3 heatmap(float t)
{
    t = saturate(t);  // Clamp to [0, 1]
    
    // Blue -> Cyan -> Green -> Yellow -> Red
    float3 color;
    if (t < 0.25)
    {
        // Blue to Cyan
        float s = t / 0.25;
        color = float3(0.0, s, 1.0);
    }
    else if (t < 0.5)
    {
        // Cyan to Green
        float s = (t - 0.25) / 0.25;
        color = float3(0.0, 1.0, 1.0 - s);
    }
    else if (t < 0.75)
    {
        // Green to Yellow
        float s = (t - 0.5) / 0.25;
        color = float3(s, 1.0, 0.0);
    }
    else
    {
        // Yellow to Red
        float s = (t - 0.75) / 0.25;
        color = float3(1.0, 1.0 - s, 0.0);
    }
    
    return color;
}

/*$(_compute:debugBinning)*/ (
    uint3 DTid : SV_DispatchThreadID
)
{
    uint2 pixel = DTid.xy;
    
    if (pixel.x >= frameDim.x || pixel.y >= frameDim.y)
        return;
    
    uint2 tile_coord = pixel / TILE_SIZE;
    uint2 numTiles = ((uint2)frameDim + TILE_SIZE - 1) / TILE_SIZE;
    uint tileIdx = tile_coord.y * numTiles.x + tile_coord.x;

    uint gaussian_count = screen_tiles_count[tileIdx];
    
    float normalized = float(gaussian_count) / float(HEATMAP_MAX);

    // Apply heatmap
    float3 color = heatmap(normalized);

    // Special case: empty tiles are white
    if (gaussian_count == 0)
        color = float3(1.0, 1.0, 1.0);

    uint2 pixel_in_tile = pixel % TILE_SIZE;
    int2 center_offset = int2(pixel_in_tile) - int2(TILE_SIZE / 2, TILE_SIZE / 2);
    
    bool is_overflowing = gaussian_count > MAX_GAUSSIANS_PER_TILE;
    
    if (is_overflowing)
    {
		//We draw a + sign at the center of the tile to indicate overflow
        int dist = max(abs(center_offset.x), abs(center_offset.y));

        bool is_cross = (abs(center_offset.x) == abs(center_offset.y)) && dist <= 2;
        bool is_square = abs(center_offset.x) <= 1 && abs(center_offset.y) <= 1;
        bool is_plus = (center_offset.x == 0 && abs(center_offset.y) <= 2) || 
                       (center_offset.y == 0 && abs(center_offset.x) <= 2);
        
        if (is_square)
        {
            color = float3(0.0, 0.0, 0.0);  // black for warning indicator
        }
    }

    // Draw tile borders (darken edges)
    if (pixel_in_tile.x == 0 || pixel_in_tile.y == 0)
    {
        color *= 0.5;  // Darken borders
    }

    // Write to output
    tile_binning_debug[pixel] = float4(color, 1.0);
}