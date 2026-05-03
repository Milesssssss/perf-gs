// Per-frame clear kernels. Each resets one of the scratch/output
// resources used by the tile-binned forward pipeline.

[shader("compute")]
[numthreads(64, 1, 1)]
void clearTilesCount(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<uint> screen_tiles_count,
    uniform uint num_tiles)
{
    uint i = dtid.x;
    if (i >= num_tiles) return;
    screen_tiles_count[i] = 0;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void clearTextures(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,
    RWTexture2D<float> output_transmittance,
    RWTexture2D<uint> gaussian_count_per_pixel,
    uniform uint2 frame_dim)
{
    uint2 p = dtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;
    output_image[p] = float4(0.0, 0.0, 0.0, 1.0);
    output_transmittance[p] = 1.0;
    gaussian_count_per_pixel[p] = 0;
}
