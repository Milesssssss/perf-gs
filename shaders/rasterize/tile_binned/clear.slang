// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Per-frame clears of the tile-binned forward's output textures.

[shader("compute")]
[numthreads(16, 16, 1)]
void clearTextures(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,
    RWTexture2D<uint> gaussian_count_per_pixel,
    uniform uint2 frame_dim)
{
    uint2 p = dtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;
    // alpha = 0 is T = 1: the backward reads T = 1 - output_image.a.
    output_image[p] = float4(0.0, 0.0, 0.0, 0.0);
    gaussian_count_per_pixel[p] = 0;
}
