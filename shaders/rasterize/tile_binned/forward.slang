// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Tile-binned forward rasterizer: one thread per pixel, one workgroup per tile, front-to-back.
// Writes each pixel's early-break stop index so the backward replays the same entries.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/eval.slang"
#include "../../primitives/gaussian/blend.slang"
#include "blend_forward.slang"
#include "common/execution.slang"

[shader("compute")]
[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void forwardRasterize(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    // Compact blend record packed by bin_count, gathered identically by the backward.
    StructuredBuffer<BlendData> blend_data,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,   // the value array after sort
    StructuredBuffer<CameraData> camera,
    // Ragged (CSR) (T, C_in_front) checkpoints, one per RASTER_BUCKET_SIZE splats, for the per-splat backward.
    StructuredBuffer<uint> bucket_offsets,
    RWStructuredBuffer<uint2> raster_checkpoints,
    RWTexture2D<float4> output_image,
    RWTexture2D<uint> gaussian_count_per_pixel,
    uniform float extinction_hardness)
{
    uint2 pixel = dtid.xy;
    uint2 frame_dim_px = uint2(camera[0].image_size);

    uint2 num_tiles = (frame_dim_px + uint2(TILE_WIDTH, TILE_HEIGHT) - 1u)
                      / uint2(TILE_WIDTH, TILE_HEIGHT);
    uint tile_idx = gid.y * num_tiles.x + gid.x;
    uint t_start = tile_start[tile_idx];
    uint t_end   = tile_end[tile_idx];
    uint tile_count = (t_end > t_start) ? (t_end - t_start) : 0;

    // Row-major (stride TILE_WIDTH) slot in each bucket's TILE_AREA-entry checkpoint block.
    uint pixel_in_tile = gtid.y * TILE_WIDTH + gtid.x;
    uint tile_bucket_base = bucket_offsets[tile_idx];

    bool valid_pixel = (pixel.x < frame_dim_px.x && pixel.y < frame_dim_px.y);

    float2 pixel_center = float2(pixel) + 0.5;

    GaussianSource src;
    src.blend_data = blend_data;

    float3 C;
    float T;
    uint processed;
    blendForward<GaussianSource>(
        src, t_start, tile_count, pixel_center, valid_pixel,
        pixel_in_tile, tile_bucket_base, sorted_gaussian_ids,
        raster_checkpoints, extinction_hardness, C, T, processed);

    if (!valid_pixel)
        return;

    // The backward recovers final transmittance as T = 1 - output_image.a.
    output_image[pixel] = float4(C, 1.0 - T);
    gaussian_count_per_pixel[pixel] = processed;
}
