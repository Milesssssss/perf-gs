// Tile-binned forward rasterizer.
//
// One thread per pixel, one workgroup per tile. Walks the tile's
// depth-sorted gaussian entries front-to-back, accumulating color with
// front-to-back transmittance. Breaks early when transmittance drops
// below a small threshold; writes the stop index (relative to the tile's
// local entry range) to gaussian_count_per_pixel so the backward can
// iterate the same entries in reverse.
//
// Entries come from the globally-sorted (key, value) array produced by
// the radix sort. tile_start[t] / tile_end[t] define [first, last)
// indices of tile t's entries within sorted_gaussian_ids[]. No per-tile
// cap — capacity is bounded only by the sorted-pair buffer allocation.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/eval.slang"
#include "../../primitives/gaussian/blend.slang"
#include "blend_forward.slang"
#include "generated/execution_contract.slang"

// Non-square tiles: TILE_WIDTH pixels across, TILE_HEIGHT down. Pixel-in-tile
// row stride is TILE_WIDTH.
static const uint TILE_WIDTH  = CONTRACT_TILE_WIDTH;
static const uint TILE_HEIGHT = CONTRACT_TILE_HEIGHT;

[shader("compute")]
[numthreads(TILE_WIDTH, TILE_HEIGHT, 1)]
void forwardRasterize(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    // Compact per-splat blend record packed by bin_count (mean, cov2d, opacity,
    // sh0, opacity_compensation) — same repacked fields as the fat gaussians[]
    // + projected_2d[] pair, one 36 B gather instead of 80 B. Matches the
    // backward's gather so both read the primitive identically.
    StructuredBuffer<BlendData> blend_data,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,   // the value array after sort
    StructuredBuffer<CameraData> camera,
    // CSR checkpoint outputs: base bucket index of this tile + the ragged
    // (T, C_in_front) checkpoint buffer written once per RASTER_BUCKET_SIZE
    // splats, consumed by the transposed backward.
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

    // Pixel index within the tile (row-major, row stride = TILE_WIDTH), used to
    // address the per-pixel slot inside each bucket's TILE_AREA-entry block.
    // Computed even for out-of-frame pixels so the whole workgroup agrees on
    // layout, but out-of-frame threads skip all writes.
    uint pixel_in_tile = gtid.y * TILE_WIDTH + gtid.x;
    uint tile_bucket_base = bucket_offsets[tile_idx];

    bool valid_pixel = (pixel.x < frame_dim_px.x && pixel.y < frame_dim_px.y);

    float2 pixel_center = float2(pixel) + 0.5;

    // Generic tile blend over the primitive source. GaussianSource is the only
    // Gaussian-specific glue here; the loop, compositing, checkpointing, and
    // early-out live in blendForward<S> (shared by every primitive). Gathers
    // the compact BlendData bin_count packed.
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

    // alpha = 1 - T. The backward recovers final transmittance as
    // T = 1 - output_image.a (exact in float), so no separate
    // output_transmittance texture is stored.
    output_image[pixel] = float4(C, 1.0 - T);
    gaussian_count_per_pixel[pixel] = processed;
}
