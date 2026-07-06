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
#include "checkpoint.slang"

static const uint TILE_SIZE = 16;

// Break once transmittance drops below this. Not a magnitude guess — the
// canonical 3DGS threshold from the reference implementation.
static const float T_EARLY_BREAK = 1e-4;

// Semantic cap on the per-gaussian alpha. Alpha represents the fraction
// of light absorbed by a gaussian at a pixel; it is a probability in
// [0, 1]. Our analytical alpha = opacity * exp(power) can exceed 1 when
// the projected 2D covariance becomes numerically degenerate (negative
// determinant after projection + low-pass filter), causing runaway
// pixel saturation. Capping at 0.99 matches the reference 3DGS
// implementation and keeps the backward's T_i recovery well-conditioned
// (1 - 0.99 = 0.01 is a safe divisor). This is a semantic bound
// (probability ≤ 1), not a magnitude guess.
static const float ALPHA_MAX = 0.99;

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void forwardRasterize(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<Projected2DGaussian> projected_2d,
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
    RWTexture2D<float> output_transmittance,
    RWTexture2D<uint> gaussian_count_per_pixel)
{
    uint2 pixel = dtid.xy;
    uint2 frame_dim_px = uint2(camera[0].image_size);

    uint2 num_tiles = (frame_dim_px + TILE_SIZE - 1) / TILE_SIZE;
    uint tile_idx = gid.y * num_tiles.x + gid.x;
    uint t_start = tile_start[tile_idx];
    uint t_end   = tile_end[tile_idx];
    uint tile_count = (t_end > t_start) ? (t_end - t_start) : 0;

    // Pixel index within the 16x16 tile (row-major), used to address the
    // per-pixel slot inside each bucket's 256-entry block. Computed even
    // for out-of-frame pixels so the whole workgroup agrees on layout,
    // but out-of-frame threads skip all writes.
    uint pixel_in_tile = gtid.y * TILE_SIZE + gtid.x;
    uint tile_bucket_base = bucket_offsets[tile_idx];

    bool valid_pixel = (pixel.x < frame_dim_px.x && pixel.y < frame_dim_px.y);

    float2 pixel_center = float2(pixel) + 0.5;
    float3 C = float3(0.0, 0.0, 0.0);
    float T = 1.0;
    uint processed = 0;

    for (uint i = 0; i < tile_count; ++i)
    {
        // Checkpoint the state at the START of each bucket (before
        // blending splat i == k*RASTER_BUCKET_SIZE). Bucket 0's checkpoint
        // is the initial (T=1, C=0). We record (T, C_in_front): the
        // transmittance and accumulated color IN FRONT of splat i. The
        // backward derives C_behind = C_total - C_in_front - c_i*alpha_i*T_i.
        if (valid_pixel && (i % RASTER_BUCKET_SIZE) == 0u)
        {
            uint gb = tile_bucket_base + (i / RASTER_BUCKET_SIZE);
            raster_checkpoints[checkpointIndex(gb, pixel_in_tile)] =
                packCheckpoint(T, C);
        }

        uint gaussian_idx = sorted_gaussian_ids[t_start + i];

        Projected2DGaussian proj = projected_2d[gaussian_idx];
        Gaussian3D g = gaussians[gaussian_idx];

        float alpha_raw = evalGaussian2DFromCov(pixel_center, proj.mean, proj.cov2d, g.opacity);
        float alpha = min(alpha_raw, ALPHA_MAX);
        float3 color_rgb = shDC(g.sh0);

        C += color_rgb * alpha * T;
        T *= (1.0 - alpha);
        ++processed;

        if (T < T_EARLY_BREAK)
            break;
    }

    if (!valid_pixel)
        return;

    output_image[pixel] = float4(C, 1.0 - T);
    output_transmittance[pixel] = T;
    gaussian_count_per_pixel[pixel] = processed;
}
