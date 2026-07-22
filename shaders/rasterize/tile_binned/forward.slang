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
#include "../../core/compositor.slang"
#include "checkpoint.slang"

static const uint TILE_SIZE = 16;

// Break once transmittance drops below this. Not a magnitude guess - the
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

// Alpha-min contribution threshold (canonical 3DGS / LiteGS 1/255). A splat
// whose per-pixel alpha is below this is skipped: it contributes < 1 LSB of
// 8-bit color and a negligible transmittance step, so blending it is wasted
// work and inflates the per-pixel iteration depth. MUST match the backward's
// threshold exactly (backward_persplat.slang) so both agree on which splats
// contribute — otherwise the T reconstruction desyncs and gradients corrupt.
static const float ALPHA_MIN = 1.0 / 255.0;

// Generic tile blend: front-to-back composite of a tile's depth-sorted
// primitives, checkpointing (T, C_in_front) at each RASTER_BUCKET_SIZE boundary
// and breaking once transmittance drops below T_EARLY_BREAK. Generic over the
// IPrimitiveSource, so every primitive reuses this exact loop, compositing, and
// checkpoint bookkeeping — only primitiveAt() (the per-primitive gather) varies.
// The loop runs for invalid (out-of-frame) pixels too, exactly as before; only
// the checkpoint write and the caller's output write are gated on valid_pixel.
void blendForward<S : IPrimitiveSource>(
    S src,
    uint t_start,
    uint tile_count,
    no_diff float2 pixel_center,
    bool valid_pixel,
    uint pixel_in_tile,
    uint tile_bucket_base,
    StructuredBuffer<uint> sorted_primitive_ids,
    RWStructuredBuffer<uint2> raster_checkpoints,
    float extinction_hardness,
    out float3 C,
    out float T,
    out uint processed)
{
    C = float3(0.0, 0.0, 0.0);
    T = 1.0;
    processed = 0;

    // Compositing / extinction operator. hardness is a uniform: 1 = alpha-over,
    // k > 1 sharpens extinction / occlusion culling (see core/compositor.slang).
    Compositor comp;
    comp.hardness = extinction_hardness;

    for (uint local_index = 0; local_index < tile_count; ++local_index)
    {
        // Checkpoint the state at the START of each bucket (before
        // blending primitive i == k*RASTER_BUCKET_SIZE). Bucket 0's checkpoint
        // is the initial (T=1, C=0). We record (T, C_in_front): the
        // transmittance and accumulated color IN FRONT of primitive i. The
        // backward derives C_behind = C_total - C_in_front - c_i*alpha_i*T_i.
        if (valid_pixel && (local_index % RASTER_BUCKET_SIZE) == 0u)
        {
            uint gb = tile_bucket_base + (local_index / RASTER_BUCKET_SIZE);
            raster_checkpoints[checkpointIndex(gb, pixel_in_tile)] = packCheckpoint(T, C);
        }

        uint primitive_idx = sorted_primitive_ids[t_start + local_index];

        // Primitive seam: the source owns the per-primitive gather + record
        // build; the loop owns compositing. Same loads + math, any primitive.
        let prim = src.primitiveAt(primitive_idx);

        float alpha_raw = prim.coverage(pixel_center);
        float alpha = min(alpha_raw, ALPHA_MAX);

        if (alpha < ALPHA_MIN)
            continue;

        float3 color_rgb = prim.color();

        C += color_rgb * comp.weight(alpha) * T;
        T *= comp.transmittance(alpha);
        processed = local_index + 1u;   // absolute stop-index of last contributor

        if (T < T_EARLY_BREAK)
            break;
    }
}

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
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
