// FastGS importance scoring rasterizer.
//
// Tile-binned pass identical to forward.slang in structure, but instead of
// writing color it counts per-gaussian "high-error pixel touches". For each
// gaussian that contributes meaningfully (alpha >= threshold) to a pixel
// marked high-error in the error_mask, atomically increment that gaussian's
// importance counter.
//
// Runs once per scoring view; the accumulated counts across N views form
// the multi-view importance signal that drives split/clone decisions.

#include "../common/types.slang"
#include "../common/math.slang"
#include "../primitives/gaussian/eval.slang"

static const uint TILE_SIZE = 16;
static const float T_EARLY_BREAK = 1e-4;
static const float ALPHA_MAX = 0.99;
static const float ALPHA_MIN_SCORE = 1.0 / 255.0;

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void scoreForward(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<Projected2DGaussian> projected_2d,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,
    StructuredBuffer<CameraData> camera,
    RWTexture2D<uint> error_mask,
    RWStructuredBuffer<uint> importance_score_accum)
{
    uint2 pixel = dtid.xy;
    uint2 frame_dim_px = uint2(camera[0].image_size);
    if (pixel.x >= frame_dim_px.x || pixel.y >= frame_dim_px.y)
        return;

    if (error_mask[pixel] == 0u) return;

    uint2 num_tiles = (frame_dim_px + TILE_SIZE - 1) / TILE_SIZE;
    uint tile_idx = gid.y * num_tiles.x + gid.x;
    uint t_start = tile_start[tile_idx];
    uint t_end   = tile_end[tile_idx];
    uint tile_count = (t_end > t_start) ? (t_end - t_start) : 0;

    float2 pixel_center = float2(pixel) + 0.5;
    float T = 1.0;

    for (uint i = 0; i < tile_count; ++i)
    {
        uint g_idx = sorted_gaussian_ids[t_start + i];
        Projected2DGaussian proj = projected_2d[g_idx];
        Gaussian3D g = gaussians[g_idx];

        float alpha_raw = evalGaussian2DFromCov(pixel_center, proj.mean, proj.cov2d, g.opacity, proj.opacity_compensation);
        float alpha = min(alpha_raw, ALPHA_MAX);

        if (alpha >= ALPHA_MIN_SCORE)
        {
            uint dummy;
            InterlockedAdd(importance_score_accum[g_idx], 1u, dummy);
        }

        T *= (1.0 - alpha);
        if (T < T_EARLY_BREAK) break;
    }
}
