// Tile-binned forward rasterizer.
//
// One thread per pixel, one workgroup per tile. Walks the tile's
// depth-sorted gaussian list front-to-back, accumulating color with
// front-to-back transmittance. Breaks early when transmittance drops
// below a small threshold; writes the stop index to
// gaussian_count_per_pixel so v1's backward pass can iterate exactly the
// gaussians that contributed to this pixel (PLAN.md Q11).

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/eval.slang"

static const uint TILE_SIZE = 16;
static const uint MAX_GAUSSIANS_PER_TILE = 1024;

// Break once transmittance drops below this. Not a magnitude guess — the
// canonical 3DGS threshold from the reference implementation. At T < 1e-4
// a gaussian's contribution to the pixel would need alpha near 1 to
// produce a visible color change.
static const float T_EARLY_BREAK = 1e-4;

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void forwardRasterize(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<Projected2DGaussian> projected_2d,
    StructuredBuffer<uint> screen_tiles_count,
    StructuredBuffer<uint> screen_tiles_indices,
    StructuredBuffer<CameraData> camera,
    RWTexture2D<float4> output_image,
    RWTexture2D<float> output_transmittance,
    RWTexture2D<uint> gaussian_count_per_pixel)
{
    uint2 pixel = dtid.xy;
    uint2 frame_dim_px = uint2(camera[0].image_size);
    if (pixel.x >= frame_dim_px.x || pixel.y >= frame_dim_px.y)
        return;

    uint2 num_tiles = (frame_dim_px + TILE_SIZE - 1) / TILE_SIZE;
    uint tile_idx = gid.y * num_tiles.x + gid.x;
    uint base = tile_idx * MAX_GAUSSIANS_PER_TILE;
    uint tile_count = min(screen_tiles_count[tile_idx], MAX_GAUSSIANS_PER_TILE);

    float2 pixel_center = float2(pixel) + 0.5;
    float3 C = float3(0.0, 0.0, 0.0);
    float T = 1.0;
    uint processed = 0;

    for (uint i = 0; i < tile_count; ++i)
    {
        uint gaussian_idx = screen_tiles_indices[base + i];

        Projected2DGaussian proj = projected_2d[gaussian_idx];
        Gaussian3D g = gaussians[gaussian_idx];

        float alpha = evalGaussian2DFromCov(pixel_center, proj.mean, proj.cov2d, g.opacity);
        float3 color_rgb = sigmoid3(g.color_logit);

        C += color_rgb * alpha * T;
        T *= (1.0 - alpha);
        ++processed;

        if (T < T_EARLY_BREAK)
            break;
    }

    output_image[pixel] = float4(C, 1.0);
    output_transmittance[pixel] = T;
    gaussian_count_per_pixel[pixel] = processed;
}
