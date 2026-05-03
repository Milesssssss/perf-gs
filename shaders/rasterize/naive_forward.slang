// Naive one-thread-per-pixel forward rasterizer used for M1 + M2.
//
// For each pixel: loop over every active gaussian, project it, evaluate its
// alpha at the pixel center, and accumulate front-to-back with
// transmittance. No tiling, no sorting. Correct only when gaussians are
// single or non-overlapping — M3 replaces this with the tile-binned sorted
// pipeline.

#include "../common/types.slang"
#include "../common/math.slang"
#include "../primitives/gaussian/project.slang"
#include "../primitives/gaussian/eval.slang"

[shader("compute")]
[numthreads(16, 16, 1)]
void naiveForward(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    RWTexture2D<float4> output_image)
{
    uint2 pixel = dtid.xy;
    uint2 frame_dim_px = uint2(camera[0].image_size);
    if (pixel.x >= frame_dim_px.x || pixel.y >= frame_dim_px.y)
        return;

    float2 pixel_center = float2(pixel) + 0.5;
    float3 C = float3(0.0, 0.0, 0.0);
    float T = 1.0;

    uint num_gaussians = num_gaussians_counter[0];

    for (uint g = 0; g < num_gaussians; ++g)
    {
        Gaussian3D gauss = gaussians[g];

        float2 mean_2d;
        float3 cov2d;
        float depth_vs;
        projectGaussian3DTo2D(
            gauss.mean, gauss.log_scale, gauss.rot_quat,
            camera[0].view_matrix, camera[0].proj_matrix, camera[0].image_size,
            mean_2d, cov2d, depth_vs);

        // Behind-camera cull. Not a defensive clamp — points with
        // non-positive view-space depth do not project to meaningful
        // screen coordinates and must be dropped.
        if (depth_vs <= 0.0)
            continue;

        float alpha = evalGaussian2DFromCov(pixel_center, mean_2d, cov2d, gauss.opacity);
        float3 color_rgb = sigmoid3(gauss.color_logit);

        C += color_rgb * alpha * T;
        T *= (1.0 - alpha);
    }

    output_image[pixel] = float4(C, 1.0);
}
