// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Naive one-thread-per-pixel forward rasterizer over every gaussian (no tiling, no sorting).
// Correct only when gaussians are single or non-overlapping.

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
        float cov2d_det;
        float comp_unused;
        projectGaussian3DTo2D(
            gauss.mean, gauss.log_scale, gauss.rot_quat,
            camera[0].view_matrix, camera[0].proj_matrix, camera[0].image_size,
            0.0, 0.3, 0u,
            mean_2d, cov2d, depth_vs, cov2d_det, comp_unused);

        // Non-positive view depth does not project to meaningful screen coordinates.
        if (depth_vs <= 0.0)
            continue;

        float alpha = evalGaussian2DFromCov(pixel_center, mean_2d, cov2d, gauss.opacity, 1.0);
        float3 color_rgb = shDC(gauss.sh0);

        C += color_rgb * alpha * T;
        T *= (1.0 - alpha);
    }

    output_image[pixel] = float4(C, 1.0);
}
