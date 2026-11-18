// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_CAMERA_SLANG
#define TRAINER_CAMERA_SLANG

struct CameraData
{
    float4x4 view_matrix;   // world -> view
    float4x4 proj_matrix;   // view  -> clip
    float2 image_size;      // width, height in pixels
    float natural_vignetting;
    float padding;

    // Ideal cos^4 falloff in gamma-2.2 RGB. For learned per-camera vignetting see Deutsch et al., PPISP (2026), sec. 4.2, https://arxiv.org/abs/2601.18336.
    float lensGain(uint2 pixel)
    {
        if (natural_vignetting == 0.0) return 1.0;
        float2 center = (1.0 + float2(proj_matrix[0][2], proj_matrix[1][2])) * (0.5 * image_size);
        float2 inv_focal = 2.0 / (float2(proj_matrix[0][0], proj_matrix[1][1]) * image_size);
        float2 ray = (float2(pixel) + 0.5 - center) * inv_focal;
        return pow(1.0 + dot(ray, ray), -2.0 / 2.2);
    }
};

#endif // TRAINER_CAMERA_SLANG
