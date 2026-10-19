// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_CAMERA_SLANG
#define TRAINER_CAMERA_SLANG

struct CameraData
{
    float4x4 view_matrix;   // world -> view
    float4x4 proj_matrix;   // view  -> clip
    float2 image_size;      // width, height in pixels
};

#endif // TRAINER_CAMERA_SLANG
