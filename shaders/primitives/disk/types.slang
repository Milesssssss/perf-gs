// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_DISK_TYPES_SLANG
#define TRAINER_DISK_TYPES_SLANG

struct Disk2D
{
    float2 center;       // screen-space pixels
    float log_radius;    // log of the radius in pixels
    float opacity;       // pre-sigmoid logit
    float3 color_value;
};

namespace DiskChannel
{
    static const uint Center = 0u;
    static const uint LogRadius = 2u;
    static const uint Opacity = 3u;
    static const uint Color = 4u;
    static const int Count = 7;
}

#endif // TRAINER_DISK_TYPES_SLANG
