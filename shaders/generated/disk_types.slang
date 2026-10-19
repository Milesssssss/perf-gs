// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// DO NOT EDIT. Generated from src/trainer/contracts.py.
// Regenerate with:  python -m trainer.contracts
// Validated not-stale by tests/test_contracts.py.

#ifndef TRAINER_DISK_TYPES
#define TRAINER_DISK_TYPES

struct Disk2D
{
    float2 center;  // screen-space center in pixels
    float log_radius;  // logarithm of disk radius in pixels
    float opacity;  // pre-sigmoid opacity logit
    float3 color_value;  // view-independent RGB color
};

namespace DiskChannel
{
    static const uint Center = 0u;  // 2 scalar(s)
    static const uint LogRadius = 2u;  // 1 scalar(s)
    static const uint Opacity = 3u;  // 1 scalar(s)
    static const uint Color = 4u;  // 3 scalar(s)
    static const int Count = 7;
}

#endif // TRAINER_DISK_TYPES
