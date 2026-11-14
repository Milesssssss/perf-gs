// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// fp16 pairs packed into uint32 for the SH buffers, so no 16-bit-storage extension is needed.
// features_rest / features_rest_grad pack each gaussian's scalar stream two per uint.

#ifndef TRAINER_HALF_PACK_SLANG
#define TRAINER_HALF_PACK_SLANG

uint packHalf2x(float a, float b)
{
    return (f32tof16(a) & 0xFFFFu) | (f32tof16(b) << 16u);
}

float2 unpackHalf2x(uint packed)
{
    return float2(f16tof32(packed & 0xFFFFu), f16tof32(packed >> 16u));
}

// uints per gaussian for a 2-per-uint packed scalar stream of `rest_floats`.
uint halfFeatUints(uint rest_floats)
{
    return (rest_floats + 1u) / 2u;
}

// Scalar `s` of the packed gaussian block starting at uint `base`.
float readHalfScalar(StructuredBuffer<uint> buf, uint base, uint s)
{
    uint u = buf[base + s / 2u];
    return (s & 1u) != 0u ? f16tof32(u >> 16u) : f16tof32(u & 0xFFFFu);
}

#endif // TRAINER_HALF_PACK_SLANG
