// fp16-into-uint32 packing for the optional SH buffers. Native 16-bit storage
// buffers aren't relied on here (mirrors checkpoint.slang, which packs T/RGB the
// same way with f32tof16), so this works without the 16-bit-storage extension.
//
// Two coefficients per uint. features_rest / features_rest_grad pack the flat
// scalar stream 2-per-uint ((rest_floats+1)/2 uints per gaussian). adam_rest
// packs (m, v) of ONE scalar per uint (rest_floats uints per gaussian).

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

// Read scalar `s` (0..rest_floats-1) from a 2-per-uint packed buffer whose
// gaussian block starts at uint `base`.
float readHalfScalar(StructuredBuffer<uint> buf, uint base, uint s)
{
    uint u = buf[base + s / 2u];
    return (s & 1u) != 0u ? f16tof32(u >> 16u) : f16tof32(u & 0xFFFFu);
}

#endif // TRAINER_HALF_PACK_SLANG
