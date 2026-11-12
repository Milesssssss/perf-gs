// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Storage precision of the Adam moments: float32 by default, bf16 under
// ADAM_MOMENTS_BF16 (one uint packs a scalar's m in its low half, v in its
// high half). bf16 keeps float32's exponent range, so the ~1e-12 SH-rest v that
// underflowed fp16 stays representable. Rounding is stochastic: v's per-step
// EMA change (1 - beta2 = 0.1%) is below half a bf16 ulp (0.4%), so
// round-to-nearest would freeze v.

#ifndef TRAINER_ADAM_PRECISION_SLANG
#define TRAINER_ADAM_PRECISION_SLANG

#if defined(ADAM_MOMENTS_BF16)
static const uint SH_MOMENT_WORDS = 1u;
#else
static const uint SH_MOMENT_WORDS = 2u;
#endif

// One SH-rest coefficient's stored moments. The host sizes adam_rest from this
// struct's reflected stride.
struct ShRestMoments
{
    uint words[SH_MOMENT_WORDS];
};

// Rounding-noise seed for one gaussian at one optimizer step.
uint adamRoundingSeed(uint idx, uint step)
{
    return (idx * 0x9E3779B9u) ^ (step * 0x85EBCA6Bu);
}

// PCG hash: 32 fresh random bits per call.
uint nextRoundingNoise(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float  bf16Lo(uint  p) { return asfloat(p << 16u); }
float3 bf16Lo(uint3 p) { return asfloat(p << 16u); }
float4 bf16Lo(uint4 p) { return asfloat(p << 16u); }
float  bf16Hi(uint  p) { return asfloat(p & 0xFFFF0000u); }
float3 bf16Hi(uint3 p) { return asfloat(p & 0xFFFF0000u); }
float4 bf16Hi(uint4 p) { return asfloat(p & 0xFFFF0000u); }

// Round lo/hi to bf16 by adding 16 random bits below the kept mantissa, then
// truncating. Values already representable in bf16 pass through unchanged.
uint packBf16(float lo, float hi, inout uint rng)
{
    uint noise = nextRoundingNoise(rng);
    return ((asuint(lo) + (noise & 0xFFFFu)) >> 16u)
         | ((asuint(hi) + (noise >> 16u)) & 0xFFFF0000u);
}

uint3 packBf16(float3 lo, float3 hi, inout uint rng)
{
    uint3 p;
    p.x = packBf16(lo.x, hi.x, rng);
    p.y = packBf16(lo.y, hi.y, rng);
    p.z = packBf16(lo.z, hi.z, rng);
    return p;
}

uint4 packBf16(float4 lo, float4 hi, inout uint rng)
{
    uint4 p;
    p.x = packBf16(lo.x, hi.x, rng);
    p.y = packBf16(lo.y, hi.y, rng);
    p.z = packBf16(lo.z, hi.z, rng);
    p.w = packBf16(lo.w, hi.w, rng);
    return p;
}

// SH-rest moments are SoA planes of max_splats words; coefficient s owns
// planes [s * SH_MOMENT_WORDS, (s + 1) * SH_MOMENT_WORDS).
float2 loadRestMoments(RWStructuredBuffer<uint> adam_rest, uint s, uint idx, uint max_splats)
{
#if defined(ADAM_MOMENTS_BF16)
    uint p = adam_rest[s * max_splats + idx];
    return float2(bf16Lo(p), bf16Hi(p));
#else
    return float2(asfloat(adam_rest[(s * 2u) * max_splats + idx]),
                  asfloat(adam_rest[(s * 2u + 1u) * max_splats + idx]));
#endif
}

void storeRestMoments(RWStructuredBuffer<uint> adam_rest, uint s, uint idx, uint max_splats,
                      float2 mv, inout uint rng)
{
#if defined(ADAM_MOMENTS_BF16)
    adam_rest[s * max_splats + idx] = packBf16(mv.x, mv.y, rng);
#else
    adam_rest[(s * 2u) * max_splats + idx] = asuint(mv.x);
    adam_rest[(s * 2u + 1u) * max_splats + idx] = asuint(mv.y);
#endif
}

#endif // TRAINER_ADAM_PRECISION_SLANG
