// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Shared SH-rest carry helpers for the densify split/clone variants: children inherit the parent's
// coefficients with fresh (zeroed) Adam moments, like their zeroed AdamState.

#ifndef TRAINER_SH_CARRY_SLANG
#define TRAINER_SH_CARRY_SLANG

#include "../common/half_pack.slang"
#include "../common/adam_precision.slang"

// Copies rest coefficients parent->child and zeroes the child's rest moments (a zero word is a zero
// moment in either precision). features_rest is fp16 AoS, (rest_floats+1)/2 uints per gaussian;
// adam_rest is SoA: word k of `slot` at k*max_splats + slot, k < SH_MOMENT_WORDS*rest_floats.
void shInheritRest(uint child, uint parent, uint rest_floats, uint max_splats,
                   RWStructuredBuffer<uint> features_rest,
                   RWStructuredBuffer<uint> adam_rest)
{
    uint fu = halfFeatUints(rest_floats);
    uint fc = child * fu, fp = parent * fu;
    for (uint c = 0u; c < fu; ++c)
        features_rest[fc + c] = features_rest[fp + c];
    uint au = rest_floats * SH_MOMENT_WORDS;
    for (uint c = 0u; c < au; ++c)
        adam_rest[c * max_splats + child] = 0u;
}

// A SPLIT parent slot becomes child-0: it keeps its features_rest but gets fresh rest moments.
void shZeroRestAdam(uint slot, uint rest_floats, uint max_splats,
                    RWStructuredBuffer<uint> adam_rest)
{
    uint au = rest_floats * SH_MOMENT_WORDS;
    for (uint c = 0u; c < au; ++c)
        adam_rest[c * max_splats + slot] = 0u;
}

#endif // TRAINER_SH_CARRY_SLANG
