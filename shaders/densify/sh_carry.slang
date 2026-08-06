// Shared SH-rest carry helpers for the densify split/clone variants.
//
// The optional features_rest coefficients + their Adam moments must follow each
// gaussian through densification exactly as Gaussian3D + AdamState do. Children
// INHERIT the parent's coefficients with FRESH (zeroed) Adam moments, mirroring
// how split/clone give children a zeroed AdamState.

#ifndef TRAINER_SH_CARRY_SLANG
#define TRAINER_SH_CARRY_SLANG

#include "../common/half_pack.slang"

// Copy SH rest coefficients parent->child and zero the child's rest Adam
// moments. All SH buffers are fp16-packed uint32: features_rest is
// (rest_floats+1)/2 uints/gaussian (2 coeffs each); adam_rest is rest_floats
// uints (one packed (m, v) each). A zero uint = two 0.0 halves.
void shInheritRest(uint child, uint parent, uint rest_floats,
                   RWStructuredBuffer<uint> features_rest,
                   RWStructuredBuffer<uint> adam_rest)
{
    uint fu = halfFeatUints(rest_floats);
    uint fc = child * fu, fp = parent * fu;
    for (uint c = 0u; c < fu; ++c)
        features_rest[fc + c] = features_rest[fp + c];
    uint ac = child * rest_floats;
    for (uint c = 0u; c < rest_floats; ++c)
        adam_rest[ac + c] = 0u;
}

// Zero a slot's rest Adam moments in place (a SPLIT parent slot becomes child-0:
// it keeps its own features_rest but gets fresh Adam moments).
void shZeroRestAdam(uint slot, uint rest_floats, RWStructuredBuffer<uint> adam_rest)
{
    uint a = slot * rest_floats;
    for (uint c = 0u; c < rest_floats; ++c)
        adam_rest[a + c] = 0u;
}

#endif // TRAINER_SH_CARRY_SLANG
