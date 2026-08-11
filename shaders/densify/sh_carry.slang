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
// adam_rest is SoA float32 (coefficient-major): word k of gaussian `slot` lives
// at k*max_splats + slot (k in [0, 2*rest_floats)). features_rest stays fp16 AoS.
void shInheritRest(uint child, uint parent, uint rest_floats, uint max_splats,
                   RWStructuredBuffer<uint> features_rest,
                   RWStructuredBuffer<uint> adam_rest)
{
    uint fu = halfFeatUints(rest_floats);
    uint fc = child * fu, fp = parent * fu;
    for (uint c = 0u; c < fu; ++c)
        features_rest[fc + c] = features_rest[fp + c];
    uint au = rest_floats * 2u;               // float32 m,v per coeff (SoA words)
    for (uint c = 0u; c < au; ++c)
        adam_rest[c * max_splats + child] = 0u;
}

// Zero a slot's rest Adam moments in place (a SPLIT parent slot becomes child-0:
// it keeps its own features_rest but gets fresh Adam moments).
void shZeroRestAdam(uint slot, uint rest_floats, uint max_splats,
                    RWStructuredBuffer<uint> adam_rest)
{
    uint au = rest_floats * 2u;
    for (uint c = 0u; c < au; ++c)
        adam_rest[c * max_splats + slot] = 0u;
}

#endif // TRAINER_SH_CARRY_SLANG
