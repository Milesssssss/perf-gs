// Shared SH-rest carry helpers for the densify split/clone variants.
//
// The optional features_rest coefficients + their Adam moments must follow each
// gaussian through densification exactly as Gaussian3D + AdamState do. Children
// INHERIT the parent's coefficients with FRESH (zeroed) Adam moments, mirroring
// how split/clone give children a zeroed AdamState.

#ifndef TRAINER_SH_CARRY_SLANG
#define TRAINER_SH_CARRY_SLANG

// Copy SH rest coefficients parent->child and zero the child's rest Adam
// moments. Flat layout: features_rest[slot*rest_floats + c],
// adam_rest[slot*rest_floats*2 + c*2 + {0,1}] (m, v).
void shInheritRest(uint child, uint parent, uint rest_floats,
                   RWStructuredBuffer<float> features_rest,
                   RWStructuredBuffer<float> adam_rest)
{
    uint pc = child * rest_floats, pp = parent * rest_floats;
    uint ac = child * rest_floats * 2u;
    for (uint c = 0u; c < rest_floats; ++c)
    {
        features_rest[pc + c] = features_rest[pp + c];
        adam_rest[ac + c * 2u + 0u] = 0.0;
        adam_rest[ac + c * 2u + 1u] = 0.0;
    }
}

// Zero a slot's rest Adam moments in place (a SPLIT parent slot becomes child-0:
// it keeps its own features_rest but gets fresh Adam moments).
void shZeroRestAdam(uint slot, uint rest_floats, RWStructuredBuffer<float> adam_rest)
{
    uint a = slot * rest_floats * 2u;
    for (uint c = 0u; c < rest_floats * 2u; ++c)
        adam_rest[a + c] = 0.0;
}

#endif // TRAINER_SH_CARRY_SLANG
