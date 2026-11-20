// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Densify phase E: scatter each live resource to its scan[i] slot in the shared scratch arena.

#include "../common/types.slang"
#include "../common/half_pack.slang"

static const uint ACTION_KEEP  = 0;
static const uint ACTION_CLONE = 2;

// Single-output variants so one shared scratch arena serves each resource in turn (scatter, copy back,
// reuse); dst = scan[i] <= i keeps the caller's arena->live copy safe.
[shader("compute")]
[numthreads(256, 1, 1)]
void scatterCompactGaussian(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians_in,
    StructuredBuffer<uint>       valid_mask,
    StructuredBuffer<uint>       scan,
    RWStructuredBuffer<Gaussian3D> gaussians_out,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (valid_mask[i] == 0u) return;
    gaussians_out[scan[i]] = gaussians_in[i];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void scatterCompactAdam(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<AdamState> adam_state_in,
    StructuredBuffer<uint>      valid_mask,
    StructuredBuffer<uint>      scan,
    RWStructuredBuffer<AdamState> adam_state_out,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (valid_mask[i] == 0u) return;
    adam_state_out[scan[i]] = adam_state_in[i];
}

// Preserve eta for surviving parents. New children start with an empty accumulation window.
[shader("compute")]
[numthreads(256, 1, 1)]
void scatterCompactEtaOnly(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> valid_mask,
    StructuredBuffer<uint> scan,
    StructuredBuffer<uint> action,
    StructuredBuffer<uint> pre_count_buf,
    StructuredBuffer<uint> eta_n_total_in,
    StructuredBuffer<uint> eta_n_high_in,
    StructuredBuffer<uint> eta_n_low_in,
    StructuredBuffer<uint> eta_max_x_in,
    StructuredBuffer<uint> eta_max_y_in,
    StructuredBuffer<uint> eta_max_z_in,
    RWStructuredBuffer<uint> eta_n_total_out,
    RWStructuredBuffer<uint> eta_n_high_out,
    RWStructuredBuffer<uint> eta_n_low_out,
    RWStructuredBuffer<uint> eta_max_x_out,
    RWStructuredBuffer<uint> eta_max_y_out,
    RWStructuredBuffer<uint> eta_max_z_out,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (valid_mask[i] == 0u) return;
    uint dst = scan[i];
    uint pre_count = pre_count_buf[0];
    bool carry = (i < pre_count) &&
                 (action[i] == ACTION_KEEP || action[i] == ACTION_CLONE);
    if (carry)
    {
        eta_n_total_out[dst] = eta_n_total_in[i];
        eta_n_high_out[dst]  = eta_n_high_in[i];
        eta_n_low_out[dst]   = eta_n_low_in[i];
        eta_max_x_out[dst]   = eta_max_x_in[i];
        eta_max_y_out[dst]   = eta_max_y_in[i];
        eta_max_z_out[dst]   = eta_max_z_in[i];
    }
    else
    {
        eta_n_total_out[dst] = 0u;
        eta_n_high_out[dst]  = 0u;
        eta_n_low_out[dst]   = 0u;
        eta_max_x_out[dst]   = 0u;
        eta_max_y_out[dst]   = 0u;
        eta_max_z_out[dst]   = 0u;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void scatterCompactFeatures(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> features_rest_in,
    StructuredBuffer<uint>  valid_mask,
    StructuredBuffer<uint>  scan,
    RWStructuredBuffer<uint> features_rest_out,
    uniform uint rest_floats,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (valid_mask[i] == 0u) return;
    uint dst = scan[i];
    uint fu = halfFeatUints(rest_floats);
    uint fd = dst * fu, fs = i * fu;
    for (uint c = 0u; c < fu; ++c)
        features_rest_out[fd + c] = features_rest_in[fs + c];
}

// Bounds scratch by scattering a subset of moment planes; arena planes start at 0, copied back to first_plane.
[shader("compute")]
[numthreads(256, 1, 1)]
void scatterCompactAdamRestPlanes(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> adam_rest_in,
    StructuredBuffer<uint> valid_mask,
    StructuredBuffer<uint> scan,
    RWStructuredBuffer<uint> adam_rest_out,
    uniform uint first_plane,
    uniform uint plane_count,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (valid_mask[i] == 0u) return;
    uint dst = scan[i];
    for (uint plane = 0u; plane < plane_count; ++plane)
        adam_rest_out[plane * max_splats + dst] =
            adam_rest_in[(first_plane + plane) * max_splats + i];
}
