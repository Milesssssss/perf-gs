// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Densify phase E: scatter valid gaussians + Adam state to their scan[i] slot; the caller swaps
// primary/scratch buffers and sets num_gaussians_counter to the scan total.

#include "../common/types.slang"
#include "../common/half_pack.slang"

static const uint ACTION_KEEP  = 0;
static const uint ACTION_SPLIT = 1;
static const uint ACTION_CLONE = 2;
static const uint ACTION_PRUNE = 3;

[shader("compute")]
[numthreads(256, 1, 1)]
void scatterCompact(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians_in,
    StructuredBuffer<AdamState>  adam_state_in,
    StructuredBuffer<uint>       valid_mask,
    StructuredBuffer<uint>       scan,
    RWStructuredBuffer<Gaussian3D> gaussians_out,
    RWStructuredBuffer<AdamState>  adam_state_out,
    uniform uint max_splats)
{
    // buildPruneMask zeroes valid_mask beyond post_count, so the max_splats-wide dispatch skips the tail.
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (valid_mask[i] == 0u) return;
    uint dst = scan[i];
    gaussians_out[dst]  = gaussians_in[i];
    adam_state_out[dst] = adam_state_in[i];
}

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

[shader("compute")]
[numthreads(256, 1, 1)]
void scatterCompactEta(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians_in,
    StructuredBuffer<AdamState>  adam_state_in,
    StructuredBuffer<uint>       valid_mask,
    StructuredBuffer<uint>       scan,
    StructuredBuffer<uint>       action,
    StructuredBuffer<uint>       pre_count_buf,
    StructuredBuffer<uint>       eta_n_total_in,
    StructuredBuffer<uint>       eta_n_high_in,
    StructuredBuffer<uint>       eta_n_low_in,
    StructuredBuffer<uint>       eta_max_x_in,
    StructuredBuffer<uint>       eta_max_y_in,
    StructuredBuffer<uint>       eta_max_z_in,
    RWStructuredBuffer<Gaussian3D> gaussians_out,
    RWStructuredBuffer<AdamState>  adam_state_out,
    RWStructuredBuffer<uint>       eta_n_total_out,
    RWStructuredBuffer<uint>       eta_n_high_out,
    RWStructuredBuffer<uint>       eta_n_low_out,
    RWStructuredBuffer<uint>       eta_max_x_out,
    RWStructuredBuffer<uint>       eta_max_y_out,
    RWStructuredBuffer<uint>       eta_max_z_out,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (valid_mask[i] == 0u) return;
    uint dst = scan[i];

    gaussians_out[dst]  = gaussians_in[i];
    adam_state_out[dst] = adam_state_in[i];

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

// Eta-only compaction for the arena path, with scatterCompactEta's carry (other slots reset to 0).
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

// Moves SH rest coefficients and their Adam moments through the same valid_mask/scan; features are
// fp16-packed uint32, Adam uses 2*rest_floats FP32/uint planes with max_splats stride.
[shader("compute")]
[numthreads(256, 1, 1)]
void scatterCompactSH(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> features_rest_in,
    StructuredBuffer<uint> adam_rest_in,
    StructuredBuffer<uint>  valid_mask,
    StructuredBuffer<uint>  scan,
    RWStructuredBuffer<uint> features_rest_out,
    RWStructuredBuffer<uint> adam_rest_out,
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
    // adam_rest SoA: word c of gaussian g at c*max_splats + g.
    uint au = rest_floats * SH_MOMENT_WORDS;
    for (uint c = 0u; c < au; ++c)
        adam_rest_out[c * max_splats + dst] = adam_rest_in[c * max_splats + i];
}

// Single-output SH compaction variants for the arena path; same layout and contract as scatterCompactSH.
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

[shader("compute")]
[numthreads(256, 1, 1)]
void scatterCompactAdamRest(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> adam_rest_in,
    StructuredBuffer<uint>  valid_mask,
    StructuredBuffer<uint>  scan,
    RWStructuredBuffer<uint> adam_rest_out,
    uniform uint rest_floats,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (valid_mask[i] == 0u) return;
    uint dst = scan[i];
    // adam_rest SoA: word c of gaussian g at c*max_splats + g.
    uint au = rest_floats * SH_MOMENT_WORDS;
    for (uint c = 0u; c < au; ++c)
        adam_rest_out[c * max_splats + dst] = adam_rest_in[c * max_splats + i];
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
