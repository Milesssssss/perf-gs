// Phase E of densification: scatter live gaussians + Adam state into
// scratch buffers using the prefix-scan-derived destination indices.
//
// For each index i in the post-split range:
//   if valid_mask[i] == 1:
//       gaussians_out[scan[i]]  = gaussians_in[i]
//       adam_state_out[scan[i]] = adam_state_in[i]
//
// The caller (Python) is responsible for:
//   - ensuring scan[] was produced on the same valid_mask[]
//   - swapping primary<->scratch buffer references after this dispatch
//   - updating num_gaussians_counter to scan[N-1] + valid_mask[N-1]
//     (available as the ``total_out`` of the scan pass).

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
    // Dispatched over [max_splats]; buildPruneMask zeroes valid_mask for
    // i >= post_count, so those slots are naturally skipped here.
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (valid_mask[i] == 0u) return;
    uint dst = scan[i];
    gaussians_out[dst]  = gaussians_in[i];
    adam_state_out[dst] = adam_state_in[i];
}

// Single-output variants of scatterCompact, used by the shared-scratch-arena
// path (see buffers.ScratchArena / densify.py). The fused two-output kernel
// above requires two full-size scratch buffers live at once; splitting the
// compaction into one-output-per-dispatch lets a SINGLE arena buffer serve as
// the scatter destination for each resource in turn (scatter -> copy arena
// back into the live buffer -> reuse arena for the next resource). Same
// valid_mask / scan contract as scatterCompact; dst = scan[i] <= i so the
// caller's arena->live copy is safe.
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

// Eta-only compaction for the shared-scratch-arena path: the 6 SAD-GS eta
// accumulators through the SAME valid_mask/scan the gaussian compaction used,
// with the identical KEEP/CLONE carry logic as scatterCompactEta (split/prune
// slots reset to 0). Gaussians + Adam are compacted separately by
// scatterCompactGaussian / scatterCompactAdam so they can route through the
// arena; eta keeps its own small dedicated scratch + swap.
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

// Additive SH-carry for compaction: move the optional features_rest coeffs +
// their Adam moments through the SAME valid_mask / scan the gaussian compaction
// used, so they land on the compacted slot. Dispatched only when SH is enabled,
// after scatterCompact / scatterCompactEta. All SH buffers are fp16-packed
// uint32 (features: (rest_floats+1)/2 uints/gaussian; adam: rest_floats uints).
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
    // adam_rest SoA float32: word c of gaussian g at c*max_splats + g.
    uint au = rest_floats * 2u;
    for (uint c = 0u; c < au; ++c)
        adam_rest_out[c * max_splats + dst] = adam_rest_in[c * max_splats + i];
}

// Single-output SH compaction variants for the shared-scratch-arena path.
// Same fp16-packed layout and valid_mask/scan contract as scatterCompactSH.
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
    // adam_rest SoA float32: word c of gaussian g at c*max_splats + g.
    uint au = rest_floats * 2u;
    for (uint c = 0u; c < au; ++c)
        adam_rest_out[c * max_splats + dst] = adam_rest_in[c * max_splats + i];
}
