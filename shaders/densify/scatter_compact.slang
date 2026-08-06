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

// Variant of scatterCompact that ALSO carries the slot-indexed SAD-GS eta
// accumulators through the compaction, so the multiview frequency statistics
// survive a filler (AbsGS) densify instead of being reset every 100 iters.
// SADGS accumulates eta over the whole ~500-iter structure interval (~50
// samples/gaussian); resetting after every filler pass collapsed that to ~10.
//
// Carry rule (matches SADGS: surviving gaussians keep their eta, brand-new
// primitives start at zero):
//   i >= pre_count                          -> appended child        -> zero
//   i <  pre_count, action == KEEP          -> untouched survivor    -> carry
//   i <  pre_count, action == CLONE         -> untouched parent slot -> carry
//   i <  pre_count, action == SPLIT         -> slot overwritten with
//                                              a NEW child-1          -> zero
//   i <  pre_count, action == PRUNE         -> dropped (valid_mask 0) -> n/a
// So: carry iff (i < pre_count && action in {KEEP, CLONE}); else zero.
//
// eta_n_total is bundled with the 5 eta_buffers components (n_high, n_low,
// max_x, max_y, max_z) — all 6 permute identically. Dispatched over
// [max_splats]; the same valid_mask / scan produced for the gaussian scatter
// drive the eta scatter, so the two permutations are identical by construction.
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

// Additive SH-carry for compaction: move the optional features_rest coeffs +
// their Adam moments through the SAME valid_mask / scan the gaussian compaction
// used, so they land on the compacted slot. Dispatched only when SH is enabled,
// after scatterCompact / scatterCompactEta (whichever moved the gaussians).
[shader("compute")]
[numthreads(256, 1, 1)]
void scatterCompactSH(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<float> features_rest_in,
    StructuredBuffer<float> adam_rest_in,
    StructuredBuffer<uint>  valid_mask,
    StructuredBuffer<uint>  scan,
    RWStructuredBuffer<float> features_rest_out,
    RWStructuredBuffer<float> adam_rest_out,
    uniform uint rest_floats,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (valid_mask[i] == 0u) return;
    uint dst = scan[i];
    uint pd = dst * rest_floats, ps = i * rest_floats;
    uint ad = dst * rest_floats * 2u, asrc = i * rest_floats * 2u;
    for (uint c = 0u; c < rest_floats; ++c)
    {
        features_rest_out[pd + c] = features_rest_in[ps + c];
        adam_rest_out[ad + c * 2u + 0u] = adam_rest_in[asrc + c * 2u + 0u];
        adam_rest_out[ad + c * 2u + 1u] = adam_rest_in[asrc + c * 2u + 1u];
    }
}
