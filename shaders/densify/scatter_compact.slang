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
