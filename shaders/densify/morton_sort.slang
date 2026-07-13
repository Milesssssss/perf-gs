// Periodic Morton-order (Z-curve) sort of the gaussian buffer.
//
// Reorders the [0, num_gaussians) prefix of buffers.gaussians and
// buffers.adam_state so that spatially-adjacent primitives are also
// memory-adjacent. Improves cache locality for later per-splat reads
// (bin_count, blend, backward, adam) once the array grows large.
//
// Pipeline (all inside one command encoder, invoked between train_steps):
//
//   1. Python: snapshotCount(num_gaussians_counter -> scan_total) so the
//      radix sort and both morton kernels see the same GPU-side n.
//
//   2. mortonEncode  — this file. Reads gaussians[i].mean, normalizes into
//      [0, 1]^3 via a caller-supplied world AABB, quantizes each axis to
//      21 bits, interleaves into a 63-bit Morton code, writes (keys_lo,
//      keys_hi, values=i) into the A-slot sort scratch. Tail (i >= n)
//      writes 0xFFFFFFFF sentinels so the radix's partial-partition tail
//      sorts to the end — but the scatter early-outs on i >= n anyway.
//
//   3. Python: Pipeline._radix_sort_pairs() — sorts (keys_hi:keys_lo,
//      values) ascending. Deterministic 8-pass LSD returns to the A slot.
//
//   4. mortonScatter — this file. dst[i] = src[sorted_values[i]] for
//      i in [0, n). Copies BOTH Gaussian3D and AdamState in one dispatch
//      (whole-struct assign — Slang emits per-scalar loads/stores).
//
//   5. Python: swap buffers.gaussians <-> buffers.gaussians_scratch and
//      buffers.adam_state <-> buffers.adam_state_scratch.
//
// Slot-indexed window accumulators (grad_mean_2d_mag_accum,
// visibility_count_accum, eta_*, importance_score_accum) are NOT permuted
// by this kernel — the caller MUST zero them (clear_tracking, and
// clear_structure_scores on the SAD path) immediately after the swap,
// or the next densify reads window statistics for the wrong slot.

#include "../common/types.slang"

// Split a 21-bit integer into every third bit (positions 0, 3, 6, ...).
// Classic magic-bit-spread for a 3-way Morton interleave; returns a
// 63-bit value packed into a uint64.
//
// Slang's integer literals default to 32-bit — 64-bit constants must be
// assembled from (hi<<32)|lo halves, otherwise the literal silently
// truncates to 32 bits. Warning E40005 flags this at compile time.
uint64_t _u64(uint hi, uint lo)
{
    return (uint64_t(hi) << 32) | uint64_t(lo);
}

uint64_t _spread21(uint v)
{
    // Confine to 21 bits.
    uint64_t x = uint64_t(v & 0x1FFFFFu);
    // 5-step de-Bruijn-style spread: each mask isolates the bits that
    // stay after the corresponding shift.
    x = (x | (x << 32)) & _u64(0x001F0000u, 0x0000FFFFu);
    x = (x | (x << 16)) & _u64(0x001F0000u, 0xFF0000FFu);
    x = (x | (x <<  8)) & _u64(0x100F00F0u, 0x0F00F00Fu);
    x = (x | (x <<  4)) & _u64(0x10C30C30u, 0xC30C30C3u);
    x = (x | (x <<  2)) & _u64(0x12492492u, 0x49249249u);
    return x;
}

// Interleave three 21-bit ints into one 63-bit Morton code
// (x-bits at 0,3,6..., y-bits at 1,4,7..., z-bits at 2,5,8...).
uint64_t _morton63(uint3 q)
{
    return _spread21(q.x) | (_spread21(q.y) << 1) | (_spread21(q.z) << 2);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mortonEncode(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint>       count_buf,     // scan_total[0] = n
    RWStructuredBuffer<uint>     keys_lo,       // sort_keys_lo_a
    RWStructuredBuffer<uint>     keys_hi,       // sort_keys_hi_a
    RWStructuredBuffer<uint>     values,        // sort_values_a
    uniform float3 world_min,                   // AABB min (all-mean min - pad)
    uniform float3 inv_extent,                  // 1 / (world_max - world_min)
    uniform uint   max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    uint n = count_buf[0];

    if (i >= n) {
        // Tail slots: put them past any valid Morton code so the sort
        // parks them at the end. Scatter early-outs on i >= n anyway,
        // so the sentinel doesn't need to be a specific value beyond
        // "sorts last"; 0xFFFFFFFF/0xFFFFFFFF is the maximum 64-bit key.
        keys_lo[i] = 0xFFFFFFFFu;
        keys_hi[i] = 0xFFFFFFFFu;
        values[i]  = i;
        return;
    }

    float3 p = gaussians[i].mean;
    // Normalize into [0, 1] via world AABB; saturate against NaN/Inf and
    // out-of-box means so quantization always lands in [0, 2^21 - 1].
    // A stale AABB is fine — Morton ordering is a cache-friendliness
    // heuristic, not a correctness constraint.
    float3 t = saturate((p - world_min) * inv_extent);
    const float MAXQ = 2097151.0;  // 2^21 - 1
    uint3 q = uint3(t * MAXQ);

    uint64_t m = _morton63(q);
    keys_lo[i] = uint(m & 0xFFFFFFFFuL);
    keys_hi[i] = uint(m >> 32);
    values[i]  = i;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mortonScatter(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians_in,
    StructuredBuffer<AdamState>  adam_state_in,
    StructuredBuffer<uint>       sorted_values, // radix output values slot
    StructuredBuffer<uint>       count_buf,     // scan_total[0] = n
    RWStructuredBuffer<Gaussian3D> gaussians_out,
    RWStructuredBuffer<AdamState>  adam_state_out,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    uint n = count_buf[0];
    if (i >= n) return;
    // src is a permutation of [0, n); every valid slot is written once.
    // Tail slots [n, max_splats) in gaussians_out / adam_state_out are
    // NOT written and hold whatever the scratch buffer had before —
    // matches scatter_compact.slang's contract (only [0, count) is
    // semantically meaningful; the tail is never read).
    uint src = sorted_values[i];
    gaussians_out[i]  = gaussians_in[src];
    adam_state_out[i] = adam_state_in[src];
}

// Permute the slot-indexed SAD-GS eta accumulators by the SAME sorted_values
// permutation mortonScatter applies to the gaussians, so the eta window
// survives a Morton re-sort instead of being reset. Pure permutation (Morton
// neither creates nor destroys primitives): dst[i] = src[perm[i]], no zeroing.
//
// Separate entry point (not folded into mortonScatter) so the gaussian re-sort
// and the render-only viewer path stay untouched; this is dispatched only on
// the structure-densify path, reading the identical sorted_values slot so the
// two permutations are guaranteed consistent. eta_n_total travels with the 5
// eta_buffers components (n_high, n_low, max_x, max_y, max_z).
[shader("compute")]
[numthreads(256, 1, 1)]
void mortonScatterEta(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> eta_n_total_in,
    StructuredBuffer<uint> eta_n_high_in,
    StructuredBuffer<uint> eta_n_low_in,
    StructuredBuffer<uint> eta_max_x_in,
    StructuredBuffer<uint> eta_max_y_in,
    StructuredBuffer<uint> eta_max_z_in,
    StructuredBuffer<uint> sorted_values,
    StructuredBuffer<uint> count_buf,
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
    uint n = count_buf[0];
    if (i >= n) return;
    uint src = sorted_values[i];
    eta_n_total_out[i] = eta_n_total_in[src];
    eta_n_high_out[i]  = eta_n_high_in[src];
    eta_n_low_out[i]   = eta_n_low_in[src];
    eta_max_x_out[i]   = eta_max_x_in[src];
    eta_max_y_out[i]   = eta_max_y_in[src];
    eta_max_z_out[i]   = eta_max_z_in[src];
}
