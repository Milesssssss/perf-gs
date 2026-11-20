// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Periodic Morton-order (Z-curve) sort of the live gaussian prefix so spatial neighbours are memory-adjacent.
// Slot-indexed window accumulators not permuted here must be zeroed by the caller right after the swap.

#include "../common/types.slang"
#include "../common/half_pack.slang"

// Slang integer literals are 32-bit, so 64-bit constants are assembled from halves (warning E40005).
uint64_t _u64(uint hi, uint lo)
{
    return (uint64_t(hi) << 32) | uint64_t(lo);
}

// Spreads a 21-bit integer to every third bit (0, 3, 6, ...) for a 3-way Morton interleave.
uint64_t _spread21(uint v)
{
    uint64_t x = uint64_t(v & 0x1FFFFFu);
    // Each mask isolates the bits that stay after the corresponding shift.
    x = (x | (x << 32)) & _u64(0x001F0000u, 0x0000FFFFu);
    x = (x | (x << 16)) & _u64(0x001F0000u, 0xFF0000FFu);
    x = (x | (x <<  8)) & _u64(0x100F00F0u, 0x0F00F00Fu);
    x = (x | (x <<  4)) & _u64(0x10C30C30u, 0xC30C30C3u);
    x = (x | (x <<  2)) & _u64(0x12492492u, 0x49249249u);
    return x;
}

// 63-bit Morton code: x bits at 0,3,6..., y bits at 1,4,7..., z bits at 2,5,8...
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
        // Max-key sentinel parks tail slots at the end of the sort; the scatter skips them anyway.
        keys_lo[i] = 0xFFFFFFFFu;
        keys_hi[i] = 0xFFFFFFFFu;
        values[i]  = i;
        return;
    }

    float3 p = gaussians[i].mean;
    // Saturate so NaN/out-of-box means still quantize in range; a stale AABB only costs locality.
    float3 t = saturate((p - world_min) * inv_extent);
    const float MAXQ = 2097151.0;  // 2^21 - 1
    uint3 q = uint3(t * MAXQ);

    uint64_t m = _morton63(q);
    keys_lo[i] = uint(m & 0xFFFFFFFFuL);
    keys_hi[i] = uint(m >> 32);
    values[i]  = i;
}

// Single-output variants for the shared scratch arena, one resource at a time: out[i] = in[sorted_values[i]].
[shader("compute")]
[numthreads(256, 1, 1)]
void mortonScatterGaussian(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians_in,
    StructuredBuffer<uint>       sorted_values,
    StructuredBuffer<uint>       count_buf,
    RWStructuredBuffer<Gaussian3D> gaussians_out,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (i >= count_buf[0]) return;
    gaussians_out[i] = gaussians_in[sorted_values[i]];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mortonScatterAdam(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<AdamState> adam_state_in,
    StructuredBuffer<uint>      sorted_values,
    StructuredBuffer<uint>      count_buf,
    RWStructuredBuffer<AdamState> adam_state_out,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (i >= count_buf[0]) return;
    adam_state_out[i] = adam_state_in[sorted_values[i]];
}

// Permutes the SAD-GS eta accumulators with the Gaussians so the eta window
// survives a re-sort; dispatched only on the structure-densify path.
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

[shader("compute")]
[numthreads(256, 1, 1)]
void mortonScatterFeatures(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> features_rest_in,
    StructuredBuffer<uint> sorted_values,
    StructuredBuffer<uint> count_buf,
    RWStructuredBuffer<uint> features_rest_out,
    uniform uint rest_floats,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (i >= count_buf[0]) return;
    uint src = sorted_values[i];
    uint fu = halfFeatUints(rest_floats);
    uint fd = i * fu, fs = src * fu;
    for (uint c = 0u; c < fu; ++c)
        features_rest_out[fd + c] = features_rest_in[fs + c];
}

[shader("compute")]
[numthreads(256, 1, 1)]
void mortonScatterAdamRestPlanes(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> adam_rest_in,
    StructuredBuffer<uint> sorted_values,
    StructuredBuffer<uint> count_buf,
    RWStructuredBuffer<uint> adam_rest_out,
    uniform uint first_plane,
    uniform uint plane_count,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (i >= count_buf[0]) return;
    uint src = sorted_values[i];
    for (uint plane = 0u; plane < plane_count; ++plane)
        adam_rest_out[plane * max_splats + i] =
            adam_rest_in[(first_plane + plane) * max_splats + src];
}
