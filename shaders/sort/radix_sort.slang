// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Stable LSD radix sort of 64-bit keys as (keys_lo, keys_hi) uint pairs (no portable uint64) with
// uint32 values; each 8-bit pass runs blockSort -> globalScan -> scanDigitTotals -> scatterFinal.

#ifndef TRAINER_RADIX_SORT_SLANG
#define TRAINER_RADIX_SORT_SLANG

static const uint RADIX_BITS = 8;
static const uint RADIX_BINS = 256;
static const uint RADIX_MASK = 255;
static const uint RADIX_PASSES = 8;
static const uint SORT_BLOCK_SIZE = 256;

uint radixDigit(uint key_lo, uint key_hi, uint pass_idx)
{
    uint shifted = (pass_idx < 4)
        ? (key_lo >> (pass_idx * RADIX_BITS))
        : (key_hi >> ((pass_idx - 4) * RADIX_BITS));
    return shifted & RADIX_MASK;
}

groupshared uint gs_klo[SORT_BLOCK_SIZE];
groupshared uint gs_khi[SORT_BLOCK_SIZE];
groupshared uint gs_val[SORT_BLOCK_SIZE];
groupshared uint gs_scan[SORT_BLOCK_SIZE];
groupshared uint gs_hist[RADIX_BINS];

[shader("compute")]
[numthreads(SORT_BLOCK_SIZE, 1, 1)]
void blockSort(
    uint3 gid: SV_GroupID,
    uint lid: SV_GroupIndex,
    RWStructuredBuffer<uint> keys_lo,
    RWStructuredBuffer<uint> keys_hi,
    RWStructuredBuffer<uint> values,
    RWStructuredBuffer<uint> block_hist,      // [num_blocks_max * 256]
    RWStructuredBuffer<uint> block_start,     // [num_blocks_max * 256]
    StructuredBuffer<uint> n_buf,
    uniform uint pass_idx)
{
    uint n = n_buf[0];
    uint block_base = gid.x * SORT_BLOCK_SIZE;

    // Entirely-sentinel block: zero the histogram row and return.
    if (block_base >= n) {
        block_hist[gid.x * RADIX_BINS + lid] = 0u;
        block_start[gid.x * RADIX_BINS + lid] = 0u;
        return;
    }

    uint global_idx = block_base + lid;
    bool active = global_idx < n;

    uint klo = active ? keys_lo[global_idx] : 0xFFFFFFFFu;
    uint khi = active ? keys_hi[global_idx] : 0xFFFFFFFFu;
    uint val = active ? values[global_idx]  : 0xFFFFFFFFu;

    gs_klo[lid] = klo;
    gs_khi[lid] = khi;
    gs_val[lid] = val;
    GroupMemoryBarrierWithGroupSync();

    // 8 stable 1-bit splits: zero bits move first, then one bits, each keeping input order.
    for (uint sub = 0u; sub < 8u; ++sub)
    {
        uint k_lo_me  = gs_klo[lid];
        uint k_hi_me  = gs_khi[lid];
        uint v_me     = gs_val[lid];
        uint digit_me = radixDigit(k_lo_me, k_hi_me, pass_idx);
        uint bit      = (digit_me >> sub) & 1u;
        uint flag     = bit ^ 1u;   // 1 if the examined bit is zero

        GroupMemoryBarrierWithGroupSync();
        gs_scan[lid] = flag;
        GroupMemoryBarrierWithGroupSync();

        // Hillis-Steele inclusive scan of flags across the block.
        for (uint off = 1u; off < SORT_BLOCK_SIZE; off <<= 1u)
        {
            uint add_val = (lid >= off) ? gs_scan[lid - off] : 0u;
            GroupMemoryBarrierWithGroupSync();
            gs_scan[lid] += add_val;
            GroupMemoryBarrierWithGroupSync();
        }

        uint zeros_incl  = gs_scan[lid];
        uint total_zeros = gs_scan[SORT_BLOCK_SIZE - 1u];
        uint zeros_excl  = zeros_incl - flag;

        uint new_pos = (bit == 0u)
            ? zeros_excl
            : (total_zeros + (lid - zeros_excl));

        GroupMemoryBarrierWithGroupSync();
        gs_klo[new_pos] = k_lo_me;
        gs_khi[new_pos] = k_hi_me;
        gs_val[new_pos] = v_me;
        GroupMemoryBarrierWithGroupSync();
    }

    uint klo_s = gs_klo[lid];
    uint khi_s = gs_khi[lid];
    uint val_s = gs_val[lid];

    // Sentinel keys (digit 0xFF) keep their tail position under the stable sort, so real items
    // occupy [0, real_in_block) and only they are written or counted.
    uint real_in_block = min(SORT_BLOCK_SIZE, n - block_base);
    bool real_slot = lid < real_in_block;

    if (real_slot) {
        keys_lo[block_base + lid] = klo_s;
        keys_hi[block_base + lid] = khi_s;
        values[block_base + lid]  = val_s;
    }

    // Per-block histogram: count only real items.
    uint d_s = radixDigit(klo_s, khi_s, pass_idx);
    gs_hist[lid] = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (real_slot) {
        InterlockedAdd(gs_hist[d_s], 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    // block_start[d]: first position of digit d's real items in the block's sorted layout.
    gs_scan[lid] = gs_hist[lid];
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1u; off < RADIX_BINS; off <<= 1u)
    {
        uint add_val = (lid >= off) ? gs_scan[lid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gs_scan[lid] += add_val;
        GroupMemoryBarrierWithGroupSync();
    }
    uint excl = gs_scan[lid] - gs_hist[lid];
    block_hist[gid.x * RADIX_BINS + lid]  = gs_hist[lid];
    block_start[gid.x * RADIX_BINS + lid] = excl;
}

// One thread per digit serially scans block_hist over the active blocks only (n from n_buf).
[shader("compute")]
[numthreads(RADIX_BINS, 1, 1)]
void globalScan(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> block_hist,
    RWStructuredBuffer<uint> block_prefix,    // [num_blocks_max * 256]
    RWStructuredBuffer<uint> digit_total,     // [256]
    StructuredBuffer<uint> n_buf)
{
    uint d = dtid.x;
    if (d >= RADIX_BINS) return;

    uint n = n_buf[0];
    uint active_blocks = (n + SORT_BLOCK_SIZE - 1u) / SORT_BLOCK_SIZE;

    uint running = 0u;
    for (uint b = 0u; b < active_blocks; ++b)
    {
        block_prefix[b * RADIX_BINS + d] = running;
        running += block_hist[b * RADIX_BINS + d];
    }
    digit_total[d] = running;
}

// Single-workgroup exclusive scan of digit_total into digit_global_base.
groupshared uint gs_dtotal[RADIX_BINS];

[shader("compute")]
[numthreads(RADIX_BINS, 1, 1)]
void scanDigitTotals(
    uint lid: SV_GroupIndex,
    StructuredBuffer<uint> digit_total,
    RWStructuredBuffer<uint> digit_global_base)
{
    uint my_val = digit_total[lid];
    gs_dtotal[lid] = my_val;
    GroupMemoryBarrierWithGroupSync();

    for (uint off = 1u; off < RADIX_BINS; off <<= 1u)
    {
        uint add_val = (lid >= off) ? gs_dtotal[lid - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gs_dtotal[lid] += add_val;
        GroupMemoryBarrierWithGroupSync();
    }

    digit_global_base[lid] = gs_dtotal[lid] - my_val;   // exclusive
}

[shader("compute")]
[numthreads(SORT_BLOCK_SIZE, 1, 1)]
void scatterFinal(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> keys_lo_in,
    StructuredBuffer<uint> keys_hi_in,
    StructuredBuffer<uint> values_in,
    StructuredBuffer<uint> block_start,
    StructuredBuffer<uint> block_prefix,
    StructuredBuffer<uint> digit_global_base,
    RWStructuredBuffer<uint> keys_lo_out,
    RWStructuredBuffer<uint> keys_hi_out,
    RWStructuredBuffer<uint> values_out,
    StructuredBuffer<uint> n_buf,
    uniform uint pass_idx)
{
    uint n = n_buf[0];
    uint global_idx = dtid.x;
    if (global_idx >= n) return;

    uint block_id = global_idx / SORT_BLOCK_SIZE;
    uint local_id = global_idx - block_id * SORT_BLOCK_SIZE;

    uint klo = keys_lo_in[global_idx];
    uint khi = keys_hi_in[global_idx];
    uint val = values_in[global_idx];
    uint d   = radixDigit(klo, khi, pass_idx);

    uint bs  = block_start[block_id * RADIX_BINS + d];
    uint bp  = block_prefix[block_id * RADIX_BINS + d];
    uint gb  = digit_global_base[d];
    uint final_pos = gb + bp + (local_id - bs);

    keys_lo_out[final_pos] = klo;
    keys_hi_out[final_pos] = khi;
    values_out[final_pos]  = val;
}

#endif // TRAINER_RADIX_SORT_SLANG
