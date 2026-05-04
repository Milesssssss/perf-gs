// Block-local stable radix sort on 64-bit keys with paired uint32 values.
// Keys split into (keys_lo, keys_hi) to avoid relying on native uint64
// (Slang lacks portable uint64 atomics / arithmetic).
//
// Per-pass structure (8 passes, one 8-bit digit each):
//   blockSort        — 256 threads/wg, 256 items/wg. 8 sub-passes of
//                      stable 1-bit splits in LDS. Writes block-sorted
//                      data back in place, plus per-block histogram
//                      (block_hist) and per-(block,digit) exclusive
//                      scan within block (block_start).
//   globalScan       — 256 threads (one per digit). Each walks all
//                      blocks for its digit, writes per-(block,digit)
//                      global prefix and the digit's total count.
//   scanDigitTotals  — single wg, exclusive scan of 256 digit totals
//                      to get each digit's global base offset.
//   scatterFinal     — N threads. O(1) per element: final_pos =
//                      digit_global_base[d] + block_prefix[gid,d]
//                      + (lid - block_start[gid,d]).
//
// Stability: each 1-bit split uses an exclusive scan of the flag
// (bit==0 ? 1 : 0). Items with bit=0 keep their input order; items
// with bit=1 keep their input order. Stable by construction. Over
// 8 sub-passes the block is sorted by 8-bit digit, stable.
//
// Sentinel handling: threads past ``n_buf[0]`` load 0xFFFFFFFF keys
// (digit 0xFF). These participate in the LDS sort (so the 256-wide
// scans remain uniform) but are *not counted* in block_hist and are
// *not written* back by blockSort or scatterFinal. Because sentinels
// hit digit 0xFF and stable sort preserves their original (end-of-
// block) position within that digit, they end up strictly after all
// real items in each block's sorted layout. scatter therefore writes
// real items into positions [0, n) in the destination buffer.
//
// Previous implementation (one-thread-per-bin global scatter looping
// over N) cost O(256·N) global reads per pass and was 98% of the
// training step's GPU time. This replacement is O(N·log N_block)
// mostly in LDS; measured ~20× speedup on cactus.

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

    // 8 stable 1-bit splits. Each iteration computes the flag
    // (bit == 0) per thread, exclusive-scans it over the block to get
    // each thread's rank among zero-bits, then reshuffles LDS so all
    // zero-bits come first (in input order), followed by one-bits (also
    // in input order).
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

    // LDS is now block-sorted by 8-bit digit. Write real items back.
    uint klo_s = gs_klo[lid];
    uint khi_s = gs_khi[lid];
    uint val_s = gs_val[lid];

    // Because sentinels all have digit 0xFF and stable sort preserved
    // their original (per-block tail) position within that digit, real
    // items now occupy positions [0, real_in_block) in this block's
    // sorted LDS. See the file-level comment.
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
        uint dummy;
        InterlockedAdd(gs_hist[d_s], 1u, dummy);
    }
    GroupMemoryBarrierWithGroupSync();

    // Exclusive scan of gs_hist -> block_start (position of digit d's
    // real items within this block's sorted layout).
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

// Per-digit serial scan across all active blocks. 256 threads, one per
// digit. Each thread walks ``ceil(n / SORT_BLOCK_SIZE)`` entries
// (block_hist[b * 256 + d]), writes per-(block, digit) running prefix
// into block_prefix[], and writes the per-digit total to digit_total[].
//
// Reading ``n`` from ``n_buf`` means we skip trailing all-sentinel
// blocks (n_pairs << max_sort_pairs is the common case: 50k pairs vs
// a 4M cap means 15k of 15625 blocks are pure sentinel). Walking only
// the active blocks drops this from O(max_blocks) to O(n/256).
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

// Single-workgroup exclusive scan of the 256 digit_total values into
// digit_global_base. Identical in spirit to the old scanHistogram but
// separated out so its intent is obvious.
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

// Final scatter. One thread per element; O(1) per thread.
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
