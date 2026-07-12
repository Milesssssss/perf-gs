// Device-level LSD radix sort ported from Thomas Smith's GPUSorting
// (b0nes164/GPUSorting, MIT). Reduce-then-scan variant (Upsweep -> Scan ->
// Downsweep), no cross-workgroup spin loops or forward-progress requirement.
//
// This is the WGE16 (wave lane count >= 16), ascending-only path,
// generalized from the reference's 32-bit key + optional 32-bit payload
// to a 3-word record (keys_lo, keys_hi, values) so the existing pipeline
// contract holds: 64-bit key stored as (hi:lo) uint32 pair + uint32
// gaussian-index payload, 8 passes (passes 0-3 digit from keys_lo, passes
// 4-7 from keys_hi).
//
// Kernels:
//   initGlobalHist  — clear b_global_hist[2048] once per full sort.
//   upsweep         — per PART_SIZE partition: histogram digit-source word,
//                     write per-(digit,partition) counts to b_pass_hist,
//                     atomic-add per-digit totals into b_global_hist.
//   scan            — one workgroup per digit; exclusive-scan
//                     b_pass_hist[digit * max_blocks + 0..active_blocks).
//   downsweep       — per partition: wave-multisplit rank digit-source word,
//                     scatter all 3 words to *_out using the same offsets.
//
// GPU-resident count: `n = n_buf[0]` (scan_total[0]). Buffers are sized to
// max capacity; inactive partitions early-out on n. Kernels are dispatched
// with a fixed group count derived from max_blocks; b_pass_hist stride is
// also max_blocks so indexing is uniform across kernels.
//
// Constraints:
//   RADIX = 256 (8-bit digit).
//   PART_SIZE = 2048 keys per partition.
//   Max supported n bounded by 1D dispatch limit at PART_SIZE resolution
//   (~134M keys with 65535 groups). Well above the pipeline's 8M cap.

#ifndef TRAINER_DEVICE_RADIX_SORT_SLANG
#define TRAINER_DEVICE_RADIX_SORT_SLANG

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const uint RADIX = 256u;
static const uint RADIX_MASK = 255u;
static const uint RADIX_LOG = 8u;

// Upsweep / Scan block sizes (from GPUSorting reference).
static const uint US_DIM = 128u;
static const uint SCAN_DIM = 128u;

// Downsweep block size and keys/thread. PART_SIZE = D_DIM * KEYS_PER_THREAD.
static const uint D_DIM = 256u;
static const uint KEYS_PER_THREAD = 8u;
static const uint PART_SIZE = D_DIM * KEYS_PER_THREAD;  // 2048

// Shared-memory size for the Downsweep. Sized as
//   max(waveHist@minLane, PART_SIZE + RADIX)
// waveHist = D_DIM / laneCount * RADIX. At lane=16 (WGE16 minimum): 4096.
// PART_SIZE + RADIX = 2304. Max = 4096. At lane=32 this leaves the tail
// [waveHist..end) unused, which is fine.
static const uint D_TOTAL_SMEM = 4096u;

// Digit source: pass_idx 0-3 use keys_lo, 4-7 use keys_hi. Encoded on host
// as a uniform `key_word` (0 or 1) plus `radix_shift` (0/8/16/24).
static const uint SENTINEL = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Groupshared memory
// ---------------------------------------------------------------------------
// Two histograms in Upsweep: threads [0..63] write hist 0, [64..127] write
// hist 1. Then reduced to one row in the write-back step.
groupshared uint g_us[RADIX * 2];

// SCAN_DIM staging for the per-digit scan.
groupshared uint g_scan[SCAN_DIM];

// Downsweep's shared memory. Layout during the pass:
//   [0, waveHistSize)          -> wave-local histograms (during ranking)
//   [0, PART_SIZE)             -> staged keys (after ScatterKeysShared)
//   [PART_SIZE, PART_SIZE+RADIX) -> device scatter offsets (LoadThreadBlockReductions)
groupshared uint g_d[D_TOTAL_SMEM];

struct KeyStruct { uint k[KEYS_PER_THREAD]; };
struct OffsetStruct { uint o[KEYS_PER_THREAD]; };
struct DigitStruct { uint d[KEYS_PER_THREAD]; };

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
uint activeBlocksOf(uint n) { return (n + PART_SIZE - 1u) / PART_SIZE; }

uint loadDigitSourceWord(uint idx, uint key_word,
                         StructuredBuffer<uint> keys_lo,
                         StructuredBuffer<uint> keys_hi)
{
    return (key_word == 0u) ? keys_lo[idx] : keys_hi[idx];
}

uint extractDigit(uint word, uint radix_shift)
{
    return (word >> radix_shift) & RADIX_MASK;
}

uint getWaveIndex(uint gtid)
{
    return gtid / WaveGetLaneCount();
}

uint waveHistsSize()
{
    return (D_DIM / WaveGetLaneCount()) * RADIX;
}

// ===========================================================================
// InitGlobalHist — clears b_global_hist[2048]
// ===========================================================================
// One thread per entry. Dispatched with thread_count = [2048, 1, 1].
[shader("compute")]
[numthreads(256, 1, 1)]
void initGlobalHist(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<uint> b_global_hist)
{
    if (dtid.x < RADIX * 8u)
        b_global_hist[dtid.x] = 0u;
}

// ===========================================================================
// PrepareDispatchArgs — writes indirect-dispatch args for upsweep/downsweep
// ===========================================================================
// Single-thread kernel. Reads n_buf[0], computes active_blocks, and writes
// two DispatchIndirectCommand triples (each: groups_x, groups_y, groups_z):
//   args[0..2] = (active_blocks, 1, 1)   — upsweep + downsweep group count
//   (scan is dispatched normally with 256 groups.)
[shader("compute")]
[numthreads(1, 1, 1)]
void prepareDispatchArgs(
    RWStructuredBuffer<uint> n_buf,
    RWStructuredBuffer<uint> dispatch_args,
    uniform uint max_blocks,     // defensive cap: never launch more groups than we have scratch for
    uniform uint max_sort_pairs) // capacity of the sort-pair buffers
{
    // Clamp the pair count to the sort-buffer capacity, IN PLACE, so every
    // downstream reader (upsweep/downsweep/identifyTileRanges) sees a value
    // that never exceeds what bin_emit actually wrote. bin_emit drops pairs
    // past max_sort_pairs (graceful overflow), so the valid written region is
    // min(N, max_sort_pairs); without this clamp the downsweep's last active
    // block would set partition_end = N > capacity and READ past the buffer.
    // This is the single choke point: prepareDispatchArgs runs once, before
    // any n_buf consumer in the sort and before tile-range identification.
    uint n = min(n_buf[0], max_sort_pairs);
    n_buf[0] = n;

    uint active_blocks = (n + PART_SIZE - 1u) / PART_SIZE;
    // Vulkan/D3D12 require at least (1,1,1) even for zero-work dispatches
    // (else the driver may error). If active_blocks==0, launch 1 group; the
    // in-shader early-out on (g >= active_blocks) makes it a no-op.
    uint g = (active_blocks == 0u) ? 1u : active_blocks;
    // Clamp: if n is somehow corrupted, don't queue billions of workgroups.
    g = min(g, max_blocks);
    dispatch_args[0] = g;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}

// ===========================================================================
// Upsweep — per-partition digit histogram
// ===========================================================================
//
// Steps (mirrors GPUSorting DeviceRadixSort.hlsl Upsweep, WGE16 path):
//   1. Zero g_us[0..2*RADIX).
//   2. Histogram digit-source word into two shared histograms (threads
//      0..63 -> hist 0, threads 64..127 -> hist 1), using InterlockedAdd
//      on groupshared.
//   3. Merge (hist0 += hist1) and write per-(digit, partition) counts to
//      b_pass_hist[digit * max_blocks + gid]. Then start a prefix sum for
//      the atomic-add-into-global step.
//   4. Exclusive scan across all 256 digits, atomically added into
//      b_global_hist[digit + pass_idx * RADIX].
//
// Inactive partitions (gid >= active_blocks) write 0 into b_pass_hist and
// return without touching b_global_hist. This keeps Scan's exclusive scan
// exact using the fixed-stride layout.
[shader("compute")]
[numthreads(US_DIM, 1, 1)]
void upsweep(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<uint> keys_lo,
    StructuredBuffer<uint> keys_hi,
    RWStructuredBuffer<uint> b_pass_hist,
    RWStructuredBuffer<uint> b_global_hist,
    StructuredBuffer<uint> n_buf,
    uniform uint radix_shift,
    uniform uint key_word,
    uniform uint pass_idx,
    uniform uint max_blocks)
{
    uint n = n_buf[0];
    uint active_blocks = activeBlocksOf(n);
    uint g = gid.x;
    uint t = gtid.x;

    // Inactive partition: zero this column of b_pass_hist and return.
    if (g >= active_blocks)
    {
        for (uint i = t; i < RADIX; i += US_DIM)
            b_pass_hist[i * max_blocks + g] = 0u;
        return;
    }

    // Clear the two shared histograms.
    for (uint i = t; i < RADIX * 2u; i += US_DIM)
        g_us[i] = 0u;
    GroupMemoryBarrierWithGroupSync();

    // Histogram: two histograms in parallel to reduce shared atomic pressure.
    // Threads 0..63 accumulate into hist 0, threads 64..127 into hist 1.
    uint partition_start = g * PART_SIZE;
    uint partition_end = (g == active_blocks - 1u) ? n : partition_start + PART_SIZE;
    uint hist_offset = (t / 64u) * RADIX;
    for (uint i = partition_start + t; i < partition_end; i += US_DIM)
    {
        uint word = loadDigitSourceWord(i, key_word, keys_lo, keys_hi);
        uint dummy;
        InterlockedAdd(g_us[extractDigit(word, radix_shift) + hist_offset], 1u, dummy);
    }
    GroupMemoryBarrierWithGroupSync();

    // Merge (hist0 += hist1), write per-(digit,partition) counts to b_pass_hist,
    // then begin a running exclusive prefix sum in g_us (still holding the
    // pre-scan hist0 values -- we upgrade to a full scan below).
    for (uint i = t; i < RADIX; i += US_DIM)
    {
        g_us[i] += g_us[i + RADIX];
        b_pass_hist[i * max_blocks + g] = g_us[i];
        g_us[i] += WavePrefixSum(g_us[i]);  // inclusive within the wave
    }

    // GlobalHistExclusiveScanWGE16: propagate the wave-level scans into a
    // full 256-wide exclusive prefix, then atomic-add to b_global_hist in
    // the pass's own [pass_idx * RADIX, (pass_idx+1) * RADIX) region.
    GroupMemoryBarrierWithGroupSync();
    uint lane_count = WaveGetLaneCount();
    if (t < (RADIX / lane_count))
    {
        g_us[(t + 1u) * lane_count - 1u] +=
            WavePrefixSum(g_us[(t + 1u) * lane_count - 1u]);
    }
    GroupMemoryBarrierWithGroupSync();

    uint global_hist_offset = pass_idx * RADIX;
    uint lane_mask = lane_count - 1u;
    uint circular_lane_shift = (WaveGetLaneIndex() + 1u) & lane_mask;
    for (uint i = t; i < RADIX; i += US_DIM)
    {
        uint index = circular_lane_shift + (i & ~lane_mask);
        uint add = (WaveGetLaneIndex() != lane_mask ? g_us[i] : 0u)
                 + (i >= lane_count ? WaveReadLaneAt(g_us[i - 1u], 0u) : 0u);
        uint dummy;
        InterlockedAdd(b_global_hist[index + global_hist_offset], add, dummy);
    }
}

// ===========================================================================
// Scan — exclusive scan of b_pass_hist across active partitions per digit
// ===========================================================================
//
// One workgroup per digit (dispatch 256 groups). SCAN_DIM threads. Uses the
// GPUSorting circular-lane-shift trick to write scanned values back one lane
// shifted so consecutive lanes' writes don't collide.
//
// b_pass_hist layout: fixed stride = max_blocks. We only scan the first
// `active_blocks` entries; the rest were zeroed by Upsweep.
[shader("compute")]
[numthreads(SCAN_DIM, 1, 1)]
void scan(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    RWStructuredBuffer<uint> b_pass_hist,
    StructuredBuffer<uint> n_buf,
    uniform uint max_blocks)
{
    uint n = n_buf[0];
    uint active_blocks = activeBlocksOf(n);
    uint digit = gid.x;
    uint t = gtid.x;

    uint device_offset = digit * max_blocks;
    uint lane_count = WaveGetLaneCount();
    uint lane_mask = lane_count - 1u;
    uint circular_lane_shift = (WaveGetLaneIndex() + 1u) & lane_mask;

    // Number of full SCAN_DIM chunks over active_blocks.
    uint parts_end = (active_blocks / SCAN_DIM) * SCAN_DIM;
    uint reduction = 0u;

    // Full chunks.
    for (uint i = t; i < parts_end; i += SCAN_DIM)
    {
        g_scan[t] = b_pass_hist[i + device_offset];
        g_scan[t] += WavePrefixSum(g_scan[t]);
        GroupMemoryBarrierWithGroupSync();

        if (t < SCAN_DIM / lane_count)
        {
            g_scan[(t + 1u) * lane_count - 1u] +=
                WavePrefixSum(g_scan[(t + 1u) * lane_count - 1u]);
        }
        GroupMemoryBarrierWithGroupSync();

        b_pass_hist[circular_lane_shift + (i & ~lane_mask) + device_offset] =
            (WaveGetLaneIndex() != lane_mask ? g_scan[t] : 0u)
            + (t >= lane_count ? WaveReadLaneAt(g_scan[t - 1u], 0u) : 0u)
            + reduction;

        reduction += g_scan[SCAN_DIM - 1u];
        GroupMemoryBarrierWithGroupSync();
    }

    // Partial chunk: the last (< SCAN_DIM) active blocks.
    uint i = t + parts_end;
    if (i < active_blocks)
        g_scan[t] = b_pass_hist[device_offset + i];
    else
        g_scan[t] = 0u;
    g_scan[t] += WavePrefixSum(g_scan[t]);
    GroupMemoryBarrierWithGroupSync();

    if (t < SCAN_DIM / lane_count)
    {
        g_scan[(t + 1u) * lane_count - 1u] +=
            WavePrefixSum(g_scan[(t + 1u) * lane_count - 1u]);
    }
    GroupMemoryBarrierWithGroupSync();

    uint index = circular_lane_shift + (i & ~lane_mask);
    if (index < active_blocks)
    {
        b_pass_hist[index + device_offset] =
            (WaveGetLaneIndex() != lane_mask ? g_scan[t] : 0u)
            + (t >= lane_count ? g_scan[(t & ~lane_mask) - 1u] : 0u)
            + reduction;
    }
}

// ===========================================================================
// Downsweep — wave-multisplit rank + scatter (3-word record)
// ===========================================================================
//
// Ports the WGE16 path from SortCommon.hlsl. Generalized to a 3-word record:
// the digit is extracted from a single word (`digit_source_word`, selected
// by `key_word` uniform), and all three words (keys_lo, keys_hi, values)
// are scattered using the same ranking offsets.

// WarpLevelMultiSplit: for a given key, find the lanes in the wave that
// share its 8-bit digit at radix_shift. `waveFlags` is a per-part bitmask;
// iteratively AND with (t ? 0 : ~0) ^ ballot(bit==1) across the 8 bits of
// the digit until only equal-digit peers remain.
void warpLevelMultiSplitWGE16(uint key, uint radix_shift, uint wave_parts,
                              inout uint4 waveFlags)
{
    [unroll]
    for (uint k = 0u; k < RADIX_LOG; ++k)
    {
        bool t_bit = ((key >> (k + radix_shift)) & 1u) != 0u;
        uint4 ballot = WaveActiveBallot(t_bit);
        uint mask_all = t_bit ? 0u : 0xFFFFFFFFu;
        for (uint wp = 0u; wp < wave_parts; ++wp)
            waveFlags[wp] &= mask_all ^ ballot[wp];
    }
}

uint waveFlagsWGE16()
{
    return ((WaveGetLaneCount() & 31u) != 0u) ? ((1u << WaveGetLaneCount()) - 1u) : 0xFFFFFFFFu;
}

void countPeerBits(inout uint peerBits, inout uint totalBits,
                   uint4 waveFlags, uint wave_parts)
{
    for (uint wp = 0u; wp < wave_parts; ++wp)
    {
        if (WaveGetLaneIndex() >= wp * 32u)
        {
            uint ltMask = (WaveGetLaneIndex() >= (wp + 1u) * 32u)
                ? 0xFFFFFFFFu
                : ((1u << (WaveGetLaneIndex() & 31u)) - 1u);
            peerBits += countbits(waveFlags[wp] & ltMask);
        }
        totalBits += countbits(waveFlags[wp]);
    }
}

uint findLowestRankPeer(uint4 waveFlags, uint wave_parts)
{
    uint lowest = 0u;
    for (uint wp = 0u; wp < wave_parts; ++wp)
    {
        uint fbl = firstbitlow(waveFlags[wp]);
        if (fbl == 0xFFFFFFFFu)
            lowest += 32u;
        else
            return lowest + fbl;
    }
    return 0u;  // unreachable
}

// Rank keys within the workgroup. For each key: find equal-digit wave peers,
// count how many lanes with lower index share this digit (= intra-wave rank),
// have the lowest-ranked peer atomically bump the wave-histogram bucket.
// Returns offsets in [0, wave_hist_size) that are position-within-wave for
// this key's digit; UpdateOffsetsWGE16 later adds the cross-wave device
// offsets to make them absolute.
OffsetStruct rankKeysWGE16(uint gtid, KeyStruct keys, uint radix_shift)
{
    OffsetStruct offsets;
    uint wave_parts = (WaveGetLaneCount() + 31u) / 32u;
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
    {
        uint4 waveFlags;
        waveFlags.x = waveFlagsWGE16();
        waveFlags.y = 0u; waveFlags.z = 0u; waveFlags.w = 0u;
        // If wave has more than 32 lanes (e.g. AMD RDNA 64) also fill next parts.
        if (wave_parts > 1u) waveFlags.y = 0xFFFFFFFFu;
        if (wave_parts > 2u) waveFlags.z = 0xFFFFFFFFu;
        if (wave_parts > 3u) waveFlags.w = 0xFFFFFFFFu;

        warpLevelMultiSplitWGE16(keys.k[i], radix_shift, wave_parts, waveFlags);

        uint index = extractDigit(keys.k[i], radix_shift) + getWaveIndex(gtid) * RADIX;
        uint lowest = findLowestRankPeer(waveFlags, wave_parts);

        uint peerBits = 0u;
        uint totalBits = 0u;
        countPeerBits(peerBits, totalBits, waveFlags, wave_parts);

        uint preInc = 0u;
        if (peerBits == 0u)
            InterlockedAdd(g_d[index], totalBits, preInc);
        offsets.o[i] = WaveReadLaneAt(preInc, lowest) + peerBits;
    }
    return offsets;
}

// After ranking, g_d[0..waveHistSize) holds each wave's per-digit hist
// counts (contiguous by wave). This function does a circular-shift
// inclusive scan across waves for each digit (so g_d[digit + w*RADIX] ends
// up holding the sum of hist[digit] over waves [0..w) after the exclusive
// scan step). Returns the total for this digit across all waves.
uint waveHistInclusiveScanCircularShiftWGE16(uint gtid)
{
    uint histReduction = g_d[gtid];
    for (uint i = gtid + RADIX; i < waveHistsSize(); i += RADIX)
    {
        histReduction += g_d[i];
        g_d[i] = histReduction - g_d[i];  // exclusive scan across waves at fixed digit
    }
    return histReduction;
}

// Given per-digit reductions (one value per lane), form an exclusive scan
// across 256 digits using the circular-lane-shift pattern.
void waveHistReductionExclusiveScanWGE16(uint gtid, uint histReduction)
{
    if (gtid < RADIX)
    {
        uint lane_mask = WaveGetLaneCount() - 1u;
        g_d[((WaveGetLaneIndex() + 1u) & lane_mask) + (gtid & ~lane_mask)] = histReduction;
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid < RADIX / WaveGetLaneCount())
    {
        g_d[gtid * WaveGetLaneCount()] = WavePrefixSum(g_d[gtid * WaveGetLaneCount()]);
    }
    GroupMemoryBarrierWithGroupSync();

    if (gtid < RADIX && WaveGetLaneIndex() != 0u)
        g_d[gtid] += WaveReadLaneAt(g_d[gtid - 1u], 1u);
}

// Add cross-wave device offsets to per-key intra-wave offsets so offsets.o[i]
// becomes a position in [0, PART_SIZE) staging.
void updateOffsetsWGE16(uint gtid, inout OffsetStruct offsets, KeyStruct keys,
                        uint radix_shift)
{
    if (gtid >= WaveGetLaneCount())
    {
        uint tw = getWaveIndex(gtid) * RADIX;
        [unroll]
        for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
        {
            uint d = extractDigit(keys.k[i], radix_shift);
            offsets.o[i] += g_d[d + tw] + g_d[d];
        }
    }
    else
    {
        [unroll]
        for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
            offsets.o[i] += g_d[extractDigit(keys.k[i], radix_shift)];
    }
}

// After ranking, offsets.o[i] identifies a position in [0, PART_SIZE) for
// each of this thread's KEYS_PER_THREAD keys. Stage into g_d.
void scatterKeysShared(OffsetStruct offsets, KeyStruct keys)
{
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
        g_d[offsets.o[i]] = keys.k[i];
}

// Read the "digit-source" word at index idx: keys_lo if key_word==0 else keys_hi.
uint readWord(uint idx, uint key_word,
              StructuredBuffer<uint> keys_lo, StructuredBuffer<uint> keys_hi)
{
    return (key_word == 0u) ? keys_lo[idx] : keys_hi[idx];
}

// Load KEYS_PER_THREAD words from (keys_lo | keys_hi) selected by key_word.
KeyStruct loadKeysWGE16Sel(uint gtid, uint part_index, uint key_word,
                           StructuredBuffer<uint> keys_lo,
                           StructuredBuffer<uint> keys_hi)
{
    KeyStruct keys;
    uint base = WaveGetLaneIndex() + getWaveIndex(gtid) * KEYS_PER_THREAD * WaveGetLaneCount()
              + part_index * PART_SIZE;
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
    {
        keys.k[i] = readWord(base + i * WaveGetLaneCount(), key_word, keys_lo, keys_hi);
    }
    return keys;
}

KeyStruct loadKeysPartialWGE16Sel(uint gtid, uint part_index, uint key_word,
                                  StructuredBuffer<uint> keys_lo,
                                  StructuredBuffer<uint> keys_hi,
                                  uint n_bound)
{
    KeyStruct keys;
    uint base = WaveGetLaneIndex() + getWaveIndex(gtid) * KEYS_PER_THREAD * WaveGetLaneCount()
              + part_index * PART_SIZE;
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
    {
        uint idx = base + i * WaveGetLaneCount();
        keys.k[i] = (idx < n_bound) ? readWord(idx, key_word, keys_lo, keys_hi) : SENTINEL;
    }
    return keys;
}

// Single-buffer variants for the value payload.
KeyStruct loadKeysWGE16(uint gtid, uint part_index, StructuredBuffer<uint> buf)
{
    KeyStruct keys;
    uint base = WaveGetLaneIndex() + getWaveIndex(gtid) * KEYS_PER_THREAD * WaveGetLaneCount()
              + part_index * PART_SIZE;
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
        keys.k[i] = buf[base + i * WaveGetLaneCount()];
    return keys;
}

KeyStruct loadKeysPartialWGE16(uint gtid, uint part_index,
                               StructuredBuffer<uint> buf, uint n_bound)
{
    KeyStruct keys;
    uint base = WaveGetLaneIndex() + getWaveIndex(gtid) * KEYS_PER_THREAD * WaveGetLaneCount()
              + part_index * PART_SIZE;
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
    {
        uint idx = base + i * WaveGetLaneCount();
        keys.k[i] = (idx < n_bound) ? buf[idx] : SENTINEL;
    }
    return keys;
}

// Load device scatter offsets (per digit) into g_d[PART_SIZE + digit].
void loadThreadBlockReductions(uint gtid, uint part_index,
                                RWStructuredBuffer<uint> b_pass_hist,
                                RWStructuredBuffer<uint> b_global_hist,
                                uint pass_idx, uint max_blocks,
                                uint exclusiveHistReduction)
{
    if (gtid < RADIX)
    {
        g_d[gtid + PART_SIZE] =
            b_global_hist[gtid + pass_idx * RADIX]
            + b_pass_hist[gtid * max_blocks + part_index]
            - exclusiveHistReduction;
    }
}

// ---- 3-word scatter (ascending) ----
//
// Extract digit from staged key word in g_d[t] and cache it in digits.d[i].
// Write the key word to the correct output buffer.
void scatterKeyPhaseOne(uint gtid, uint radix_shift,
                        RWStructuredBuffer<uint> key_out,
                        inout DigitStruct digits)
{
    [unroll]
    for (uint i = 0u, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
    {
        uint word = g_d[t];
        uint d = extractDigit(word, radix_shift);
        digits.d[i] = d;
        uint dev_off = g_d[d + PART_SIZE] + t;
        key_out[dev_off] = word;
    }
}

void scatterKeyPhasePartialOne(uint gtid, uint radix_shift, uint final_size,
                               RWStructuredBuffer<uint> key_out,
                               inout DigitStruct digits)
{
    [unroll]
    for (uint i = 0u, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
    {
        if (t < final_size)
        {
            uint word = g_d[t];
            uint d = extractDigit(word, radix_shift);
            digits.d[i] = d;
            uint dev_off = g_d[d + PART_SIZE] + t;
            key_out[dev_off] = word;
        }
    }
}

// key_word-selected wrappers.
void scatterKeyPhase(uint gtid, uint radix_shift, uint key_word,
                     RWStructuredBuffer<uint> keys_lo_out,
                     RWStructuredBuffer<uint> keys_hi_out,
                     inout DigitStruct digits)
{
    if (key_word == 0u)
        scatterKeyPhaseOne(gtid, radix_shift, keys_lo_out, digits);
    else
        scatterKeyPhaseOne(gtid, radix_shift, keys_hi_out, digits);
}

void scatterKeyPhasePartial(uint gtid, uint radix_shift, uint key_word,
                            uint final_size,
                            RWStructuredBuffer<uint> keys_lo_out,
                            RWStructuredBuffer<uint> keys_hi_out,
                            inout DigitStruct digits)
{
    if (key_word == 0u)
        scatterKeyPhasePartialOne(gtid, radix_shift, final_size, keys_lo_out, digits);
    else
        scatterKeyPhasePartialOne(gtid, radix_shift, final_size, keys_hi_out, digits);
}

// Load payload word into g_d[offsets.o[i]] using the ranking offsets.
void scatterPayloadShared(OffsetStruct offsets, KeyStruct payloads)
{
    [unroll]
    for (uint i = 0u; i < KEYS_PER_THREAD; ++i)
        g_d[offsets.o[i]] = payloads.k[i];
}

// Scatter the payload word to its output buffer using cached digits.
void scatterPayloadDevice(uint gtid, DigitStruct digits,
                          RWStructuredBuffer<uint> out_buf)
{
    [unroll]
    for (uint i = 0u, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
    {
        uint dev_off = g_d[digits.d[i] + PART_SIZE] + t;
        out_buf[dev_off] = g_d[t];
    }
}

void scatterPayloadDevicePartial(uint gtid, uint final_size, DigitStruct digits,
                                 RWStructuredBuffer<uint> out_buf)
{
    [unroll]
    for (uint i = 0u, t = gtid; i < KEYS_PER_THREAD; ++i, t += D_DIM)
    {
        if (t < final_size)
        {
            uint dev_off = g_d[digits.d[i] + PART_SIZE] + t;
            out_buf[dev_off] = g_d[t];
        }
    }
}

// key_word-selected variants: DXC requires the target buffer be statically
// resolvable at the write site. We branch on key_word once around the whole
// scatter so both branches are pure-buffer writes.
void scatterPayloadDeviceSel(uint gtid, DigitStruct digits, uint key_word,
                             RWStructuredBuffer<uint> keys_lo_out,
                             RWStructuredBuffer<uint> keys_hi_out)
{
    if (key_word == 0u)
        scatterPayloadDevice(gtid, digits, keys_hi_out);   // other = hi
    else
        scatterPayloadDevice(gtid, digits, keys_lo_out);   // other = lo
}

void scatterPayloadDevicePartialSel(uint gtid, uint final_size, DigitStruct digits,
                                    uint key_word,
                                    RWStructuredBuffer<uint> keys_lo_out,
                                    RWStructuredBuffer<uint> keys_hi_out)
{
    if (key_word == 0u)
        scatterPayloadDevicePartial(gtid, final_size, digits, keys_hi_out);
    else
        scatterPayloadDevicePartial(gtid, final_size, digits, keys_lo_out);
}

// ---- The Downsweep entry point ----
[shader("compute")]
[numthreads(D_DIM, 1, 1)]
void downsweep(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<uint> keys_lo_in,
    StructuredBuffer<uint> keys_hi_in,
    StructuredBuffer<uint> values_in,
    RWStructuredBuffer<uint> keys_lo_out,
    RWStructuredBuffer<uint> keys_hi_out,
    RWStructuredBuffer<uint> values_out,
    RWStructuredBuffer<uint> b_pass_hist,
    RWStructuredBuffer<uint> b_global_hist,
    StructuredBuffer<uint> n_buf,
    uniform uint radix_shift,
    uniform uint key_word,
    uniform uint pass_idx,
    uniform uint max_blocks)
{
    uint n = n_buf[0];
    uint active_blocks = activeBlocksOf(n);
    uint g = gid.x;
    uint t = gtid.x;

    if (g >= active_blocks) return;

    KeyStruct keys;
    OffsetStruct offsets;
    DigitStruct digits;

    // 1. Clear wave histograms in shared memory.
    uint hist_end = waveHistsSize();
    for (uint i = t; i < hist_end; i += D_DIM)
        g_d[i] = 0u;

    // 2. Load digit-source-word keys. Last active partition may be partial.
    bool last = (g == active_blocks - 1u);
    if (!last)
        keys = loadKeysWGE16Sel(t, g, key_word, keys_lo_in, keys_hi_in);
    else
        keys = loadKeysPartialWGE16Sel(t, g, key_word, keys_lo_in, keys_hi_in, n);

    GroupMemoryBarrierWithGroupSync();

    // 3. Rank within the block using wave multi-split.
    offsets = rankKeysWGE16(t, keys, radix_shift);
    GroupMemoryBarrierWithGroupSync();

    // 4. Wave-histogram inclusive-scan (circular shift) + reduce/exclusive-scan
    //    across digits.
    uint histReduction;
    if (t < RADIX)
    {
        histReduction = waveHistInclusiveScanCircularShiftWGE16(t);
        histReduction += WavePrefixSum(histReduction);
    }
    GroupMemoryBarrierWithGroupSync();

    waveHistReductionExclusiveScanWGE16(t, histReduction);
    GroupMemoryBarrierWithGroupSync();

    // 5. Add device offsets to per-key intra-wave offsets.
    updateOffsetsWGE16(t, offsets, keys, radix_shift);

    uint exclusiveHistReduction = 0u;
    if (t < RADIX)
        exclusiveHistReduction = g_d[t];  // wave-0's original hist[digit=t] row
    GroupMemoryBarrierWithGroupSync();

    // 6. Stage the digit-source-word into g_d[0..PART_SIZE).
    scatterKeysShared(offsets, keys);

    // 7. Load per-digit device scatter offsets into g_d[PART_SIZE + digit].
    loadThreadBlockReductions(t, g, b_pass_hist, b_global_hist,
                              pass_idx, max_blocks, exclusiveHistReduction);
    GroupMemoryBarrierWithGroupSync();

    // 8. Key phase: extract digit from g_d[t], write to the digit-source
    //    output buffer, cache digits.
    if (!last)
    {
        scatterKeyPhase(t, radix_shift, key_word,
                        keys_lo_out, keys_hi_out, digits);
    }
    else
    {
        uint final_size = n - g * PART_SIZE;
        scatterKeyPhasePartial(t, radix_shift, key_word, final_size,
                               keys_lo_out, keys_hi_out, digits);
    }
    GroupMemoryBarrierWithGroupSync();

    // 9. Payload phase 1: the OTHER key word (hi if key_word==0 else lo).
    //    Load via the key_word-selected loader with the arguments swapped so
    //    we read the "other" word.
    KeyStruct otherKey;
    uint other_key_word = 1u - key_word;
    if (!last)
        otherKey = loadKeysWGE16Sel(t, g, other_key_word, keys_lo_in, keys_hi_in);
    else
        otherKey = loadKeysPartialWGE16Sel(t, g, other_key_word, keys_lo_in, keys_hi_in, n);
    scatterPayloadShared(offsets, otherKey);
    GroupMemoryBarrierWithGroupSync();

    if (!last)
        scatterPayloadDeviceSel(t, digits, key_word, keys_lo_out, keys_hi_out);
    else
    {
        uint final_size = n - g * PART_SIZE;
        scatterPayloadDevicePartialSel(t, final_size, digits, key_word, keys_lo_out, keys_hi_out);
    }
    GroupMemoryBarrierWithGroupSync();

    // 10. Payload phase 2: the value word.
    KeyStruct valuePayload;
    if (!last)
        valuePayload = loadKeysWGE16(t, g, values_in);
    else
        valuePayload = loadKeysPartialWGE16(t, g, values_in, n);
    scatterPayloadShared(offsets, valuePayload);
    GroupMemoryBarrierWithGroupSync();

    if (!last)
        scatterPayloadDevice(t, digits, values_out);
    else
    {
        uint final_size = n - g * PART_SIZE;
        scatterPayloadDevicePartial(t, final_size, digits, values_out);
    }
}

#endif // TRAINER_DEVICE_RADIX_SORT_SLANG
