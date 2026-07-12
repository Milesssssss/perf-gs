// Parallel exclusive prefix scan on a uint[] buffer.
//
// Three-kernel Blelloch scan. These SHADERS are level-agnostic: each
// scanBlock covers up to SCAN_BLOCK_SIZE elements per workgroup, and
// scanBlockSums covers up to SCAN_BLOCK_SIZE block_sums in ONE workgroup.
// The Python wrapper in pipeline._scan chains them into either a 2-level
// (n <= SCAN_BLOCK_SIZE^2 = 1,048,576) or 3-level (n <= SCAN_BLOCK_SIZE^3
// = ~1.07 B) scan by re-dispatching scanBlock+scanBlockSums on the
// block_sums buffer when num_blocks > SCAN_BLOCK_SIZE. Adequate for
// max_splats up to ~1 B; beyond that a 4-level chain would be required.
//
// Exclusive semantics: out[i] = sum(in[0..i-1]), out[0] = 0.
// Also returns the total sum so the caller can know the post-compaction
// element count without a separate readback of out[N-1] + in[N-1].
//
// Layout:
//   scanBlock: per-block scan, writes out[] + block_sums[]
//   scanBlockSums: single-workgroup scan of block_sums[] in place
//   addBlockOffsets: per-block add of block_sums[bid] to every out[] in bid

#ifndef TRAINER_SCAN_SLANG
#define TRAINER_SCAN_SLANG

static const uint SCAN_BLOCK_SIZE = 1024;
// scanBlockSums is a single-workgroup scan so it caps at SCAN_BLOCK_SIZE
// inputs. When the level-2 count exceeds this, pipeline._scan does an
// extra scanBlock+scanBlockSums pass (the 3-level branch).
static const uint SCAN_MAX_BLOCKS = 1024;

groupshared uint _scan_lds[SCAN_BLOCK_SIZE];

// Single-workgroup exclusive Hillis-Steele prefix scan operating on
// _scan_lds[0..count). Fills _scan_lds with exclusive prefix at return;
// the return value is the total sum (inclusive of count).
uint _scan_lds_inplace(uint lid, uint count)
{
    // Hillis-Steele: inclusive scan, stride doubles each step.
    for (uint offset = 1u; offset < count; offset <<= 1u)
    {
        GroupMemoryBarrierWithGroupSync();
        uint add_val = 0u;
        if (lid >= offset && lid < count) add_val = _scan_lds[lid - offset];
        GroupMemoryBarrierWithGroupSync();
        if (lid < count) _scan_lds[lid] += add_val;
    }
    GroupMemoryBarrierWithGroupSync();

    // Convert inclusive -> exclusive by shifting right 1.
    uint total = (count > 0u) ? _scan_lds[count - 1u] : 0u;
    GroupMemoryBarrierWithGroupSync();
    uint my_prev = (lid == 0u || lid >= count) ? 0u : _scan_lds[lid - 1u];
    GroupMemoryBarrierWithGroupSync();
    if (lid < count) _scan_lds[lid] = my_prev;
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[shader("compute")]
[numthreads(SCAN_BLOCK_SIZE, 1, 1)]
void scanBlock(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint lid: SV_GroupIndex,
    StructuredBuffer<uint> in_buf,
    RWStructuredBuffer<uint> out_buf,
    RWStructuredBuffer<uint> block_sums,
    uniform uint n)
{
    uint global_idx = gid.x * SCAN_BLOCK_SIZE + lid;
    uint block_start = gid.x * SCAN_BLOCK_SIZE;
    uint block_count = min(SCAN_BLOCK_SIZE, (n > block_start) ? (n - block_start) : 0u);

    _scan_lds[lid] = (global_idx < n) ? in_buf[global_idx] : 0u;

    uint total = _scan_lds_inplace(lid, block_count);

    if (global_idx < n) out_buf[global_idx] = _scan_lds[lid];
    if (lid == 0u) block_sums[gid.x] = total;
}

// Single workgroup. Scans block_sums[] in place for a scan of at most
// SCAN_MAX_BLOCKS elements. ``num_blocks`` is the actual count.
[shader("compute")]
[numthreads(SCAN_BLOCK_SIZE, 1, 1)]
void scanBlockSums(
    uint3 dtid: SV_DispatchThreadID,
    uint lid: SV_GroupIndex,
    RWStructuredBuffer<uint> block_sums,
    RWStructuredBuffer<uint> total_out,
    uniform uint num_blocks)
{
    _scan_lds[lid] = (lid < num_blocks) ? block_sums[lid] : 0u;
    uint total = _scan_lds_inplace(lid, num_blocks);
    if (lid < num_blocks) block_sums[lid] = _scan_lds[lid];
    if (lid == 0u) total_out[0] = total;
}

[shader("compute")]
[numthreads(SCAN_BLOCK_SIZE, 1, 1)]
void addBlockOffsets(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint lid: SV_GroupIndex,
    RWStructuredBuffer<uint> out_buf,
    StructuredBuffer<uint> block_sums,
    uniform uint n)
{
    uint global_idx = gid.x * SCAN_BLOCK_SIZE + lid;
    if (global_idx >= n) return;
    out_buf[global_idx] += block_sums[gid.x];
}

#endif // TRAINER_SCAN_SLANG
