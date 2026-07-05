// GPU-side helpers to keep densify_simple wait-free.
//
// The densify pipeline needs two counts that were previously read back
// to the CPU:
//   pre_count  = num_gaussians_counter[0] BEFORE splitAndClone runs
//   post_count = num_gaussians_counter[0] AFTER  splitAndClone runs
//
// pre_count is also needed as an immutable input to splitAndClone
// (which atomically bumps num_gaussians_counter[0] — so it can't read
// pre_count from the same buffer it writes to), and as the group count
// for the indirect dispatch. Snapshot buffers (pre_count_buf,
// post_count_buf) hold those values GPU-side.
//
// buildPruneMask and scatterCompact are dispatched over [max_splats]
// and self-bound off post_count_buf/valid_mask, so they need no indirect
// dispatch of their own. Only splitAndClone (which is O(pre_count) and
// pre_count varies widely) uses indirect dispatch.
//
// final_count = scan_total[0] after _scan; copied into
// num_gaussians_counter[0] via snapshotCount (src=scan_total, dst=counter).

// -----------------------------------------------------------------------
// snapshotCount — dst[0] = src[0]. Uses:
//   1) src=num_gaussians_counter, dst=pre_count_buf  (before splitAndClone)
//   2) src=num_gaussians_counter, dst=post_count_buf (after  splitAndClone)
//   3) src=scan_total,             dst=num_gaussians_counter (at end)
// -----------------------------------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void snapshotCount(
    StructuredBuffer<uint> src,
    RWStructuredBuffer<uint> dst)
{
    dst[0] = src[0];
}

// -----------------------------------------------------------------------
// prepareDensifyArgs — from a count buffer, write DispatchIndirect args
// (groups_x, 1, 1) with groups_x = ceil(count / 256). Vulkan/D3D12 need
// groups_x >= 1 to avoid errors on zero-work dispatches; the shader
// itself early-outs per-thread on i >= count.
//
// Used to size the splitAndClone indirect dispatch from pre_count_buf.
// -----------------------------------------------------------------------
[shader("compute")]
[numthreads(1, 1, 1)]
void prepareDensifyArgs(
    StructuredBuffer<uint> count_buf,
    RWStructuredBuffer<uint> dispatch_args,
    uniform uint max_splats)   // defensive cap
{
    uint n = count_buf[0];
    uint groups = (n + 255u) / 256u;
    if (groups == 0u) groups = 1u;
    // Clamp so a corrupted count_buf can never queue an insane number of groups.
    uint max_groups = (max_splats + 255u) / 256u;
    if (max_groups == 0u) max_groups = 1u;
    groups = min(groups, max_groups);
    dispatch_args[0] = groups;
    dispatch_args[1] = 1u;
    dispatch_args[2] = 1u;
}
