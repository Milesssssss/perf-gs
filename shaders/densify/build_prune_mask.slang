// Phase C of densification: build a valid-mask over the POST-split gaussian
// range (i.e. including children appended in Phase B).
//
// valid[i] = 1 if this gaussian should survive the compaction pass.
//          = 0 if it should be dropped (or if i >= post_count — beyond
//              the live range, so the subsequent max_splats-wide scan
//              contributes 0 for those slots).
//
// Rules for i < post_count:
// - i >= pre_count  -> newly-appended child, always valid.
// - i <  pre_count and action[i] == PRUNE -> drop.
// - otherwise -> keep (includes KEEP, SPLIT (slot was overwritten with
//   child-1), and CLONE (parent survives)).
//
// Dispatched over [max_splats] so we can also zero the tail
// [post_count, max_splats) without a separate memset — this lets
// scatterCompact and _scan both run at max_splats too, keeping the
// full densify_simple wait-free path free of CPU readbacks.

static const uint ACTION_KEEP  = 0;
static const uint ACTION_SPLIT = 1;
static const uint ACTION_CLONE = 2;
static const uint ACTION_PRUNE = 3;

[shader("compute")]
[numthreads(256, 1, 1)]
void buildPruneMask(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> action,
    RWStructuredBuffer<uint> valid_mask,
    // Snapshots taken before/after splitAndClone. See densify_dispatch.slang.
    StructuredBuffer<uint> pre_count_buf,
    StructuredBuffer<uint> post_count_buf,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;

    uint post_count = post_count_buf[0];
    if (i >= post_count)
    {
        // Zero the tail so the following max_splats-wide scan doesn't
        // pick up stale values from the previous frame.
        valid_mask[i] = 0u;
        return;
    }

    uint pre_count = pre_count_buf[0];
    if (i < pre_count && action[i] == ACTION_PRUNE)
    {
        valid_mask[i] = 0u;
    }
    else
    {
        valid_mask[i] = 1u;
    }
}
