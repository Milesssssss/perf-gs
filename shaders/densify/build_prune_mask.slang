// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Densify phase C: valid mask over the post-split range (children appended in phase B always survive);
// dispatched over max_splats so the tail is zeroed and the scan/scatter need no CPU readback.

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
        // Zero the tail so the max_splats-wide scan never sees stale values.
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
