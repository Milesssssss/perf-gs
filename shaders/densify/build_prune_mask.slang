// Phase C of densification: build a valid-mask over the POST-split gaussian
// range (i.e. including children appended in Phase B).
//
// valid[i] = 1 if this gaussian should survive the compaction pass.
//          = 0 if it should be dropped.
//
// Rules:
// - i >= pre_count  -> newly-appended child, always valid.
// - i <  pre_count and action[i] == PRUNE -> drop.
// - otherwise -> keep (includes KEEP, SPLIT (slot was overwritten with
//   child-1), and CLONE (parent survives)).

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
    uniform uint pre_count,
    uniform uint post_count)
{
    uint i = dtid.x;
    if (i >= post_count) return;

    if (i < pre_count && action[i] == ACTION_PRUNE)
    {
        valid_mask[i] = 0u;
    }
    else
    {
        valid_mask[i] = 1u;
    }
}
