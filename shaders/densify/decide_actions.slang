// Phase A of densification: decide per-gaussian action using FastGS
// multi-view importance score with budget-limited growth AND VCP pruning.
//
// Decision rules:
//   importance = importance_score_accum[i] / n_scoring_views
//   max_scale  = max(exp(log_scale))
//
//   PRUNE if:
//     - max_scale > t_big_scale (oversized, unconditional)
//     - sigmoid(opacity) < t_opacity (budget-limited)
//     - importance > prune_thresh (VCP: multi-view consistent pruning,
//       gaussian is associated with high error across views, budget-limited)
//   SPLIT if:
//     - importance > importance_thresh AND max_scale > t_split_scale
//   CLONE if:
//     - importance > importance_thresh AND max_scale <= t_split_scale
//   KEEP otherwise

#include "../common/types.slang"
#include "../common/math.slang"

static const uint ACTION_KEEP  = 0;
static const uint ACTION_SPLIT = 1;
static const uint ACTION_CLONE = 2;
static const uint ACTION_PRUNE = 3;

[shader("compute")]
[numthreads(256, 1, 1)]
void decideActions(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<uint> importance_score_accum,
    RWStructuredBuffer<uint> action,
    RWStructuredBuffer<int> growth_budget,
    RWStructuredBuffer<int> prune_budget,
    uniform float importance_thresh,
    uniform float t_split_scale,
    uniform float t_opacity,
    uniform float t_big_scale,
    uniform uint n_scoring_views)
{
    uint i = dtid.x;
    uint n = num_gaussians_counter[0];

    if (i >= n)
    {
        action[i] = ACTION_KEEP;
        return;
    }

    Gaussian3D g = gaussians[i];
    float3 scale = exp(g.log_scale);
    float max_scale = max(scale.x, max(scale.y, scale.z));
    float opacity = sigmoid(g.opacity);

    // Oversized gaussians are always pruned unconditionally.
    if (max_scale > t_big_scale)
    {
        action[i] = ACTION_PRUNE;
        return;
    }

    float importance = float(importance_score_accum[i]) / float(max(n_scoring_views, 1u));

    // Low-opacity pruning (budget-limited).
    if (opacity < t_opacity)
    {
        int remaining;
        InterlockedAdd(prune_budget[0], -1, remaining);
        action[i] = (remaining > 0) ? ACTION_PRUNE : ACTION_KEEP;
        return;
    }

    bool is_large = max_scale > t_split_scale;

    // Growth: importance score is the primary signal (FastGS VCD).
    if (importance > importance_thresh)
    {
        int remaining;
        InterlockedAdd(growth_budget[0], -1, remaining);
        if (remaining <= 0)
        {
            action[i] = ACTION_KEEP;
            return;
        }
        action[i] = is_large ? ACTION_SPLIT : ACTION_CLONE;
    }
    else
    {
        action[i] = ACTION_KEEP;
    }
}
