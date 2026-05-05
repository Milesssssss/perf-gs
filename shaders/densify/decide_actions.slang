// Phase A of densification: decide per-gaussian action using FastGS
// multi-view importance score with budget-limited growth AND pruning.
//
// Decision rules:
//   importance = importance_score_accum[i] / n_scoring_views
//   max_scale  = max(exp(log_scale))
//
//   PRUNE candidates:
//     sigmoid(opacity) < t_opacity  OR  max_scale > t_big_scale
//     Budget-limited: only 50% of candidates are actually pruned (atomic).
//   SPLIT  if  importance > importance_thresh  AND  max_scale > t_split_scale
//   CLONE  if  importance > importance_thresh  AND  max_scale <= t_split_scale
//     Budget-limited by growth_budget.
//   KEEP   otherwise

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

    // Prune candidates: low opacity or oversized.
    bool is_prune_candidate = (opacity < t_opacity) || (max_scale > t_big_scale);
    if (is_prune_candidate)
    {
        // Budget-limited: only 50% of candidates actually get pruned.
        int remaining;
        InterlockedAdd(prune_budget[0], -1, remaining);
        if (remaining > 0)
        {
            action[i] = ACTION_PRUNE;
        }
        else
        {
            action[i] = ACTION_KEEP;
        }
        return;
    }

    float importance = float(importance_score_accum[i]) / float(max(n_scoring_views, 1u));

    if (importance > importance_thresh)
    {
        int remaining;
        InterlockedAdd(growth_budget[0], -1, remaining);
        if (remaining <= 0)
        {
            action[i] = ACTION_KEEP;
            return;
        }
        action[i] = (max_scale > t_split_scale) ? ACTION_SPLIT : ACTION_CLONE;
    }
    else
    {
        action[i] = ACTION_KEEP;
    }
}
