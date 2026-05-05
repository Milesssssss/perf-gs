// Phase A of densification: decide per-gaussian action using FastGS
// multi-view importance score with growth budget.
//
// One thread per gaussian. Reads the importance score accumulated over
// N scoring views + current log_scale/opacity, writes
// ``action[i] in {KEEP, SPLIT, CLONE, PRUNE}``.
//
// Decision rules:
//   importance = importance_score_accum[i] / n_scoring_views
//   max_scale  = max(exp(log_scale))
//
//   PRUNE  if  sigmoid(opacity) < t_opacity  OR  max_scale > t_big_scale
//   SPLIT  if  importance > importance_thresh  AND  max_scale > t_split_scale
//   CLONE  if  importance > importance_thresh  AND  max_scale <= t_split_scale
//   KEEP   otherwise
//
// Growth is budget-limited: a shared atomic counter (growth_budget) is
// decremented by each thread that wants to split/clone. Once the budget
// hits zero, remaining candidates are demoted to KEEP. This gives
// approximate top-N selection (threads with higher importance that race
// first get the budget — not perfectly ranked but good enough for
// stochastic training).

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

    // Prune: low opacity or oversized world-space scale.
    if (opacity < t_opacity || max_scale > t_big_scale)
    {
        action[i] = ACTION_PRUNE;
        return;
    }

    float importance = float(importance_score_accum[i]) / float(max(n_scoring_views, 1u));

    if (importance > importance_thresh)
    {
        // Consume one unit of growth budget via atomic decrement.
        // If budget is already <= 0, demote to KEEP.
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
