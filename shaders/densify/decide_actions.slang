// Phase A of densification: compute per-gaussian densification scores
// and apply unconditional pruning rules.
//
// Threshold-free growth selection: instead of hard thresholds on gradient
// magnitude or importance, this kernel computes a continuous score per
// gaussian. The Python host reads scores back, ranks them, and selects
// the top-B for split/clone based on the growth budget.
//
// Pruning decisions (oversized, low-opacity) are still applied here
// since they're structural invariants, not signal-quality judgments.
//
// Output:
//   action[i] = ACTION_PRUNE for pruned gaussians
//   action[i] = ACTION_KEEP  for all others (may be overwritten by host)
//   score[i]  = densification priority (0 for pruned/inactive)

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
    StructuredBuffer<int64_t> grad_mean_2d_mag_accum,
    StructuredBuffer<uint> visibility_count_accum,
    RWStructuredBuffer<uint> action,
    RWStructuredBuffer<float> densify_score,
    RWStructuredBuffer<int> prune_budget,
    uniform float t_opacity,
    uniform float t_big_scale,
    uniform uint n_scoring_views)
{
    uint i = dtid.x;
    uint n = num_gaussians_counter[0];

    if (i >= n)
    {
        action[i] = ACTION_KEEP;
        densify_score[i] = 0.0;
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
        densify_score[i] = 0.0;
        return;
    }

    // Low-opacity pruning (budget-limited).
    if (opacity < t_opacity)
    {
        int remaining;
        InterlockedAdd(prune_budget[0], -1, remaining);
        if (remaining > 0)
        {
            action[i] = ACTION_PRUNE;
            densify_score[i] = 0.0;
            return;
        }
    }

    // Densification score: combines multiple signals (inspired by Taming 3DGS §3.3).
    //
    //   importance : multi-view error attribution (how many error pixels touch this gaussian)
    //   avg_grad   : under-reconstruction signal (2D position gradient magnitude)
    //   opacity    : high opacity = real geometry worth densifying, low = floater/ghost
    //   scale_prod : surface area proxy — larger gaussians benefit more from subdivision
    //
    // All signals are multiplied together. Opacity acts as a gate (near-zero kills
    // the score), scale_prod biases toward gaussians that cover more area.

    float importance = float(importance_score_accum[i]) / float(max(n_scoring_views, 1u));

    uint vis = visibility_count_accum[i];
    float avg_grad = (vis > 0)
        ? fromFixed(grad_mean_2d_mag_accum[i]) / float(vis)
        : 0.0;

    float scale_prod = scale.x * scale.y + scale.x * scale.z + scale.y * scale.z;

    float score = importance * avg_grad * opacity * scale_prod;

    action[i] = ACTION_KEEP;
    densify_score[i] = score;
}
