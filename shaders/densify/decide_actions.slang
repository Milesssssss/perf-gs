// Phase A of densification: decide per-gaussian action.
//
// One thread per gaussian. Reads tracking accumulators + current
// log_scale/opacity, writes ``action[i] ∈ {KEEP, SPLIT, CLONE, PRUNE}``.
// Paper heuristic from Kerbl et al. (3D Gaussian Splatting):
//
//   mean_2d_grad_norm  =  fixed-to-float(grad_mean_2d_mag_accum[i]) / window_iters
//   max_scale          =  max(exp(log_scale))
//
//   PRUNE  if  opacity < t_opacity  or  max_scale > t_big_scale
//   SPLIT  if  mean_2d_grad_norm > t_grad  AND  max_scale > t_split_scale
//   CLONE  if  mean_2d_grad_norm > t_grad  AND  max_scale <= t_split_scale
//   KEEP   otherwise
//
// ``window_iters`` lets the Python side normalize gradients accumulated
// over N iters into a per-iter-equivalent signal so the same thresholds
// work regardless of densification cadence.

#include "../common/types.slang"

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
    StructuredBuffer<int> grad_mean_2d_mag_accum,
    StructuredBuffer<uint> pixel_coverage_accum,
    RWStructuredBuffer<uint> action,
    uniform float window_iters,
    uniform float grad_scale,     // 1 / max(frame_dim) to normalize grad to image units
    uniform float t_grad,
    uniform float t_split_scale,
    uniform float t_opacity,
    uniform float t_big_scale)
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

    if (g.opacity < t_opacity || max_scale > t_big_scale)
    {
        action[i] = ACTION_PRUNE;
        return;
    }

    // Normalize accumulated pixel-space grad mag into per-iter,
    // image-relative units. ``grad_scale`` is typically 1/max(W,H).
    float mag_total = fromFixed(grad_mean_2d_mag_accum[i]);
    float mag_per_iter = (mag_total * grad_scale) / max(window_iters, 1.0);

    if (mag_per_iter > t_grad)
    {
        action[i] = (max_scale > t_split_scale) ? ACTION_SPLIT : ACTION_CLONE;
    }
    else
    {
        action[i] = ACTION_KEEP;
    }
}
