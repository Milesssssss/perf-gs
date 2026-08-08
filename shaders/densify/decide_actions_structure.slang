#include "../common/types.slang"
#include "../common/math.slang"

static const uint ACTION_KEEP  = 0;
static const uint ACTION_SPLIT = 1;
static const uint ACTION_CLONE = 2;
static const uint ACTION_PRUNE = 3;

uint per_axis_split_factor(float eta_axis, float split_power, uint max_split_factor)
{
    float e = max(eta_axis, 0.0);
    float raw = ceil(pow(e, split_power));
    return clamp(uint(raw), 1u, max_split_factor);
}

// Single-pass structure-aware decision. One thread per gaussian: reads the
// multiview eta accumulators + grad window, writes KEEP/SPLIT/CLONE/PRUNE into
// `action` and the per-axis grid factors into `split_factor_{x,y,z}`.
//
// No capacity clamp / demotion: if a window's requested splits exceed the buffer
// headroom, split_anisotropic drops the overflow children at the tail and the
// host logs it (densify_children_sum is read back after the tick). We surface an
// overbudget event rather than silently downsizing splits — see densify.py.
[shader("compute")]
[numthreads(256, 1, 1)]
void decideActionsStructure(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<uint> eta_n_total,
    StructuredBuffer<uint> eta_n_high,
    StructuredBuffer<uint> eta_n_low,
    StructuredBuffer<uint> eta_max_x,
    StructuredBuffer<uint> eta_max_y,
    StructuredBuffer<uint> eta_max_z,
    StructuredBuffer<int64_t> grad_mean_2d_mag_accum,
    StructuredBuffer<uint> visibility_count_accum,
    RWStructuredBuffer<uint> action,
    RWStructuredBuffer<uint> split_factor_x,
    RWStructuredBuffer<uint> split_factor_y,
    RWStructuredBuffer<uint> split_factor_z,
    RWStructuredBuffer<uint> densify_children_sum,  // atomic: total requested tail growth (for overbudget log)
    uniform float tau_split, uniform float tau_prune,
    uniform float grad_threshold_internal,
    uniform float split_power, uniform uint max_split_factor,
    uniform float t_opacity, uniform float dense_scale,
    uniform uint max_splats)
{
    uint gaussian_idx = dtid.x;
    if (gaussian_idx >= max_splats) return;

    // Tail slots (past the live count): pad KEEP with nk=1.
    if (gaussian_idx >= num_gaussians_counter[0])
    {
        action[gaussian_idx] = ACTION_KEEP;
        split_factor_x[gaussian_idx] = 1u;
        split_factor_y[gaussian_idx] = 1u;
        split_factor_z[gaussian_idx] = 1u;
        return;
    }

    uint n_total_u = eta_n_total[gaussian_idx];
    bool seen = n_total_u > 0u;
    float inv_total = seen ? (1.0 / float(max(n_total_u, 1u))) : 0.0;

    float high_ratio = float(eta_n_high[gaussian_idx]) * inv_total;
    float low_ratio  = float(eta_n_low[gaussian_idx])  * inv_total;

    // Per-axis eta maxima: uint fixed-point (uint(eta*FIXED_SCALE)), decode with
    // a plain divide — NOT fromFixed (that is the int64 grad convention).
    float eta_x = float(eta_max_x[gaussian_idx]) / FIXED_SCALE;
    float eta_y = float(eta_max_y[gaussian_idx]) / FIXED_SCALE;
    float eta_z = float(eta_max_z[gaussian_idx]) / FIXED_SCALE;

    uint vis = visibility_count_accum[gaussian_idx];
    float avg_grad = 0.0;
    if (vis > 0u)
        avg_grad = fromFixed(grad_mean_2d_mag_accum[gaussian_idx]) / float(vis);
    bool is_grad_high = avg_grad >= grad_threshold_internal;

    Gaussian3D g = gaussians[gaussian_idx];
    float opacity = sigmoid(g.opacity);
    float3 scale = exp(g.log_scale);
    float max_scale = max(scale.x, max(scale.y, scale.z));

    // Defaults; overwritten below on the split/clone/prune branches.
    uint out_action = ACTION_KEEP;
    uint nkx = 1u, nky = 1u, nkz = 1u;

    bool is_prune = ((low_ratio > tau_prune) && seen) || (opacity < t_opacity);
    bool triggered = (high_ratio > tau_split) && is_grad_high && seen;

    if (is_prune)
    {
        out_action = ACTION_PRUNE;
    }
    else if (triggered)
    {
        nkx = per_axis_split_factor(eta_x, split_power, max_split_factor);
        nky = per_axis_split_factor(eta_y, split_power, max_split_factor);
        nkz = per_axis_split_factor(eta_z, split_power, max_split_factor);
        uint prod = nkx * nky * nkz;

        bool is_small = (max_scale <= dense_scale);
        if (is_small || prod <= 1u)
        {
            out_action = ACTION_CLONE;
            nkx = 1u; nky = 1u; nkz = 1u;
        }
        else
        {
            out_action = ACTION_SPLIT;
            //TODO: all splitting threads hammering one counter; fine at this cadence.
            InterlockedAdd(densify_children_sum[0], prod - 1u);
        }
    }

    action[gaussian_idx] = out_action;
    split_factor_x[gaussian_idx] = nkx;
    split_factor_y[gaussian_idx] = nky;
    split_factor_z[gaussian_idx] = nkz;
}
