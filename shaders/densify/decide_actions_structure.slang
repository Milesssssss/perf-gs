// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#include "../common/types.slang"
#include "../common/math.slang"
#include "blend_weight.slang"

// Exclusive popcount of lower lanes. Metal compute has no WavePrefixCountBits; the ballot
// is the same operation (lane 0 is the least significant bit). Other targets keep the
// intrinsic so their codegen does not change.
uint wavePrefixCountBits(bool pred)
{
    __target_switch
    {
    case metal:
        {
            uint4 ballot = WaveActiveBallot(pred);
            uint lane = WaveGetLaneIndex();
            uint bit = lane & 31u;
            uint word = lane >> 5u;
            uint below = (bit == 0u) ? 0u : ((1u << bit) - 1u);
            uint count = countbits(ballot[word] & below);
            if (word > 0u) count += countbits(ballot[0]);
            if (word > 1u) count += countbits(ballot[1]);
            if (word > 2u) count += countbits(ballot[2]);
            return count;
        }
    default:
        return WavePrefixCountBits(pred);
    }
}

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

// Propose each Gaussian's action and per-axis split factors before applying the slot budget.
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
    StructuredBuffer<int64_t> blend_weight_accum,
    RWStructuredBuffer<uint> action,
    RWStructuredBuffer<uint> split_factor_x,
    RWStructuredBuffer<uint> split_factor_y,
    RWStructuredBuffer<uint> split_factor_z,
    RWStructuredBuffer<uint> densify_children_sum,  // atomic: total requested tail growth (for overbudget log)
    RWStructuredBuffer<uint> selected_parents,
    RWStructuredBuffer<uint> selected_count,
    uniform float tau_split, uniform float tau_prune,
    uniform float grad_threshold_internal,
    uniform float split_power, uniform uint max_split_factor,
    uniform float t_opacity, uniform float dense_scale,
    uniform float min_blend_weight,
    uniform uint max_splats)
{
    uint gaussian_idx = dtid.x;
    // Every lane must reach the wave ops below. Metal simdgroup instructions are undefined
    // if any lane in the simdgroup has already returned, and a 256-thread group is several
    // Apple simdgroups. Lanes outside the live range contribute the inactive identity
    // (not selected, zero growth), which is what WaveActive* ignores on the other targets.
    uint live_count = num_gaussians_counter[0];
    bool live = gaussian_idx < max_splats && gaussian_idx < live_count;

    uint out_action = ACTION_KEEP;
    uint nkx = 1u, nky = 1u, nkz = 1u;
    if (live)
    {
        uint n_total_u = eta_n_total[gaussian_idx];
        bool seen = n_total_u > 0u;
        float inv_total = seen ? (1.0 / float(max(n_total_u, 1u))) : 0.0;

        float high_ratio = float(eta_n_high[gaussian_idx]) * inv_total;
        float low_ratio  = float(eta_n_low[gaussian_idx])  * inv_total;

        // Eta maxima are uint(eta*FIXED_SCALE): decode with a plain divide, not fromFixed (int64 grad convention).
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

        bool is_prune = ((low_ratio > tau_prune) && seen) || (opacity < t_opacity)
            || isUnderweight(blend_weight_accum[gaussian_idx], vis, min_blend_weight);
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
            }
        }

        action[gaussian_idx] = out_action;
        split_factor_x[gaussian_idx] = nkx;
        split_factor_y[gaussian_idx] = nky;
        split_factor_z[gaussian_idx] = nkz;
    }

    // One reservation per wave; only these parents need split workgroups.
    bool selected = out_action == ACTION_SPLIT || out_action == ACTION_CLONE;
    uint wave_count = WaveActiveCountBits(selected);
    uint offset = wavePrefixCountBits(selected);
    uint base = 0u;
    uint requested = out_action == ACTION_SPLIT ? nkx * nky * nkz - 1u
                   : (out_action == ACTION_CLONE ? 1u : 0u);
    uint wave_growth = WaveActiveSum(requested);
    if (WaveIsFirstLane() && wave_count > 0u)
    {
        InterlockedAdd(selected_count[0], wave_count, base);
        InterlockedAdd(densify_children_sum[0], wave_growth);
    }
    base = WaveReadLaneFirst(base);
    if (selected) selected_parents[base + offset] = gaussian_idx;
}
