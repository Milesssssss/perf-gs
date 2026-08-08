#include "../common/types.slang"
#include "../common/math.slang"

static const uint ACTION_KEEP  = 0;
static const uint ACTION_SPLIT = 1;
static const uint ACTION_CLONE = 2;
static const uint ACTION_PRUNE = 3;

static const uint FANOUT_BINS = 513;   // prod - 1 in [0, max_split_factor^3 - 1] <= 511
static const uint NO_DEMOTE   = 0xFFFFFFFFu;

uint _nk_from_eta(float eta_axis, float split_power, uint max_split_factor)
{
    float e = max(eta_axis, 0.0);
    // pow(0, split_power) is 0 for split_power > 0; ceil(0) = 0, clamped to 1.
    // pow behaves identically to numpy's np.power for the >=0 domain we use.
    float raw = ceil(pow(e, split_power));
    // clamp(1, ...) also handles the eta=0 case (raw=0 -> nk=1).
    return clamp(uint(raw), 1u, max_split_factor);
}

// Tentative per-gaussian decide. Fills `action_out`, `nk_out.{x,y,z}`, and
// (via out-param) the tentative prod. Common to Pass 1 and Pass 2 so we don't
// duplicate the decision logic.
void _tentative_decide(
    uint i,
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
    float tau_split, float tau_prune,
    float grad_threshold_internal,
    float split_power, uint max_split_factor,
    float t_opacity, float dense_scale,
    out uint action_out, out uint nkx_out, out uint nky_out, out uint nkz_out,
    out uint prod_out, out bool is_tentative_split_out)
{
    // Sane defaults so early-out paths don't write garbage.
    action_out = ACTION_KEEP;
    nkx_out = 1u; nky_out = 1u; nkz_out = 1u;
    prod_out = 1u;
    is_tentative_split_out = false;

    // Out-of-count slot: no-op (matches CPU behavior — CPU only iterates
    // pre_count and pads the rest with ACTION_KEEP in _upload).
    if (i >= num_gaussians_counter[0]) return;

    uint n_total_u = eta_n_total[i];
    bool seen = n_total_u > 0u;
    // inv_total matches CPU: np.where(seen, 1.0 / max(n_total, 1), 0.0).
    float inv_total = seen ? (1.0 / float(max(n_total_u, 1u))) : 0.0;

    float high_ratio = float(eta_n_high[i]) * inv_total;
    float low_ratio  = float(eta_n_low[i])  * inv_total;

    // eta_max_{x,y,z} are uint32 fixed-point (score_structure_aware.slang
    // stores `uint(eta * FIXED_SCALE)` via InterlockedMax). Divide by
    // FIXED_SCALE — do NOT use fromFixed which is int64.
    float eta_x = float(eta_max_x[i]) / FIXED_SCALE;
    float eta_y = float(eta_max_y[i]) / FIXED_SCALE;
    float eta_z = float(eta_max_z[i]) / FIXED_SCALE;

    // Grad gate. Matches CPU: avg_grad = fromFixed(raw) / vis, threshold is
    // grad_threshold * W*H*3 (pre-multiplied by the host, same convention as
    // decide_actions_simple.slang).
    uint vis = visibility_count_accum[i];
    float avg_grad = 0.0;
    if (vis > 0u)
        avg_grad = fromFixed(grad_mean_2d_mag_accum[i]) / float(vis);
    bool is_grad_high = avg_grad >= grad_threshold_internal;

    // Opacity + max scale.
    Gaussian3D g = gaussians[i];
    float opacity = sigmoid(g.opacity);
    float3 scale = exp(g.log_scale);
    float max_scale = max(scale.x, max(scale.y, scale.z));

    // Prune gate (mirrors CPU: (low_ratio > tau_prune & seen) OR opacity < t_opacity).
    bool is_prune = ((low_ratio > tau_prune) && seen) || (opacity < t_opacity);

    if (is_prune)
    {
        action_out = ACTION_PRUNE;
        return;
    }

    // Densification trigger.
    bool triggered = (high_ratio > tau_split) && is_grad_high && seen;
    if (!triggered) return;   // ACTION_KEEP (default)

    // Per-axis nk + prod.
    uint nkx = _nk_from_eta(eta_x, split_power, max_split_factor);
    uint nky = _nk_from_eta(eta_y, split_power, max_split_factor);
    uint nkz = _nk_from_eta(eta_z, split_power, max_split_factor);
    uint prod = nkx * nky * nkz;

    // Clone vs split by max world-space scale.
    bool is_small = (max_scale <= dense_scale);
    // Degenerate split (prod == 1) falls back to CLONE, matching CPU.
    if (is_small || prod <= 1u)
    {
        action_out = ACTION_CLONE;
        // Clones keep nk = (1,1,1) — split_anisotropic ignores split_factor
        // for CLONE actions (it appends one exact duplicate).
        return;
    }

    // Tentative anisotropic split.
    action_out = ACTION_SPLIT;
    nkx_out = nkx; nky_out = nky; nkz_out = nkz;
    prod_out = prod;
    is_tentative_split_out = true;
}

// ---------- Pass 0: clear the capacity-bound scratch ----------

// Dispatched at [FANOUT_BINS, 1, 1]. Thread 0 additionally zeroes
// children_sum and fanout_threshold (with the "no demote" sentinel — Pass 1.5
// overwrites it if overflow occurs).
[shader("compute")]
[numthreads(64, 1, 1)]
void clearCapacityScratch(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<uint> densify_children_sum,
    RWStructuredBuffer<uint> densify_fanout_hist,
    RWStructuredBuffer<uint> densify_fanout_threshold)
{
    uint i = dtid.x;
    if (i >= FANOUT_BINS) return;
    densify_fanout_hist[i] = 0u;
    if (i == 0u)
    {
        densify_children_sum[0] = 0u;
        densify_fanout_threshold[0] = NO_DEMOTE;
    }
}

// ---------- Pass 1: tentative decide + accumulate fan-out sum + histogram ----------

[shader("compute")]
[numthreads(256, 1, 1)]
void decideActionsStructurePass1(
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
    RWStructuredBuffer<uint> densify_children_sum,
    RWStructuredBuffer<uint> densify_fanout_hist,
    uniform float tau_split, uniform float tau_prune,
    uniform float grad_threshold_internal,
    uniform float split_power, uniform uint max_split_factor,
    uniform float t_opacity, uniform float dense_scale,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;

    // Default write for slots past the counter — matches the padded-KEEP
    // upload the CPU used to do (`_upload` at densify.py:761-764).
    if (i >= num_gaussians_counter[0])
    {
        action[i] = ACTION_KEEP;
        split_factor_x[i] = 1u;
        split_factor_y[i] = 1u;
        split_factor_z[i] = 1u;
        return;
    }

    uint act; uint nkx, nky, nkz; uint prod; bool is_split;
    _tentative_decide(i, gaussians, num_gaussians_counter,
        eta_n_total, eta_n_high, eta_n_low,
        eta_max_x, eta_max_y, eta_max_z,
        grad_mean_2d_mag_accum, visibility_count_accum,
        tau_split, tau_prune, grad_threshold_internal,
        split_power, max_split_factor,
        t_opacity, dense_scale,
        act, nkx, nky, nkz, prod, is_split);

    action[i] = act;
    split_factor_x[i] = nkx;
    split_factor_y[i] = nky;
    split_factor_z[i] = nkz;

    // For tentative splits, record fan-out for the capacity bound. The
    // histogram bin is (prod - 1) — the number of extra children this split
    // would append at the tail (split_anisotropic.slang overwrites the parent
    // slot for child 0 and atomic-appends the remaining prod-1 children).
    if (is_split)
    {
        uint dummy;
        InterlockedAdd(densify_children_sum[0], prod - 1u, dummy);
        uint bin = min(prod - 1u, FANOUT_BINS - 1u);   // safety clamp
        InterlockedAdd(densify_fanout_hist[bin], 1u, dummy);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void decideActionsStructureThreshold(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<uint> densify_children_sum,
    StructuredBuffer<uint> densify_fanout_hist,
    RWStructuredBuffer<uint> densify_fanout_threshold,
    uniform uint max_splats)
{
    uint pre_count = num_gaussians_counter[0];
    uint children_sum = densify_children_sum[0];

    uint headroom = (max_splats > pre_count) ? (max_splats - pre_count) : 0u;

    if (children_sum <= headroom)
    {
        densify_fanout_threshold[0] = NO_DEMOTE;
        return;
    }

    uint overflow = children_sum - headroom;

    uint saved = 0u;
    uint bin = FANOUT_BINS;   // exclusive upper — loop counter, not the answer
    while (bin > 0u)
    {
        bin -= 1u;
        uint pop = densify_fanout_hist[bin];
        saved += pop * bin;
        if (saved >= overflow)
        {
            densify_fanout_threshold[0] = bin;
            return;
        }
    }

    densify_fanout_threshold[0] = 1u;
}


[shader("compute")]
[numthreads(256, 1, 1)]
void decideActionsStructurePass2(
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
    StructuredBuffer<uint> densify_fanout_threshold,
    RWStructuredBuffer<uint> action,
    RWStructuredBuffer<uint> split_factor_x,
    RWStructuredBuffer<uint> split_factor_y,
    RWStructuredBuffer<uint> split_factor_z,
    uniform float tau_split, uniform float tau_prune,
    uniform float grad_threshold_internal,
    uniform float split_power, uniform uint max_split_factor,
    uniform float t_opacity, uniform float dense_scale,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;

    // Preserve Pass 1's padded-KEEP for slots past the counter.
    if (i >= num_gaussians_counter[0])
    {
        // Already set to KEEP by Pass 1; nothing to do.
        return;
    }

    uint threshold = densify_fanout_threshold[0];
    if (threshold == NO_DEMOTE) return;   // Common case: nothing to demote.

    uint act; uint nkx, nky, nkz; uint prod; bool is_split;
    _tentative_decide(i, gaussians, num_gaussians_counter,
        eta_n_total, eta_n_high, eta_n_low,
        eta_max_x, eta_max_y, eta_max_z,
        grad_mean_2d_mag_accum, visibility_count_accum,
        tau_split, tau_prune, grad_threshold_internal,
        split_power, max_split_factor,
        t_opacity, dense_scale,
        act, nkx, nky, nkz, prod, is_split);

    if (!is_split) return;   // Only tentative SPLITs can be demoted.

    // prod - 1 >= threshold means this split's fan-out is >= threshold — demote.
    if ((prod - 1u) >= threshold)
    {
        action[i] = ACTION_CLONE;
        split_factor_x[i] = 1u;
        split_factor_y[i] = 1u;
        split_factor_z[i] = 1u;
    }
    // Otherwise Pass 1's tentative SPLIT + nk stands; nothing to write.
}
