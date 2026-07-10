// Structure-aware densification decision — wait-free GPU port.
//
// Byte-for-byte translation of the former CPU-side decision in
// densify.py::densify_structure_aware. Runs entirely on the GPU: no readbacks,
// no host numpy, no wait_for_idle. Called from a single command encoder,
// mirroring densify_simple.
//
// Two logical passes with a tiny threshold-picker between them, so the
// capacity bound stays deterministic without a host argsort:
//
//   Pass 1  (decideActionsStructurePass1, [max_splats])
//     Per-gaussian tentative decide: writes action + split_factor_{x,y,z}
//     as if there were no capacity limit. For each tentative SPLIT, atomically
//     adds its (prod - 1) to `children_sum` and bumps `fanout_hist[prod - 1]`.
//     Cheap: one atomic per triggered split.
//
//   Pass 1.5 (decideActionsStructureThreshold, [1])
//     Single-thread. Reads `children_sum`, `pre_count`, `max_splats`. Computes
//     `overflow = max(0, pre_count + children_sum - max_splats)`. If zero,
//     writes UINT_MAX to `fanout_threshold` (sentinel — "no demotion"). Else
//     scans `fanout_hist` top-down, accumulating (bin_pop * bin_fanout) until
//     the running sum >= overflow, and writes that bin's fan-out as the
//     threshold. Every SPLIT with `prod - 1 >= threshold` will be demoted.
//     Bounded above by 513 iterations because prod <= max_split_factor^3 <= 512.
//
//   Pass 2  (decideActionsStructurePass2, [max_splats])
//     Per-gaussian finalize: recomputes the same tentative decide (cheap —
//     the eta reads are ~40 bytes/gaussian, cached), and if the tentative
//     action is SPLIT with `prod - 1 >= threshold`, demotes to CLONE and
//     resets split_factor_{x,y,z} to 1. Otherwise the write is idempotent.
//     Re-computing avoids allocating a `prod-1` scratch buffer for the
//     inter-pass hand-off.
//
// Pass 0 (clearCapacityScratch, [513]) zeros children_sum + fanout_hist
// + fanout_threshold at the start of the encoder.
//
// The tentative-then-demote design preserves the semantics of the former
// CPU argsort: within a fan-out bin, atomic-race order picks which splits
// get demoted (same non-canonical tie-break the CPU sort already had for
// equal fan-outs); across bins, the "largest fan-out demoted first" order
// is preserved exactly by scanning the histogram top-down.

#include "../common/types.slang"
#include "../common/math.slang"

static const uint ACTION_KEEP  = 0;
static const uint ACTION_SPLIT = 1;
static const uint ACTION_CLONE = 2;
static const uint ACTION_PRUNE = 3;

static const uint FANOUT_BINS = 513;   // prod - 1 in [0, max_split_factor^3 - 1] <= 511
static const uint NO_DEMOTE   = 0xFFFFFFFFu;

// ---------- helpers ----------

// Compute the per-axis split factor n_k for one axis, matching the CPU
// np.clip(np.ceil(np.power(max(eta, 0), split_power)), 1, max_split_factor).
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

// ---------- Pass 1.5: pick the fan-out demotion threshold ----------

// Dispatched at [1, 1, 1]. Scans fanout_hist from bin (FANOUT_BINS - 1) down
// to bin 0, accumulating each bin's tail-growth contribution (count * bin)
// until the accumulated demote-savings reach `overflow`. That bin's fan-out
// becomes the threshold: every SPLIT with (prod - 1) >= threshold demotes to
// CLONE. If no overflow, we write the NO_DEMOTE sentinel.
//
// Bounded above by FANOUT_BINS iterations (<= 513) on a single thread —
// microseconds even in the worst case.
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

    // Headroom = how many tail slots we have. If pre_count already >= max
    // (shouldn't happen because splitAnisotropic self-bounds too, but be
    // defensive), demote everything.
    uint headroom = (max_splats > pre_count) ? (max_splats - pre_count) : 0u;

    if (children_sum <= headroom)
    {
        densify_fanout_threshold[0] = NO_DEMOTE;
        return;
    }

    uint overflow = children_sum - headroom;

    // Scan bins high-to-low. `saved` accumulates the tail-growth we free up
    // by demoting all splits in bins [bin, FANOUT_BINS - 1] to clones. We
    // want the smallest `bin` such that `saved >= overflow`.
    uint saved = 0u;
    uint bin = FANOUT_BINS;   // exclusive upper — loop counter, not the answer
    while (bin > 0u)
    {
        bin -= 1u;
        // A bin at index `b` represents splits with prod - 1 == b. Each such
        // split contributes exactly `b` to the tail growth (child 0 overwrites
        // the parent slot, so only prod - 1 = b children go to the tail).
        // Demoting it to CLONE frees those `b` slots (CLONE appends 1 child,
        // but the parent slot stays occupied — net -b, +1; simpler to say the
        // clone still consumes 1 slot, so freed growth = b - 1... but we want
        // to preserve the semantic: after demotion, the split no longer emits
        // prod-1 tail children — the clone emits 1. So freed = (prod-1) - 1
        // for bin >= 1, and 0 for bin == 0. However, bin==0 means prod==1
        // which is already demoted to clone in _tentative_decide, so it never
        // contributes to the histogram. Thus for every populated bin b >= 1
        // the per-split demotion frees (b - 1) tail slots.
        //
        // We conservatively use `b` per split — matching the CPU code's
        // capacity accounting which counts CLONE against `headroom` first
        // and then splits against the remainder. The overshoot (at most one
        // extra clone per demoted split, i.e. up to (population of this bin)
        // extra slots) is bounded and safe: split_anisotropic.slang has a
        // per-child `dst < max_splats` check as a final safety net.
        uint pop = densify_fanout_hist[bin];
        saved += pop * bin;
        if (saved >= overflow)
        {
            densify_fanout_threshold[0] = bin;
            return;
        }
    }

    // If we ran out of bins without covering overflow (shouldn't happen given
    // children_sum was computed from the same histogram), demote every split.
    densify_fanout_threshold[0] = 1u;
}

// ---------- Pass 2: demote high-fan-out splits to CLONE ----------

// If `fanout_threshold == NO_DEMOTE`, this pass is effectively a no-op — every
// thread re-runs the tentative decide and writes back the same values it wrote
// in Pass 1. We could branch host-side and skip the dispatch, but the cost is
// small and the branch-free discipline mirrors densify_simple.
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
