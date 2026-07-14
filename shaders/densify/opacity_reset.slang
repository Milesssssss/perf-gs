// Periodic opacity reset / cap. Two modes selected by `mode`:
//
//   mode == 0 (CLAMP, default): clamp every gaussian's opacity logit DOWNWARDS
//     to logit(reset_value). Gaussians already below keep their (lower) value.
//     Matches gsplat's ``reset_opa``: ``torch.clamp(p, max=logit(value))``. Used
//     for BOTH the gsplat-style reset (value 0.01) AND the SADGS post-densify
//     opacity CAP (value 0.8 == inverse_sigmoid(min(get_opacity, 0.8))).
//
//   mode == 1 (RELATIVE, SADGS reset_opacity): multiply the ACTIVATED opacity
//     by reset_value (the decay factor, e.g. 0.1): p_new = p_old * decay, then
//     back to logit. Unlike the clamp, this preserves the ordering between
//     strong and weak primitives (0.8->0.08, 0.4->0.04, ...) instead of
//     collapsing everyone above the cap to the same value. SADGS
//     gaussian_model.py:419 reset_opacity(opacity_reset_decay=0.1).
//
// ``reset_value`` is a probability (mode 0) or a decay factor (mode 1) on the
// Python side; logit conversion is log(p / (1 - p)).

#include "../common/types.slang"
#include "../common/math.slang"

// Zeroes the opacity Adam moments alongside the parameter change. SADGS's
// replace_tensor_to_optimizer (gaussian_model.py:485) resets exp_avg/exp_avg_sq
// to zero whenever it externally replaces the opacity tensor (reset AND the 0.8
// cap). Without this, the stale momentum (m,v) computed for the OLD opacity
// keeps driving the parameter after the reset/cap — opacity rebounds under old
// momentum instead of being re-earned from gradient. Applies to both modes.
[shader("compute")]
[numthreads(256, 1, 1)]
void resetOpacity(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    RWStructuredBuffer<AdamState> adam_state,
    StructuredBuffer<uint> num_gaussians_counter,
    uniform float reset_value,
    uniform uint mode)
{
    uint i = dtid.x;
    if (i >= num_gaussians_counter[0]) return;
    if (mode == 1u)
    {
        // Relative decay in activated space: p_new = p_old * decay.
        float p_old = sigmoid(gaussians[i].opacity);
        float p_new = clamp(p_old * reset_value, 1e-5, 1.0 - 1e-5);
        gaussians[i].opacity = log(p_new / (1.0 - p_new));
    }
    else
    {
        // Downward clamp to logit(reset_value) (gsplat reset / SADGS 0.8 cap).
        float p = clamp(reset_value, 1e-5, 1.0 - 1e-5);
        float logit_cap = log(p / (1.0 - p));
        gaussians[i].opacity = min(gaussians[i].opacity, logit_cap);
    }
    // Reset opacity Adam moments (match SADGS replace_tensor_to_optimizer).
    adam_state[i].opacity_m = 0.0;
    adam_state[i].opacity_v = 0.0;
}
