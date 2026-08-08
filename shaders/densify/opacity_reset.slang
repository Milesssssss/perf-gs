#include "../common/types.slang"
#include "../common/math.slang"

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
