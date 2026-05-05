// Opacity + scale regularization gradients.
//
// Adds L1 regularization penalties to the existing dL/d_opacity and
// dL/d_log_scale gradient buffers:
//   opacity_reg:  d/d(opacity_logit) of  w_opa * mean(sigmoid(opacity_logit))
//   scale_reg:    d/d(log_scale)     of  w_scale * mean(exp(log_scale))
//
// Run AFTER the backward rasterizer (which populates grad_opacity and
// grad_log_scale with the reconstruction loss gradients) so that the
// regularization gradients accumulate on top.

#include "../common/types.slang"
#include "../common/math.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void addRegularization(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    RWStructuredBuffer<int> grad_opacity,
    RWStructuredBuffer<int> grad_log_scale_x,
    RWStructuredBuffer<int> grad_log_scale_y,
    RWStructuredBuffer<int> grad_log_scale_z,
    uniform float w_opacity,
    uniform float w_scale,
    uniform float inv_n)
{
    uint i = dtid.x;
    if (i >= num_gaussians_counter[0]) return;

    Gaussian3D g = gaussians[i];

    // Opacity reg: d/d(logit) of (w * sigmoid(logit) / N)
    //   = w * sigmoid(logit) * (1 - sigmoid(logit)) / N
    if (w_opacity > 0.0)
    {
        float sig = sigmoid(g.opacity);
        float d_opa = w_opacity * sig * (1.0 - sig) * inv_n;
        InterlockedAdd(grad_opacity[i], toFixed(d_opa));
    }

    // Scale reg: d/d(log_scale) of (w * exp(log_scale) / N)
    //   = w * exp(log_scale) / N
    if (w_scale > 0.0)
    {
        float3 scale = exp(g.log_scale);
        float3 d_scale = w_scale * scale * inv_n;
        InterlockedAdd(grad_log_scale_x[i], toFixed(d_scale.x));
        InterlockedAdd(grad_log_scale_y[i], toFixed(d_scale.y));
        InterlockedAdd(grad_log_scale_z[i], toFixed(d_scale.z));
    }
}
