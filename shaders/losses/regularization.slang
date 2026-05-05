// Opacity + scale regularization gradients (float atomic version).
//
// Adds L1 regularization penalties directly to the float gradient buffers:
//   opacity_reg:  d/d(opacity_logit) of  w_opa * mean(sigmoid(opacity_logit))
//   scale_reg:    d/d(log_scale)     of  w_scale * mean(exp(log_scale))

#include "../common/types.slang"
#include "../common/math.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void addRegularization(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    RWStructuredBuffer<float> grad_opacity,
    RWStructuredBuffer<float> grad_log_scale_x,
    RWStructuredBuffer<float> grad_log_scale_y,
    RWStructuredBuffer<float> grad_log_scale_z,
    uniform float w_opacity,
    uniform float w_scale,
    uniform float inv_n)
{
    uint i = dtid.x;
    if (i >= num_gaussians_counter[0]) return;

    Gaussian3D g = gaussians[i];

    if (w_opacity > 0.0)
    {
        float sig = sigmoid(g.opacity);
        float d_opa = w_opacity * sig * (1.0 - sig) * inv_n;
        InterlockedAdd(grad_opacity[i], d_opa);
    }

    if (w_scale > 0.0)
    {
        float3 scale = exp(g.log_scale);
        float3 d_scale = w_scale * scale * inv_n;
        InterlockedAdd(grad_log_scale_x[i], d_scale.x);
        InterlockedAdd(grad_log_scale_y[i], d_scale.y);
        InterlockedAdd(grad_log_scale_z[i], d_scale.z);
    }
}
