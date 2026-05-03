// Adam optimizer step.
//
// One thread per gaussian. Reads SoA int32 fixed-point gradients,
// converts to float, runs Adam, writes updated params and m/v state.
// No atomics needed — each thread owns its gaussian's slot across all
// 14 grad buffers and one AdamState record.
//
// The training_step counter is incremented by Python (one write per step)
// *before* this kernel runs; we read it as a uniform.
//
// LR decay: positional lr decays exponentially over a fixed schedule
// (PLAN.md Q15, restricted to lr_mean). Python passes the already-decayed
// value.

#include "../common/types.slang"

static const float ADAM_BETA1 = 0.9;
static const float ADAM_BETA2 = 0.999;
static const float ADAM_EPSILON = 1e-8;

struct AdamConfig
{
    float lr_mean;
    float lr_scale;
    float lr_rot;
    float lr_opacity;
    float lr_color;
    float bias_correction1;
    float bias_correction2;
};

// Scalar Adam step. Not a template — Slang generics get clunky for float3
// element-wise ops, and inlining this by hand keeps the kernel readable.
float adam_scalar(
    float param, float grad,
    inout float m, inout float v,
    float lr, float bc1, float bc2)
{
    m = ADAM_BETA1 * m + (1.0 - ADAM_BETA1) * grad;
    v = ADAM_BETA2 * v + (1.0 - ADAM_BETA2) * grad * grad;
    float m_hat = m / bc1;
    float v_hat = v / bc2;
    return param - lr * m_hat / (sqrt(v_hat) + ADAM_EPSILON);
}

float3 adam_vec3(float3 p, float3 g, inout float3 m, inout float3 v,
                 float lr, float bc1, float bc2)
{
    return float3(
        adam_scalar(p.x, g.x, m.x, v.x, lr, bc1, bc2),
        adam_scalar(p.y, g.y, m.y, v.y, lr, bc1, bc2),
        adam_scalar(p.z, g.z, m.z, v.z, lr, bc1, bc2)
    );
}

float4 adam_vec4(float4 p, float4 g, inout float4 m, inout float4 v,
                 float lr, float bc1, float bc2)
{
    return float4(
        adam_scalar(p.x, g.x, m.x, v.x, lr, bc1, bc2),
        adam_scalar(p.y, g.y, m.y, v.y, lr, bc1, bc2),
        adam_scalar(p.z, g.z, m.z, v.z, lr, bc1, bc2),
        adam_scalar(p.w, g.w, m.w, v.w, lr, bc1, bc2)
    );
}

[shader("compute")]
[numthreads(256, 1, 1)]
void adamStep(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    RWStructuredBuffer<AdamState> adam_state,
    StructuredBuffer<uint> num_gaussians_counter,
    // Gradient buffers (SoA, fixed-point int32)
    StructuredBuffer<int> grad_mean_x,
    StructuredBuffer<int> grad_mean_y,
    StructuredBuffer<int> grad_mean_z,
    StructuredBuffer<int> grad_log_scale_x,
    StructuredBuffer<int> grad_log_scale_y,
    StructuredBuffer<int> grad_log_scale_z,
    StructuredBuffer<int> grad_rot_w,
    StructuredBuffer<int> grad_rot_x,
    StructuredBuffer<int> grad_rot_y,
    StructuredBuffer<int> grad_rot_z,
    StructuredBuffer<int> grad_opacity,
    StructuredBuffer<int> grad_color_r,
    StructuredBuffer<int> grad_color_g,
    StructuredBuffer<int> grad_color_b,
    // Uniforms
    uniform AdamConfig cfg,
    uniform float pixels_per_step)
{
    uint idx = dtid.x;
    if (idx >= num_gaussians_counter[0]) return;

    // Load fixed-point grads; normalize by pixel count so the effective
    // per-step gradient is an average over contributing pixels (matches
    // what the loss normalization implies).
    float inv_pixels = 1.0 / max(pixels_per_step, 1.0);
    float3 g_mean = float3(
        fromFixed(grad_mean_x[idx]),
        fromFixed(grad_mean_y[idx]),
        fromFixed(grad_mean_z[idx])
    ) * inv_pixels;

    float3 g_log_scale = float3(
        fromFixed(grad_log_scale_x[idx]),
        fromFixed(grad_log_scale_y[idx]),
        fromFixed(grad_log_scale_z[idx])
    ) * inv_pixels;

    float4 g_rot = float4(
        fromFixed(grad_rot_w[idx]),
        fromFixed(grad_rot_x[idx]),
        fromFixed(grad_rot_y[idx]),
        fromFixed(grad_rot_z[idx])
    ) * inv_pixels;

    float g_op = fromFixed(grad_opacity[idx]) * inv_pixels;

    float3 g_color = float3(
        fromFixed(grad_color_r[idx]),
        fromFixed(grad_color_g[idx]),
        fromFixed(grad_color_b[idx])
    ) * inv_pixels;

    Gaussian3D g = gaussians[idx];
    AdamState s = adam_state[idx];

    g.mean = adam_vec3(g.mean, g_mean, s.mean_m, s.mean_v,
                       cfg.lr_mean, cfg.bias_correction1, cfg.bias_correction2);

    g.log_scale = adam_vec3(g.log_scale, g_log_scale, s.log_scale_m, s.log_scale_v,
                            cfg.lr_scale, cfg.bias_correction1, cfg.bias_correction2);

    g.rot_quat = adam_vec4(g.rot_quat, g_rot, s.rot_m, s.rot_v,
                           cfg.lr_rot, cfg.bias_correction1, cfg.bias_correction2);

    g.opacity = adam_scalar(g.opacity, g_op, s.opacity_m, s.opacity_v,
                            cfg.lr_opacity, cfg.bias_correction1, cfg.bias_correction2);
    // Opacity is a semantic [0, 1] bound (PLAN.md Q17 audit). Keep this.
    g.opacity = clamp(g.opacity, 0.0, 1.0);

    g.color_logit = adam_vec3(g.color_logit, g_color, s.color_m, s.color_v,
                              cfg.lr_color, cfg.bias_correction1, cfg.bias_correction2);

    gaussians[idx] = g;
    adam_state[idx] = s;
}
