// L2 loss in linear color space with optional alpha supervision.
//
// Reads the rendered output (linear RGB + alpha in .a) and the ground-truth
// target (sRGB-encoded RGB + alpha in .a), converts target RGB to linear,
// writes the per-pixel gradient ``dL/dC`` to a UAV texture that the backward
// rasterizer reads.
//
// L2 loss per pixel: L_pixel = sum_c (C_c - T_c)^2 + w_alpha * (A_r - A_t)^2
// dL/dC_c = 2 * (C_c - T_c)
// dL/dA   = 2 * w_alpha * (A_r - A_t)  (stored in dL_dC.a)

static float srgb_to_linear_scalar(float c)
{
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

static float3 srgb_to_linear(float3 c)
{
    return float3(
        srgb_to_linear_scalar(c.r),
        srgb_to_linear_scalar(c.g),
        srgb_to_linear_scalar(c.b)
    );
}

[shader("compute")]
[numthreads(16, 16, 1)]
void computeLossGrad(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,
    RWTexture2D<float4> target_image,
    RWTexture2D<float4> dL_dC,
    uniform uint2 frame_dim,
    uniform float w_alpha)
{
    uint2 p = dtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;

    float4 rendered = output_image[p];
    float4 target = target_image[p];

    float3 C = rendered.rgb;
    float3 T_srgb = target.rgb;
    float3 T_linear = srgb_to_linear(T_srgb);

    float3 grad_rgb = 2.0 * (C - T_linear);

    float grad_alpha = 0.0;
    if (w_alpha > 0.0)
    {
        float A_rendered = rendered.a;
        float A_target = target.a;
        grad_alpha = 2.0 * w_alpha * (A_rendered - A_target);
    }

    dL_dC[p] = float4(grad_rgb, grad_alpha);
}
