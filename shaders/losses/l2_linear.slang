// L1 loss in linear color space with optional alpha supervision.
//
// Reads the rendered output (linear RGB + alpha in .a) and the ground-truth
// target (sRGB-encoded RGB + alpha in .a), converts target RGB to linear,
// writes the per-pixel gradient ``dL/dC`` to a UAV texture that the backward
// rasterizer reads.
//
// L1 matches gsplat's photometric objective (F.l1_loss). Its constant-
// magnitude gradient sharpens edges, where L2's quadratic penalty smears
// low-opacity gaussians across error regions (elongation).
//
// L1 loss per pixel: L_pixel = sum_c |C_c - T_c| + w_alpha * |A_r - A_t|
// dL/dC_c = sign(C_c - T_c)
// dL/dA   = w_alpha * sign(A_r - A_t)  (stored in dL_dC.a)

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

    // the subgradient of an absolute value is the sign of its argument
    float3 grad_rgb = sign(C - T_linear);

    float grad_alpha = 0.0;
    if (w_alpha > 0.0)
    {
        float A_rendered = rendered.a;
        float A_target = target.a;
        grad_alpha = w_alpha * sign(A_rendered - A_target);
    }

    dL_dC[p] = float4(grad_rgb, grad_alpha);
}
