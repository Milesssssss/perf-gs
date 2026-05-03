// L2 loss in linear color space.
//
// Reads the rendered output (linear RGB) and the ground-truth target
// (sRGB-encoded PNG values), converts target to linear, writes the
// per-pixel gradient ``dL/dC`` to a UAV texture that the backward
// rasterizer reads. No gamma in the backward path (PLAN.md Q12).
//
// L2 loss per pixel: L_pixel = sum_c (C_c - T_c)^2
// dL/dC_c = 2 * (C_c - T_c)
// Loss is averaged over pixels in Python by dividing the accumulated
// parameter gradients by pixel count — the shader itself does not
// normalize so the per-pixel gradient is just the raw 2*(C - T).

static float srgb_to_linear_scalar(float c)
{
    // IEC 61966-2-1 sRGB -> linear.
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
    uniform uint2 frame_dim)
{
    uint2 p = dtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;

    float3 C = output_image[p].rgb;
    float3 T_srgb = target_image[p].rgb;
    float3 T_linear = srgb_to_linear(T_srgb);

    float3 grad = 2.0 * (C - T_linear);
    dL_dC[p] = float4(grad, 0.0);
}
