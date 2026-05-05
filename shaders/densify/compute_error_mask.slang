// Per-pixel binary error mask for FastGS importance scoring.
//
// Reads rendered output (linear RGB) and ground-truth target (sRGB),
// computes per-channel L1 in linear space, averages, and thresholds.
// Writes 1 to error_mask if the pixel has high error, 0 otherwise.

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
void computeErrorMask(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,
    RWTexture2D<float4> target_image,
    RWTexture2D<uint> error_mask,
    uniform uint2 frame_dim,
    uniform float loss_thresh)
{
    uint2 p = dtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;

    float3 C = output_image[p].rgb;
    float3 T_srgb = target_image[p].rgb;
    float3 T_linear = srgb_to_linear(T_srgb);

    float l1 = (abs(C.x - T_linear.x) + abs(C.y - T_linear.y) + abs(C.z - T_linear.z)) / 3.0;
    error_mask[p] = (l1 > loss_thresh) ? 1u : 0u;
}
