// Per-pixel binary error mask for FastGS importance scoring.
//
// Two-pass approach:
//   Pass 1 (computeL1MinMax): compute per-pixel L1, find global min/max via atomics.
//   Pass 2 (computeErrorMask): normalize L1 to [0,1] using min/max, threshold.
//
// This matches FastGS: l1_norm = (l1 - min) / (max - min), mask = l1_norm > thresh.

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

// Pass 1: compute L1 per pixel, write to l1_map, and atomically update min/max.
[shader("compute")]
[numthreads(16, 16, 1)]
void computeL1MinMax(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,
    RWTexture2D<float4> target_image,
    RWTexture2D<float> l1_map,
    RWStructuredBuffer<uint> minmax_buf,
    uniform uint2 frame_dim)
{
    uint2 p = dtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;

    float3 C = output_image[p].rgb;
    float3 T_srgb = target_image[p].rgb;
    float3 T_linear = srgb_to_linear(T_srgb);

    float l1 = (abs(C.x - T_linear.x) + abs(C.y - T_linear.y) + abs(C.z - T_linear.z)) / 3.0;
    l1_map[p] = l1;

    // Atomic min/max using uint reinterpretation (works because positive floats
    // sort the same as uint when interpreted via asuint).
    uint l1_bits = asuint(l1);
    uint dummy;
    InterlockedMin(minmax_buf[0], l1_bits, dummy);
    InterlockedMax(minmax_buf[1], l1_bits, dummy);
}

// Pass 2: normalize and threshold.
[shader("compute")]
[numthreads(16, 16, 1)]
void computeErrorMask(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float> l1_map,
    RWStructuredBuffer<uint> minmax_buf,
    RWTexture2D<uint> error_mask,
    uniform uint2 frame_dim,
    uniform float loss_thresh)
{
    uint2 p = dtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;

    float l1_min = asfloat(minmax_buf[0]);
    float l1_max = asfloat(minmax_buf[1]);
    float range = l1_max - l1_min;

    float l1 = l1_map[p];
    float l1_norm = (range > 1e-8) ? (l1 - l1_min) / range : 0.0;

    error_mask[p] = (l1_norm > loss_thresh) ? 1u : 0u;
}
