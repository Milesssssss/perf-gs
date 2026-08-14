// L1 (+ optional L2) photometric loss in sRGB color space, with optional
// alpha supervision. (Formerly l2_linear.slang, which — despite its name —
// only computed L1; renamed to reflect that it now handles both.)
//
// COLORSPACE: matches INRIA 3DGS / gsplat / SADGS — render output and target
// are BOTH sRGB [0,1] and compared directly, NO linearization. The renderer
// composites gamma-encoded values (a well-known convention in this lineage:
// physically imperfect for alpha blending, but the SH DC term is fit to
// whatever space we supervise, so it's self-consistent; and it aligns the
// objective with the reported metric, which is sRGB PSNR).
//
// Reads output (sRGB RGB + alpha in .a) and target (sRGB RGB + alpha in .a),
// writes the per-pixel gradient dL/dC to a UAV texture that the backward
// rasterizer reads.
//
// Combined objective per pixel (SUM-over-pixels convention — no 1/N, matching
// the backward rasterizer's accumulation):
//     L = sum_c [ |C_c - T_c|  +  lambda_l2 * (C_c - T_c)^2 ]
//         + w_alpha * |A_r - A_t|
//     dL/dC_c = sign(C_c - T_c) + lambda_l2 * 2*(C_c - T_c)
//     dL/dA   = w_alpha * sign(A_r - A_t)   (stored in dL_dC.a)
//
// L1's constant-magnitude gradient sharpens edges (gsplat's F.l1_loss); the
// optional L2 term (lambda_l2 > 0) adds SADGS's extra squared penalty, whose
// gradient scales with the residual — pulls large errors down faster early.
// lambda_l2 = 0 (default) reproduces the pure-L1 behavior exactly.

#include "../common/stbn.slang"

[shader("compute")]
[numthreads(16, 16, 1)]
void computeLossGrad(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,
    RWTexture2D<float4> target_image,
    RWTexture2D<float4> dL_dC,
    // STBN sparse supervision (all runtime; stbn_tile_pixels == TILE_AREA
    // disables it and the table is then never read). See common/stbn.slang.
    StructuredBuffer<uint> stbn_table,
    uniform uint2 stbn_dims,
    uniform uint stbn_slice,
    uniform uint stbn_tile_pixels,
    uniform uint2 frame_dim,
    uniform float w_alpha,
    uniform float lambda_l2)
{
    uint2 p = dtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;

    // Sparse supervision: a pixel this slice does not sample gets a ZERO
    // gradient, which is what makes the backward's skip of it exact rather than
    // approximate — it would have contributed nothing anyway. dL_dC is fully
    // rewritten every iteration, so the zero must be WRITTEN, not just skipped.
    if (!stbnSelected(stbn_table, p, stbn_dims, stbn_slice, stbn_tile_pixels))
    {
        dL_dC[p] = float4(0.0);
        return;
    }
    // We keep k of TILE_AREA terms of the per-pixel sum; scale by TILE_AREA/k so
    // the estimate is unbiased and the LR needs no retuning. Applied HERE rather
    // than in the backward so every backward variant (per-bucket, per-splat,
    // per-pixel) inherits it from the one place the mask is defined.
    float stbn_scale = stbnWeight(stbn_tile_pixels);

    float4 rendered = output_image[p];
    float4 target = target_image[p];

    float3 C = rendered.rgb;
    float3 T = target.rgb;   // sRGB, compared directly (no linearization)

    float3 diff = C - T;
    // the subgradient of an absolute value is the sign of its argument :)
    float3 grad_rgb = sign(diff);                    // L1
    if (lambda_l2 > 0.0)
        grad_rgb += lambda_l2 * 2.0 * diff;          // + lambda_l2 * d/dC (C-T)^2

    float grad_alpha = 0.0;
    if (w_alpha > 0.0)
    {
        float A_rendered = rendered.a;
        float A_target = target.a;
        grad_alpha = w_alpha * sign(A_rendered - A_target);
    }

    dL_dC[p] = float4(grad_rgb * stbn_scale, grad_alpha * stbn_scale);
}
