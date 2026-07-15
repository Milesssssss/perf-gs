#include "../common/types.slang"

static const int WINDOW_RADIUS = 5;   // 11x11 window
static const float SIGMA = 1.5;
static const float C1 = 0.0001;  // (0.01)^2
static const float C2 = 0.0009;  // (0.03)^2

// Unnormalized Gaussian weight for window offset (di,dj) in [-R,R].
static float gaussWeightRaw(int di, int dj)
{
    return exp(-0.5 * float(di * di + dj * dj) / (SIGMA * SIGMA));
}

// --- Pass 1: compute the three back-projection maps m1,m2,m3 (per channel) ---
[shader("compute")]
[numthreads(16, 16, 1)]
void ssimComputeMaps(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,   // rendered (sRGB, premult-over-black)
    RWTexture2D<float4> target_image,   // GT (sRGB in .rgb)
    RWTexture2D<float4> ssim_m1,
    RWTexture2D<float4> ssim_m2,
    RWTexture2D<float4> ssim_m3,
    uniform uint2 frame_dim)
{
    uint2 p = dtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;

    float3 mu_x = float3(0.0), mu_y = float3(0.0);
    float3 Ex2 = float3(0.0), Ey2 = float3(0.0), Exy = float3(0.0);
    float w_sum = 0.0;

    for (int di = -WINDOW_RADIUS; di <= WINDOW_RADIUS; di++)
    {
        for (int dj = -WINDOW_RADIUS; dj <= WINDOW_RADIUS; dj++)
        {
            float w = gaussWeightRaw(di, dj);
            w_sum += w;

            int2 q = int2(p) + int2(dj, di);

            bool valid =
                q.x >= 0 && q.y >= 0 &&
                q.x < int(frame_dim.x) &&
                q.y < int(frame_dim.y);

            if (!valid)
                continue; // zero padding

            float3 xq = output_image[uint2(q)].rgb;
            float3 yq = target_image[uint2(q)].rgb;

            mu_x += w * xq;
            mu_y += w * yq;
            Ex2  += w * xq * xq;
            Ey2  += w * yq * yq;
            Exy  += w * xq * yq;
        }
    }

    float inv_w = 1.0 / w_sum;
    mu_x *= inv_w;  mu_y *= inv_w;
    float3 sigma_x2 = Ex2 * inv_w - mu_x * mu_x;
    float3 sigma_y2 = Ey2 * inv_w - mu_y * mu_y;
    float3 sigma_xy = Exy * inv_w - mu_x * mu_y;

    float3 A = 2.0 * mu_x * mu_y + C1;
    float3 B = 2.0 * sigma_xy + C2;
    float3 Cc = mu_x * mu_x + mu_y * mu_y + C1;
    float3 D = sigma_x2 + sigma_y2 + C2;

    float3 CcD = Cc * D;
    // Partial derivatives of SSIM(q) w.r.t. the local statistics.
    float3 dS_dmux = 2.0 * B * (mu_y * Cc - mu_x * A) / (Cc * Cc * D);
    float3 dS_dsx2 = -A * B / (Cc * D * D);
    float3 dS_dsxy = 2.0 * A / CcD;

    // The window weights w (and inv_w) are re-applied in pass 2 via the same
    // Gaussian convolution, so the maps store the pre-weight coefficients.
    // sigma stats used the same inv_w-normalized window, so fold inv_w in here
    // once: d(mu_x)/dx_q = w*inv_w, d(sigma_x2)/dx_q = 2*w*inv_w*(x_q-mu_x),
    // d(sigma_xy)/dx_q = w*inv_w*(y_q-mu_y). The pass-2 conv supplies w; inv_w
    // is constant across the window (uniform Gaussian) so we bake it in now.
    float3 m1 = (dS_dmux - 2.0 * mu_x * dS_dsx2 - mu_y * dS_dsxy) * inv_w;
    float3 m2 = (2.0 * dS_dsx2) * inv_w;
    float3 m3 = (dS_dsxy) * inv_w;

    ssim_m1[p] = float4(m1, 0.0);
    ssim_m2[p] = float4(m2, 0.0);
    ssim_m3[p] = float4(m3, 0.0);
}

// --- Pass 2: convolve the maps, assemble SSIM grad, blend with L1 + alpha ---
[shader("compute")]
[numthreads(16, 16, 1)]
void ssimGradient(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,
    RWTexture2D<float4> target_image,
    RWTexture2D<float4> ssim_m1,
    RWTexture2D<float4> ssim_m2,
    RWTexture2D<float4> ssim_m3,
    RWTexture2D<float4> dL_dC,
    uniform uint2 frame_dim,
    uniform float lambda_ssim,
    uniform float w_alpha,
    uniform float lambda_l2)
{
    uint2 p = dtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;

    float4 rendered = output_image[p];
    float4 target = target_image[p];
    float3 x = rendered.rgb;
    float3 y = target.rgb;   // sRGB, no linearization

    // Convolve the back-projection maps with the SAME Gaussian.
    float3 cm1 = float3(0.0), cm2 = float3(0.0), cm3 = float3(0.0);
    for (int di = -WINDOW_RADIUS; di <= WINDOW_RADIUS; di++)
    {
        for (int dj = -WINDOW_RADIUS; dj <= WINDOW_RADIUS; dj++)
        {
            int2 q = int2(p) + int2(dj, di);

            bool valid =
                q.x >= 0 && q.y >= 0 &&
                q.x < int(frame_dim.x) &&
                q.y < int(frame_dim.y);

            if (!valid)
                continue;

            float w = gaussWeightRaw(di, dj);

            cm1 += w * ssim_m1[uint2(q)].rgb;
            cm2 += w * ssim_m2[uint2(q)].rgb;
            cm3 += w * ssim_m3[uint2(q)].rgb;
        }
    }

    //dSSIM_total/dx(p); rhe loss uses (1 - SSIM), so negate it!
    float3 grad_ssim = -(cm1 + x * cm2 + y * cm3);

    // L1 gradient (matches l1_l2_loss.slang): d|x-y|/dx = sign(x-y).
    float3 grad_l1 = sign(x - y);

    float3 grad_rgb = (1.0 - lambda_ssim) * grad_l1 + lambda_ssim * grad_ssim;

    if (lambda_l2 > 0.0)
        grad_rgb += lambda_l2 * 2.0 * (x - y);

    float grad_alpha = 0.0;
    if (w_alpha > 0.0)
        grad_alpha = w_alpha * sign(rendered.a - target.a);

    dL_dC[p] = float4(grad_rgb, grad_alpha);
}
