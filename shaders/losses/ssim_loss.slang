// Combined L1 + SSIM photometric loss with alpha supervision.
//
// 3DGS loss: L = (1 - lambda_ssim) * L1 + lambda_ssim * (1 - SSIM)
//
// The SSIM term is the MEAN over pixels of a per-pixel SSIM computed on an
// 11x11 Gaussian-weighted window (sigma=1.5). Its gradient is NOT local: a
// pixel x(p) enters the window of every neighbour q within the radius, so
//   dL_ssim/dx(p) = sum_{q : |q-p|<=R} d(1-SSIM(q))/dx(p).
// The exact gradient is therefore a TWO-PASS operation (this is what the
// reference fused-SSIM / autograd implementations do):
//
//   Pass 1 (ssimComputeMaps): for each window centre q, compute the local
//     statistics (mu_x, mu_y, sigma_x2, sigma_y2, sigma_xy), the partial
//     derivatives of SSIM(q) w.r.t. mu_x/sigma_x2/sigma_xy, and reduce them
//     into three "back-projection" maps:
//         m1 = dS/dmu_x - 2*mu_x*dS/dsigma_x2 - mu_y*dS/dsigma_xy
//         m2 = 2 * dS/dsigma_x2
//         m3 = dS/dsigma_xy
//     (derivation: mu_x, sigma_x2, sigma_xy are the only x-dependent stats;
//      d(mu_x)/dx_q = w, d(sigma_x2)/dx_q = 2w(x_q-mu_x),
//      d(sigma_xy)/dx_q = w(y_q-mu_y), and the w's are supplied by the
//      pass-2 convolution, so the maps hold the per-window coefficients.)
//
//   Pass 2 (ssimGradient): convolve m1,m2,m3 with the SAME Gaussian and
//     assemble the per-pixel SSIM gradient
//         dSSIM_total/dx(p) = conv(m1)(p) + x(p)*conv(m2)(p) + y(p)*conv(m3)(p)
//     then blend with the L1 gradient and the alpha term and write dL_dC.
//
// NOTE ON THE WINDOW LOOP: the two 11x11 (121-tap) loops below are NOT the
// bottleneck — this convolution is cache-bound, not ALU/tap-bound. Adjacent
// threads read heavily-overlapping windows straight from texture cache, so the
// taps are cheap. A "separable" H-then-V rewrite was tried and measured ~70%
// SLOWER: splitting into 4 passes forces the intermediate row-sums (and the
// maps) through global-memory scratch textures, and that added bandwidth +
// two extra dispatch barriers cost far more than the taps it saved. Keep the
// fused 121-tap form unless a future rewrite stages the tile+halo in LDS
// (groupshared) to avoid the global round-trip — and only if Nsight confirms it.
//
// Gradients are in the SUM-over-pixels convention (no 1/N), matching the L1
// kernel in l1_l2_loss.slang; the /(W*H) normalization happens once in Adam.
// COLORSPACE: SSIM is evaluated in sRGB (render and target are both sRGB,
// compared directly — no linearization), matching the L1 term and matching
// the INRIA 3DGS / gsplat / SADGS convention.

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
            int2 q = int2(p) + int2(dj, di);
            q = clamp(q, int2(0, 0), int2(frame_dim) - 1);
            float w = gaussWeightRaw(di, dj);
            w_sum += w;

            float3 xq = output_image[uint2(q)].rgb;
            float3 yq = target_image[uint2(q)].rgb;   // sRGB, no linearization

            mu_x += w * xq;  mu_y += w * yq;
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
            q = clamp(q, int2(0, 0), int2(frame_dim) - 1);
            float w = gaussWeightRaw(di, dj);
            cm1 += w * ssim_m1[uint2(q)].rgb;
            cm2 += w * ssim_m2[uint2(q)].rgb;
            cm3 += w * ssim_m3[uint2(q)].rgb;
        }
    }

    // dSSIM_total/dx(p); loss uses (1 - SSIM) so negate. Sum convention.
    float3 grad_ssim = -(cm1 + x * cm2 + y * cm3);

    // L1 gradient (matches l1_l2_loss.slang): d|x-y|/dx = sign(x-y).
    float3 grad_l1 = sign(x - y);

    float3 grad_rgb = (1.0 - lambda_ssim) * grad_l1 + lambda_ssim * grad_ssim;
    // Optional additive L2 term (SADGS lambda_l2 * mean((C-T)^2)); sum
    // convention, so per-pixel gradient is lambda_l2 * 2*(x-y).
    if (lambda_l2 > 0.0)
        grad_rgb += lambda_l2 * 2.0 * (x - y);

    float grad_alpha = 0.0;
    if (w_alpha > 0.0)
        grad_alpha = w_alpha * sign(rendered.a - target.a);

    dL_dC[p] = float4(grad_rgb, grad_alpha);
}
