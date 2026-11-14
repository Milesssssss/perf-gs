// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#include "../common/types.slang"
#include "../common/stbn.slang"

// SSIM over an 11x11 Gaussian window (sigma 1.5) with zero padding, as in
// 3DGS's F.conv2d. Both passes are LDS-tiled and separable: the 2D window is
// the outer product of a 1D Gaussian, so every 121-tap window sum becomes an
// 11-tap row sum followed by an 11-tap column sum. A 16x16 group stages its
// tile plus a 5-texel apron (26x26), row-filters 26x16 texels, then each
// thread column-filters its own pixel.

static const int WINDOW_RADIUS = 5;
static const int WINDOW = 2 * WINDOW_RADIUS + 1;
static const float C1 = 0.0001;  // (0.01)^2
static const float C2 = 0.0009;  // (0.03)^2

// exp(-k^2 / (2 * 1.5^2)) for k = -5..5 and 1 / (sum of the 121 2D weights).
static const float GAUSS[WINDOW] = {
    0.00386592014, 0.0285655008, 0.135335283, 0.411112291, 0.800737403, 1.0,
    0.800737403, 0.411112291, 0.135335283, 0.0285655008, 0.00386592014 };
static const float INV_WINDOW_SUM = 0.0707622378;

static const uint TILE = 16;                          // = numthreads per axis
static const uint THREADS = TILE * TILE;
static const uint APRON = TILE + 2 * WINDOW_RADIUS;   // staged texels per axis
static const uint APRON_TEXELS = APRON * APRON;
static const uint ROW_SUMS = APRON * TILE;            // row-filtered texels

bool insideFrame(int2 q, uint2 frame_dim)
{
    return q.x >= 0 && q.y >= 0 && q.x < int(frame_dim.x) && q.y < int(frame_dim.y);
}

// Apron texel i of the group's staging area, in image coordinates.
int2 apronTexel(uint3 group, uint i)
{
    return int2(group.xy * TILE) - WINDOW_RADIUS + int2(i % APRON, i / APRON);
}

// First staged texel of row-sum i (26 rows x 16 columns).
uint rowSumBase(uint i)
{
    return (i / TILE) * APRON + i % TILE;
}

// Row sum read by the thread's pixel at column-pass tap k.
uint columnTap(uint3 thread, int k)
{
    return (thread.y + k) * TILE + thread.x;
}

// --- Pass 1: compute the three back-projection maps m1,m2,m3 (per channel) ---
groupshared float gs_x[3][APRON_TEXELS];        // rendered, per channel
groupshared float gs_y[3][APRON_TEXELS];        // target, per channel
groupshared float gs_moments[5][ROW_SUMS];      // row sums of x, y, xx, yy, xy

[shader("compute")]
[numthreads(16, 16, 1)]
void ssimComputeMaps(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    RWTexture2D<float4> output_image,   // rendered (sRGB, premult-over-black)
    RWTexture2D<float4> target_image,   // GT (sRGB in .rgb)
    RWTexture2D<float4> ssim_m1,
    RWTexture2D<float4> ssim_m2,
    RWTexture2D<float4> ssim_m3,
    uniform uint2 frame_dim)
{
    uint t = gtid.y * TILE + gtid.x;
    for (uint i = t; i < APRON_TEXELS; i += THREADS)
    {
        int2 q = apronTexel(gid, i);
        float3 x = float3(0.0), y = float3(0.0);   // zero padding
        if (insideFrame(q, frame_dim))
        {
            x = output_image[uint2(q)].rgb;
            y = target_image[uint2(q)].rgb;
        }
        [unroll] for (uint c = 0; c < 3; c++)
        {
            gs_x[c][i] = x[c];
            gs_y[c][i] = y[c];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float3 mu_x, mu_y, Ex2, Ey2, Exy;
    [unroll] for (uint c = 0; c < 3; c++)
    {
        for (uint i = t; i < ROW_SUMS; i += THREADS)
        {
            uint base = rowSumBase(i);
            float sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
            [unroll] for (int k = 0; k < WINDOW; k++)
            {
                float x = gs_x[c][base + k];
                float y = gs_y[c][base + k];
                sx += GAUSS[k] * x;
                sy += GAUSS[k] * y;
                sxx += GAUSS[k] * x * x;
                syy += GAUSS[k] * y * y;
                sxy += GAUSS[k] * x * y;
            }
            gs_moments[0][i] = sx;
            gs_moments[1][i] = sy;
            gs_moments[2][i] = sxx;
            gs_moments[3][i] = syy;
            gs_moments[4][i] = sxy;
        }
        GroupMemoryBarrierWithGroupSync();

        float sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
        [unroll] for (int k = 0; k < WINDOW; k++)
        {
            uint i = columnTap(gtid, k);
            sx += GAUSS[k] * gs_moments[0][i];
            sy += GAUSS[k] * gs_moments[1][i];
            sxx += GAUSS[k] * gs_moments[2][i];
            syy += GAUSS[k] * gs_moments[3][i];
            sxy += GAUSS[k] * gs_moments[4][i];
        }
        mu_x[c] = sx;
        mu_y[c] = sy;
        Ex2[c] = sxx;
        Ey2[c] = syy;
        Exy[c] = sxy;
        if (c < 2)
            GroupMemoryBarrierWithGroupSync();   // next channel reuses gs_moments
    }

    uint2 p = gid.xy * TILE + gtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;

    float inv_w = INV_WINDOW_SUM;
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
//
// STBN sparse supervision masks THIS pass, not pass 1. Pass 1's maps describe
// how the loss at each window CENTER responds to the local statistics, and a
// selected pixel's window legitimately spans unselected neighbours — starving
// pass 1 would change the SSIM gradient itself. What we subsample is the
// chain-rule sum over pixels, dL/dtheta = sum_q (dL/dx_q)(dx_q/dtheta): keeping
// k of TILE_AREA of its terms and scaling by TILE_AREA/k estimates the FULL
// SSIM gradient, unbiased, with higher variance. Pass 1 therefore still runs
// over every pixel and its cost is not reduced by the mask.
groupshared float gs_maps[9][APRON_TEXELS];     // m1.rgb, m2.rgb, m3.rgb
groupshared float gs_map_rows[3][ROW_SUMS];     // one channel's row sums

[shader("compute")]
[numthreads(16, 16, 1)]
void ssimGradient(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    RWTexture2D<float4> output_image,
    RWTexture2D<float4> target_image,
    RWTexture2D<float4> ssim_m1,
    RWTexture2D<float4> ssim_m2,
    RWTexture2D<float4> ssim_m3,
    RWTexture2D<float4> dL_dC,
    StructuredBuffer<uint> stbn_table,
    uniform uint2 stbn_dims,
    uniform uint stbn_slice,
    uniform uint stbn_tile_pixels,
    uniform uint2 frame_dim,
    uniform float lambda_ssim,
    uniform float w_alpha,
    uniform float lambda_l2)
{
    uint t = gtid.y * TILE + gtid.x;
    for (uint i = t; i < APRON_TEXELS; i += THREADS)
    {
        int2 q = apronTexel(gid, i);
        float3 m1 = float3(0.0), m2 = float3(0.0), m3 = float3(0.0);   // zero padding
        if (insideFrame(q, frame_dim))
        {
            m1 = ssim_m1[uint2(q)].rgb;
            m2 = ssim_m2[uint2(q)].rgb;
            m3 = ssim_m3[uint2(q)].rgb;
        }
        [unroll] for (uint c = 0; c < 3; c++)
        {
            gs_maps[c][i] = m1[c];
            gs_maps[3 + c][i] = m2[c];
            gs_maps[6 + c][i] = m3[c];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Convolve the back-projection maps with the SAME Gaussian.
    float3 cm1, cm2, cm3;
    [unroll] for (uint c = 0; c < 3; c++)
    {
        for (uint i = t; i < ROW_SUMS; i += THREADS)
        {
            uint base = rowSumBase(i);
            float s1 = 0.0, s2 = 0.0, s3 = 0.0;
            [unroll] for (int k = 0; k < WINDOW; k++)
            {
                s1 += GAUSS[k] * gs_maps[c][base + k];
                s2 += GAUSS[k] * gs_maps[3 + c][base + k];
                s3 += GAUSS[k] * gs_maps[6 + c][base + k];
            }
            gs_map_rows[0][i] = s1;
            gs_map_rows[1][i] = s2;
            gs_map_rows[2][i] = s3;
        }
        GroupMemoryBarrierWithGroupSync();

        float s1 = 0.0, s2 = 0.0, s3 = 0.0;
        [unroll] for (int k = 0; k < WINDOW; k++)
        {
            uint i = columnTap(gtid, k);
            s1 += GAUSS[k] * gs_map_rows[0][i];
            s2 += GAUSS[k] * gs_map_rows[1][i];
            s3 += GAUSS[k] * gs_map_rows[2][i];
        }
        cm1[c] = s1;
        cm2[c] = s2;
        cm3[c] = s3;
        if (c < 2)
            GroupMemoryBarrierWithGroupSync();   // next channel reuses gs_map_rows
    }

    uint2 p = gid.xy * TILE + gtid.xy;
    if (p.x >= frame_dim.x || p.y >= frame_dim.y) return;

    // Unselected pixels: zero gradient.
    if (!stbnSelected(stbn_table, p, stbn_dims, stbn_slice, stbn_tile_pixels))
    {
        dL_dC[p] = float4(0.0);
        return;
    }
    float stbn_scale = stbnWeight(stbn_tile_pixels);

    float4 rendered = output_image[p];
    float4 target = target_image[p];
    float3 x = rendered.rgb;
    float3 y = target.rgb;   // sRGB, no linearization

    // dSSIM_total/dx(p); the loss uses (1 - SSIM), so negate it.
    float3 grad_ssim = -(cm1 + x * cm2 + y * cm3);

    // L1 gradient (matches l1_l2_loss.slang): d|x-y|/dx = sign(x-y).
    float3 grad_l1 = sign(x - y);

    float3 grad_rgb = (1.0 - lambda_ssim) * grad_l1 + lambda_ssim * grad_ssim;

    if (lambda_l2 > 0.0)
        grad_rgb += lambda_l2 * 2.0 * (x - y);

    float grad_alpha = 0.0;
    if (w_alpha > 0.0)
        grad_alpha = w_alpha * sign(rendered.a - target.a);

    dL_dC[p] = float4(grad_rgb * stbn_scale, grad_alpha * stbn_scale);
}
