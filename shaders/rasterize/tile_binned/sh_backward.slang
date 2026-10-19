// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Optional view-dependent SH backward (per-gaussian; runs only when sh_degree>0).
//
// The per-pixel blend backward already accumulated the color gradient into
// grad_color_{r,g,b}. With the effective-DC forward that equals SH_C0 *
// dL/d(rgb_clamped) = dL/d(effective DC) for the CLAMPED view-dependent color.
// This kernel (one thread per gaussian, INRIA's preprocess_backward) turns that
// per-gaussian dL/dRGB into gradients for the higher-band coefficients and,
// through the view direction, the mean:
//
//   dL/d(rgb_raw) = (grad_color / SH_C0) * clamp_mask
//   autodiff shColorFromMean(sh0, rest, mean, ...) with that seed gives
//     dp_rest.d = dL/d(features_rest),  dp_mean.d = dL/d(mean) via the direction.
//
// The DC (sh0) gradient is already grad_color for unclamped channels; we only
// zero the channels the forward clamped. The mean term is ADDED on top of the
// geometry mean gradient projection_backward wrote (this kernel runs after it).

#include "../../common/types.slang"
#include "../../primitives/gaussian/sh.slang"

#include "generated/execution_contract.slang"
static const uint CLUSTER_SIZE = CONTRACT_CLUSTER_SIZE;

[shader("compute")]
[numthreads(256, 1, 1)]
void shBackward(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    StructuredBuffer<uint> features_rest,
    RWStructuredBuffer<uint> features_rest_grad,
    // Accumulated color gradient (read; clamped channels zeroed for the DC term).
    RWStructuredBuffer<int64_t> grad_color_r,
    RWStructuredBuffer<int64_t> grad_color_g,
    RWStructuredBuffer<int64_t> grad_color_b,
    // 3D mean gradient: projection_backward is the geometry writer; this ADDS
    // the direction->mean term (runs after, single writer per gid).
    RWStructuredBuffer<int64_t> grad_mean_x,
    RWStructuredBuffer<int64_t> grad_mean_y,
    RWStructuredBuffer<int64_t> grad_mean_z,
    StructuredBuffer<uint> cluster_visible,
    uniform uint active_sh_degree,
    uniform uint rest_floats,
    uniform uint max_splats)
{
    uint gid = dtid.x;
    if (gid >= max_splats) return;
    if (gid >= num_gaussians_counter[0]) return;
    // Culled cluster: the blend backward never scattered grad_color to this
    // slot (0 tile pairs), and the rest-Adam early-outs the same way, so the
    // stale features_rest_grad is never read. Matches adam.slang.
    if (cluster_visible[gid / CLUSTER_SIZE] == 0u) return;

    Gaussian3D g = gaussians[gid];
    float3 cam = cameraCenterWorld(camera[0].view_matrix);
    int deg = int(active_sh_degree);

    float3 gaussianColor = float3(fromFixed(grad_color_r[gid]),
                       fromFixed(grad_color_g[gid]),
                       fromFixed(grad_color_b[gid]));

    // Load only the active bands: shColorFromMean + its bwd_diff ignore coeffs
    // beyond `deg`, so inactive higher-band gathers are dead work (those coeffs
    // are zero in storage and get zero gradient). active_coeffs excludes DC.
    uint active_coeffs = uint((deg + 1) * (deg + 1) - 1);
    float3 rest[SH_REST_COEFFS] = loadSHRestActive(features_rest, gid, rest_floats,
                                                   active_coeffs);

    // Primal (pre-clamp) color to reconstruct the forward's >=0 clamp mask.
    float3 rgb_raw = shColorFromMean(g.sh0, rest, g.mean, cam, deg);
    float3 mask = float3(rgb_raw.x > 0.0 ? 1.0 : 0.0,
                         rgb_raw.y > 0.0 ? 1.0 : 0.0,
                         rgb_raw.z > 0.0 ? 1.0 : 0.0);

    // grad_color == SH_C0 * dL/d(rgb_clamped); route through the clamp.
    float3 seed = (gaussianColor / SH_C0) * mask;

    var dp_rest = diffPair(rest);
    var dp_mean = diffPair(g.mean);
    bwd_diff(shColorFromMean)(g.sh0, dp_rest, dp_mean, cam, deg, seed);

    // Higher-band coefficient gradients, packed fp16 (2 scalars per uint) into
    // features_rest_grad (sole writer). dp_rest.d beyond the active degree is 0.
    uint fu = halfFeatUints(rest_floats);
    uint gbase = gid * fu;
    [ForceUnroll]
    for (uint u = 0u; u < uint(SH_FEAT_UINTS_MAX); ++u)
    {
        if (u >= fu) continue;
        uint s0 = u * 2u, s1 = s0 + 1u;
        uint k0 = s0 / 3u, c0 = s0 % 3u;
        float3 v0 = dp_rest.d[k0];
        float a = (c0 == 0u) ? v0.x : (c0 == 1u ? v0.y : v0.z);
        float b = 0.0;
        if (s1 < rest_floats)
        {
            // min() bounds the static index for the unrolled loop (s1==45 would
            // give k1==15, OOB for the size-15 array); the branch only runs for
            // real coeffs (k1 <= 14) so it is a no-op at runtime.
            uint k1 = min(s1 / 3u, uint(SH_REST_COEFFS) - 1u), c1 = s1 % 3u;
            float3 v1 = dp_rest.d[k1];
            b = (c1 == 0u) ? v1.x : (c1 == 1u ? v1.y : v1.z);
        }
        features_rest_grad[gbase + u] = packHalf2x(a, b);
    }

    // DC (sh0) gradient == grad_color for unclamped channels; zero the ones the
    // forward clamped (leaves unclamped values bit-exact for Adam).
    if (mask.x == 0.0) grad_color_r[gid] = int64_t(0);
    if (mask.y == 0.0) grad_color_g[gid] = int64_t(0);
    if (mask.z == 0.0) grad_color_b[gid] = int64_t(0);

    // Direction->mean term, added on top of the geometry mean gradient.
    grad_mean_x[gid] = grad_mean_x[gid] + toFixed(dp_mean.d.x);
    grad_mean_y[gid] = grad_mean_y[gid] + toFixed(dp_mean.d.y);
    grad_mean_z[gid] = grad_mean_z[gid] + toFixed(dp_mean.d.z);
}
