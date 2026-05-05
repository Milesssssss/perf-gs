// Backward rasterizer.
//
// One thread per pixel. Walks the tile's sorted gaussian list in reverse,
// bounded by the per-pixel stop index the forward wrote
// (gaussian_count_per_pixel), so the gaussians we iterate are exactly
// those that contributed to the pixel (PLAN.md Q11).
//
// Derivation reference: bad_attempt1/shaders/backward_rasterize.slang
// has a multi-page derivation of dC/dα and dT_j/dα_i — we reproduce the
// final formulas here and trust Slang's ``bwd_diff`` for the rest of the
// chain (eval2D → proj2D → params; sigmoid → color_logit).
//
// Forward:
//   α_i = evalGaussian2DFromCov(pixel, mean_2d_i, cov2d_i, opacity_i)
//   c_i = sigmoid(color_logit_i)
//   C = Σ_i c_i · α_i · T_i      T_{i+1} = T_i · (1 - α_i)
//
// Given dL/dC = dL_dC[pixel]:
//   dC/dc_i = α_i · T_i
//   dC/dα_i = c_i · T_i - (C_behind_i) / (1 - α_i)
//     where C_behind_i = Σ_{j>i} c_j · α_j · T_j  (built while walking back)
//
// Gradient accumulation: InterlockedAdd on int32 fixed-point buffers
// (SoA). No groupshared flush in v1 — revisit if profiling shows this is
// the bottleneck (PLAN.md v1 D4).

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/eval.slang"

static const uint TILE_SIZE = 16;

// Must match forward.slang. Semantic cap on alpha at 0.99. Gradient
// does not flow through the saturating branch of min() — we multiply
// the upstream alpha gradient by 1[alpha_raw <= ALPHA_MAX] before
// propagating.
static const float ALPHA_MAX = 0.99;

// Semantic alpha floor: below this a gaussian's contribution to the
// pixel is sub-8-bit-output, i.e. the pixel is essentially unaffected
// by this gaussian. We still flow gradients (to keep the FD test happy)
// but we skip updating the per-iter 2D-grad vector accumulators
// because those are meant to answer "is this gaussian actively
// contributing?" — which is not true for sub-display contributions.
// Without this gate, the signal gets diluted by a bunch of ~zero
// contributions and the densify threshold becomes too permissive.
static const float ALPHA_MIN_TRACK = 1.0 / 255.0;

[shader("compute")]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void backwardRasterize(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<Projected2DGaussian> projected_2d,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,
    StructuredBuffer<CameraData> camera,
    RWTexture2D<float4> output_image,
    RWTexture2D<float> output_transmittance,
    RWTexture2D<uint> gaussian_count_per_pixel,
    RWTexture2D<float4> dL_dC_texture,
    RWStructuredBuffer<int> grad_mean_x,
    RWStructuredBuffer<int> grad_mean_y,
    RWStructuredBuffer<int> grad_mean_z,
    RWStructuredBuffer<int> grad_log_scale_x,
    RWStructuredBuffer<int> grad_log_scale_y,
    RWStructuredBuffer<int> grad_log_scale_z,
    RWStructuredBuffer<int> grad_rot_w,
    RWStructuredBuffer<int> grad_rot_x,
    RWStructuredBuffer<int> grad_rot_y,
    RWStructuredBuffer<int> grad_rot_z,
    RWStructuredBuffer<int> grad_opacity,
    RWStructuredBuffer<int> grad_color_r,
    RWStructuredBuffer<int> grad_color_g,
    RWStructuredBuffer<int> grad_color_b,
    // v2 densification tracking — per-iter absgrad accumulators.
    // After the backward pass completes, a separate kernel
    // (accumulate_iter) folds (grad_2d_x_iter[g], grad_2d_y_iter[g]) into
    // the window-level buffers by taking the norm of the iter-sum and
    // incrementing visibility count by 1.
    //
    // We accumulate |grad| per pixel (absgrad semantics, gsplat default).
    // This prevents cancellation for gaussians straddling edges — those
    // are exactly the ones that need splitting.
    RWStructuredBuffer<int> grad_2d_x_iter,
    RWStructuredBuffer<int> grad_2d_y_iter)
{
    uint2 pixel = dtid.xy;
    uint2 frame_dim_px = uint2(camera[0].image_size);
    if (pixel.x >= frame_dim_px.x || pixel.y >= frame_dim_px.y) return;

    uint2 num_tiles = (frame_dim_px + TILE_SIZE - 1) / TILE_SIZE;
    uint tile_idx = gid.y * num_tiles.x + gid.x;
    uint t_start = tile_start[tile_idx];

    // ``processed`` is the number of gaussians forward walked (after any
    // early-break). The backward walks them in reverse using the same
    // t_start offset into the globally-sorted array.
    uint processed = gaussian_count_per_pixel[pixel];
    if (processed == 0) return;

    float2 pixel_center = float2(pixel) + 0.5;
    float3 dL_dC = dL_dC_texture[pixel].rgb;

    // T_post holds T_{i+1}: the transmittance *after* processing gaussian i.
    // For the last gaussian processed, that's output_transmittance[pixel].
    float T_post = output_transmittance[pixel];
    float3 C_behind = float3(0.0, 0.0, 0.0);

    float4x4 world_to_view = camera[0].view_matrix;
    float4x4 view_to_clip  = camera[0].proj_matrix;
    float2 frame_dim = camera[0].image_size;

    for (int i = int(processed) - 1; i >= 0; --i)
    {
        uint g_idx = sorted_gaussian_ids[t_start + uint(i)];
        Gaussian3D g = gaussians[g_idx];
        Projected2DGaussian proj = projected_2d[g_idx];

        float alpha_raw = evalGaussian2DFromCov(pixel_center, proj.mean, proj.cov2d, g.opacity);
        // Cap to match forward (semantic probability bound, see forward.slang).
        float alpha = min(alpha_raw, ALPHA_MAX);
        // Mask used to zero out the alpha-side gradient when min() saturates
        // (gradient of min(x, c) w.r.t. x is 1 when x < c, else 0).
        float alpha_grad_scale = (alpha_raw <= ALPHA_MAX) ? 1.0 : 0.0;

        // Recover T_i from T_{i+1} by undoing the (1 - α_i) factor. With
        // the cap, (1 - alpha) >= 0.01 > 0, so the division is always
        // well-conditioned.
        float denom = 1.0 - alpha;
        float T_i = T_post / denom;

        var dp_color_logit = diffPair(g.color_logit);
        float3 c_i = sigmoid3(dp_color_logit.p);

        // --- color gradient: dL/dc_i = dL/dC * α_i * T_i ---
        float3 dL_dc_i = dL_dC * (alpha * T_i);
        bwd_diff(sigmoid3)(dp_color_logit, dL_dc_i);

        // --- alpha gradient: dL/dα_i = dL/dC · dC/dα_i ---
        // dC/dα_i = c_i · T_i - C_behind / (1 - α_i)
        float3 dC_dalpha = c_i * T_i - C_behind / denom;
        float dL_dalpha = dot(dL_dC, dC_dalpha);
        // Zero the gradient when the forward's alpha was clipped.
        float dL_d_alpha_raw = dL_dalpha * alpha_grad_scale;

        // Chain α back through evalGaussian2DFromCov to get grads of
        // mean_2d, cov2d, opacity.
        var dp_mean_2d = diffPair(proj.mean);
        var dp_cov2d   = diffPair(proj.cov2d);
        var dp_opacity = diffPair(g.opacity);
        bwd_diff(evalGaussian2DFromCov)(
            pixel_center, dp_mean_2d, dp_cov2d, dp_opacity, dL_d_alpha_raw);

        // Chain mean_2d and cov2d back through projectGaussian3DTo2D to
        // get grads of mean, log_scale, rot_quat. For ``bwd_diff`` of
        // a function with ``out`` diff-params, the upstream gradients
        // are passed as plain values (not DifferentialPair). dp_mean_2d.d
        // and dp_cov2d.d were populated by the eval bwd_diff above.
        var dp_mean      = diffPair(g.mean);
        var dp_log_scale = diffPair(g.log_scale);
        var dp_rot_quat  = diffPair(g.rot_quat);
        float2 d_mean_2d = dp_mean_2d.d;
        float3 d_cov2d   = dp_cov2d.d;
        float  d_depth   = 0.0;
        bwd_diff(projectGaussian3DTo2D)(
            dp_mean, dp_log_scale, dp_rot_quat,
            world_to_view, view_to_clip, frame_dim,
            d_mean_2d, d_cov2d, d_depth);

        // --- atomic fixed-point accumulation into global SoA buffers ---
        InterlockedAdd(grad_mean_x[g_idx], toFixed(dp_mean.d.x));
        InterlockedAdd(grad_mean_y[g_idx], toFixed(dp_mean.d.y));
        InterlockedAdd(grad_mean_z[g_idx], toFixed(dp_mean.d.z));

        InterlockedAdd(grad_log_scale_x[g_idx], toFixed(dp_log_scale.d.x));
        InterlockedAdd(grad_log_scale_y[g_idx], toFixed(dp_log_scale.d.y));
        InterlockedAdd(grad_log_scale_z[g_idx], toFixed(dp_log_scale.d.z));

        InterlockedAdd(grad_rot_w[g_idx], toFixed(dp_rot_quat.d.x));
        InterlockedAdd(grad_rot_x[g_idx], toFixed(dp_rot_quat.d.y));
        InterlockedAdd(grad_rot_y[g_idx], toFixed(dp_rot_quat.d.z));
        InterlockedAdd(grad_rot_z[g_idx], toFixed(dp_rot_quat.d.w));

        InterlockedAdd(grad_opacity[g_idx], toFixed(dp_opacity.d));

        InterlockedAdd(grad_color_r[g_idx], toFixed(dp_color_logit.d.x));
        InterlockedAdd(grad_color_g[g_idx], toFixed(dp_color_logit.d.y));
        InterlockedAdd(grad_color_b[g_idx], toFixed(dp_color_logit.d.z));

        // Densification tracking signals (absgrad semantics). Gated on
        // alpha >= ALPHA_MIN_TRACK so sub-display contributions don't
        // dilute the per-iter sum.
        //
        // We accumulate |dp_mean_2d.d| per pixel (gsplat absgrad=True
        // default). accumulate_iter then computes
        // norm(sum_abs_x, sum_abs_y) per iter. Absgrad prevents
        // cancellation for gaussians straddling edges — exactly those
        // that need splitting.
        if (alpha >= ALPHA_MIN_TRACK)
        {
            InterlockedAdd(grad_2d_x_iter[g_idx], toFixed(abs(dp_mean_2d.d.x)));
            InterlockedAdd(grad_2d_y_iter[g_idx], toFixed(abs(dp_mean_2d.d.y)));
        }

        // Walk T and C_behind back for the next iteration.
        C_behind += c_i * alpha * T_i;
        T_post = T_i;
    }
}
