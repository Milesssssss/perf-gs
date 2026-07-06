// Per-gaussian projection backward (split out of the blend backward).
//
// The blend backward (backward_persplat.slang) parallelizes over splats-in-
// tiles, so a gaussian touching K tiles is visited K times. Running the
// heavy 3D projection backward (quaternion -> R -> cov3d -> cov2d Jacobian,
// mean3d -> mean2d) inside that loop pays that whole chain K times per
// gaussian. This kernel instead runs it EXACTLY ONCE per gaussian.
//
// This is exact, not an approximation: the projection backward is linear in
// its seed gradient (dp.d = J^T * seed, with J evaluated at the gaussian's
// tile-invariant parameters). So summing the per-tile 2D gradients first and
// applying J^T once equals applying J^T per tile and summing:
//   sum_k J^T g_k == J^T (sum_k g_k).
// The blend backward accumulates (sum_k g_k) into grad_mean2d_* / grad_cov2d_*;
// this kernel consumes that sum. Mirrors faster-gaussian-splatting's
// preprocess_backward_cu (one thread per primitive).
//
// One thread per gaussian. Sole writer of the 3D parameter gradients, so it
// writes them directly (no atomics needed). Opacity/color gradients are 2D-
// level already and are written by the blend backward, not here.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void projectionBackward(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    // 2D gradient seeds, summed over all tiles by the blend backward.
    StructuredBuffer<int64_t> grad_mean2d_x,
    StructuredBuffer<int64_t> grad_mean2d_y,
    StructuredBuffer<int64_t> grad_cov2d_a,
    StructuredBuffer<int64_t> grad_cov2d_b,
    StructuredBuffer<int64_t> grad_cov2d_c,
    // 3D parameter gradients (this kernel is the sole writer per gaussian).
    RWStructuredBuffer<int64_t> grad_mean_x,
    RWStructuredBuffer<int64_t> grad_mean_y,
    RWStructuredBuffer<int64_t> grad_mean_z,
    RWStructuredBuffer<int64_t> grad_log_scale_x,
    RWStructuredBuffer<int64_t> grad_log_scale_y,
    RWStructuredBuffer<int64_t> grad_log_scale_z,
    RWStructuredBuffer<int64_t> grad_rot_w,
    RWStructuredBuffer<int64_t> grad_rot_x,
    RWStructuredBuffer<int64_t> grad_rot_y,
    RWStructuredBuffer<int64_t> grad_rot_z,
    uniform uint max_splats)
{
    uint gid = dtid.x;
    if (gid >= max_splats) return;
    // Culled tail / dead slots: leave the (cleared) 3D grads at zero, exactly
    // as the old fused kernel did by never contributing. Matches adam.slang.
    if (gid >= num_gaussians_counter[0]) return;

    // Seed = accumulated 2D gradients. Depth/det seeds are 0 (the blend
    // backward produces no gradient w.r.t. depth_vs or cov2d_det).
    float2 g_mean2d = float2(
        fromFixed(grad_mean2d_x[gid]), fromFixed(grad_mean2d_y[gid]));
    float3 g_cov2d = float3(
        fromFixed(grad_cov2d_a[gid]), fromFixed(grad_cov2d_b[gid]),
        fromFixed(grad_cov2d_c[gid]));

    Gaussian3D g = gaussians[gid];

    var dp_mean      = diffPair(g.mean);
    var dp_log_scale = diffPair(g.log_scale);
    var dp_rot_quat  = diffPair(g.rot_quat);
    bwd_diff(projectGaussian3DTo2D)(
        dp_mean, dp_log_scale, dp_rot_quat,
        camera[0].view_matrix, camera[0].proj_matrix, camera[0].image_size,
        g_mean2d, g_cov2d, 0.0, 0.0);

    // Sole writer per gaussian -> direct store. Regularization / adam run in
    // later dispatches and InterlockedAdd on top, so store-then-add is safe.
    grad_mean_x[gid] = toFixed(dp_mean.d.x);
    grad_mean_y[gid] = toFixed(dp_mean.d.y);
    grad_mean_z[gid] = toFixed(dp_mean.d.z);
    grad_log_scale_x[gid] = toFixed(dp_log_scale.d.x);
    grad_log_scale_y[gid] = toFixed(dp_log_scale.d.y);
    grad_log_scale_z[gid] = toFixed(dp_log_scale.d.z);
    grad_rot_w[gid] = toFixed(dp_rot_quat.d.x);
    grad_rot_x[gid] = toFixed(dp_rot_quat.d.y);
    grad_rot_y[gid] = toFixed(dp_rot_quat.d.z);
    grad_rot_z[gid] = toFixed(dp_rot_quat.d.w);
}
