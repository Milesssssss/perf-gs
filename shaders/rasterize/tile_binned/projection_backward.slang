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

#include "generated/execution_contract.slang"
static const uint CLUSTER_SIZE = CONTRACT_CLUSTER_SIZE;

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
    StructuredBuffer<int64_t> grad_comp,   // Mip: dL/d(opacity_compensation)
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
    // Mip-Splatting: per-gaussian 3D filter size + the 2D dilation, so the
    // projection backward reconstructs the SAME cov2d the forward (bin_count)
    // built. filter_3D gated by mip_enabled (buffer is garbage when off).
    StructuredBuffer<float> filter_3D,
    uniform uint mip_enabled,
    uniform float kernel_size,
    // Cluster-cull mask: seeded all-ones when disabled.
    StructuredBuffer<uint> cluster_visible,
    uniform uint max_splats)
{
    uint gid = dtid.x;
    if (gid >= max_splats) return;
    // Culled tail / dead slots: leave the (cleared) 3D grads at zero, exactly
    // as the old fused kernel did by never contributing. Matches adam.slang.
    if (gid >= num_gaussians_counter[0]) return;
    // Culled cluster: the blend backward never scattered 2D seed grads to
    // this slot (its splat has 0 tile pairs), so grad_mean2d_* / grad_cov2d_*
    // are all zero and the projection backward would produce zero output.
    // Skipping saves the AD-generated projection chain and 10 store-sectors
    // of zero writes per culled slot. Same freeze-state contract as Adam:
    // 3D grads stay at zero (Adam zeroed them last iter, no writer touches
    // them until the cluster becomes visible again).
    if (cluster_visible[gid / CLUSTER_SIZE] == 0u) return;

    // Seed = accumulated 2D gradients. Depth/det seeds are 0 (the blend
    // backward produces no gradient w.r.t. depth_vs or cov2d_det).
    float2 grad_mean2d = float2(
        fromFixed(grad_mean2d_x[gid]), fromFixed(grad_mean2d_y[gid]));
    float3 grad_cov2d = float3(
        fromFixed(grad_cov2d_a[gid]), fromFixed(grad_cov2d_b[gid]),
        fromFixed(grad_cov2d_c[gid]));

    Gaussian3D g = gaussians[gid];

    // Route the projection backward through the primitive model: one diffPair
    // over the GaussianModel (mean/log_scale/rot_quat) instead of three. The
    // projection MATH is unchanged (projectInto wraps projectGaussian3DTo2D),
    // so the gradients are bit-identical.
    GaussianModel model;
    model.mean = g.mean;
    model.log_scale = g.log_scale;
    model.rot_quat = g.rot_quat;
    var dp_model = diffPair(model);

    float f3d = (mip_enabled != 0u) ? filter_3D[gid] : 0.0;
    float g_comp = fromFixed(grad_comp[gid]);   // Mip: dL/d(opacity_compensation) seed
    // Seeds match the old call exactly: dL/dmean2d, dL/dcov2d, 0 (depth), 0
    // (det), dL/dcomp. Wrapper is bwd_diff-able (interface methods aren't).
    bwd_diff(projectModel<GaussianModel>)(
        dp_model, camera[0], f3d, kernel_size, mip_enabled,
        grad_mean2d, grad_cov2d, 0.0, 0.0, g_comp);

    // Sole writer per gaussian -> direct store. Regularization / adam run in
    // later dispatches and InterlockedAdd on top, so store-then-add is safe.
    grad_mean_x[gid] = toFixed(dp_model.d.mean.x);
    grad_mean_y[gid] = toFixed(dp_model.d.mean.y);
    grad_mean_z[gid] = toFixed(dp_model.d.mean.z);
    grad_log_scale_x[gid] = toFixed(dp_model.d.log_scale.x);
    grad_log_scale_y[gid] = toFixed(dp_model.d.log_scale.y);
    grad_log_scale_z[gid] = toFixed(dp_model.d.log_scale.z);
    grad_rot_w[gid] = toFixed(dp_model.d.rot_quat.x);
    grad_rot_x[gid] = toFixed(dp_model.d.rot_quat.y);
    grad_rot_y[gid] = toFixed(dp_model.d.rot_quat.z);
    grad_rot_z[gid] = toFixed(dp_model.d.rot_quat.w);
}
