// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Per-gaussian projection backward on the blend backward's tile-summed 2D gradients; exact because
// the projection backward is linear in its seed: sum_k J^T g_k == J^T sum_k g_k.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"

#include "common/execution.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void projectionBackward(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    // Tile-summed 2D seeds; this kernel is their only reader, so it also resets them.
    RWStructuredBuffer<int64_t> grad_mean2d_x,
    RWStructuredBuffer<int64_t> grad_mean2d_y,
    RWStructuredBuffer<int64_t> grad_cov2d_a,
    RWStructuredBuffer<int64_t> grad_cov2d_b,
    RWStructuredBuffer<int64_t> grad_cov2d_c,
    RWStructuredBuffer<int64_t> grad_comp,   // Mip: dL/d(opacity_compensation)
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
    // Mip-Splatting inputs to rebuild bin_count's exact cov2d; filter_3D is garbage unless mip_enabled.
    StructuredBuffer<float> filter_3D,
    uniform uint mip_enabled,
    uniform float kernel_size,
    // Cluster-cull mask: seeded all-ones when disabled.
    StructuredBuffer<uint> cluster_visible,
    uniform uint max_splats)
{
    uint gid = dtid.x;
    if (gid >= max_splats) return;
    // Dead slots leave the cleared 3D grads at zero (matches adam.slang).
    if (gid >= num_gaussians_counter[0]) return;
    // Culled clusters have zero 2D seeds; their 3D grads stay zero (Adam's freeze-state contract).
    if (cluster_visible[gid / CLUSTER_SIZE] == 0u) return;

    int64_t seed_mean_x = grad_mean2d_x[gid], seed_mean_y = grad_mean2d_y[gid];
    int64_t seed_cov_a = grad_cov2d_a[gid], seed_cov_b = grad_cov2d_b[gid];
    int64_t seed_cov_c = grad_cov2d_c[gid], seed_comp = grad_comp[gid];
    // Reset for the next step; untouched slots are already zero, so no capacity-wide clear is needed.
    if ((seed_mean_x | seed_mean_y | seed_cov_a | seed_cov_b | seed_cov_c | seed_comp) != int64_t(0))
    {
        grad_mean2d_x[gid] = int64_t(0); grad_mean2d_y[gid] = int64_t(0);
        grad_cov2d_a[gid] = int64_t(0); grad_cov2d_b[gid] = int64_t(0);
        grad_cov2d_c[gid] = int64_t(0); grad_comp[gid] = int64_t(0);
    }
    float2 grad_mean2d = float2(fromFixed(seed_mean_x), fromFixed(seed_mean_y));
    float3 grad_cov2d = float3(
        fromFixed(seed_cov_a), fromFixed(seed_cov_b), fromFixed(seed_cov_c));

    Gaussian3D g = gaussians[gid];

    GaussianModel model;
    model.mean = g.mean;
    model.log_scale = g.log_scale;
    model.rot_quat = g.rot_quat;
    var dp_model = diffPair(model);

    float f3d = (mip_enabled != 0u) ? filter_3D[gid] : 0.0;
    float g_comp = fromFixed(seed_comp);   // Mip: dL/d(opacity_compensation) seed
    // Depth/det seeds are 0; the projectModel wrapper exists because interface methods aren't bwd_diff-able.
    bwd_diff(projectModel<GaussianModel>)(
        dp_model, camera[0], f3d, kernel_size, mip_enabled,
        grad_mean2d, grad_cov2d, 0.0, 0.0, g_comp);

    // Sole writer, so a plain store; later regularization/Adam dispatches InterlockedAdd on top.
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
