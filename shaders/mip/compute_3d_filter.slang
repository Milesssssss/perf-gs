// Mip-Splatting 3D smoothing filter (Yu et al., CVPR 2024).
//
// Per-gaussian world-space low-pass size, computed from the maximal sampling
// rate across all training views (mirrors mip-splatting's
// gaussian_model.compute_3D_filter, run under no_grad periodically / after
// densification):
//
//   filter_3D[g] = min_over_valid_views(z_depth) * sqrt(0.2) / max_focal
//
// A view is "valid" for gaussian g if g is in front of the camera (z > 0.2)
// and projects within a 1.15x-expanded image rectangle (tangent-space margin,
// matching the reference). `max_focal` (max focal_x over all cameras) and the
// sqrt(0.2) constant are folded on the CPU into `filter_coef = sqrt(0.2)/max_focal`
// and passed as a uniform, so the shader only needs the per-view min depth.
//
// The result is consumed by projectGaussian3DTo2D when --mip-splatting is on:
//   scale_eff = sqrt(exp(log_scale)^2 + filter_3D^2)   (isotropic, all 3 axes)
//   opacity  *= sqrt(prod(scale^2) / prod(scale^2 + filter_3D^2))
// filter_3D itself is a CONSTANT w.r.t. autodiff (recomputed, never a grad target).

#include "../common/types.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void computeMip3DFilter(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> cameras,
    RWStructuredBuffer<float> filter_3D,     // [max_splats], output
    uniform uint num_cameras,
    uniform float filter_coef,               // sqrt(0.2) / max_focal
    uniform uint max_splats)
{
    uint gid = dtid.x;
    if (gid >= max_splats) return;

    // Default (culled / unseen slots): no smoothing.
    filter_3D[gid] = 0.0;
    if (gid >= num_gaussians_counter[0]) return;

    float3 mean = gaussians[gid].mean;

    float min_z = 1e30;
    bool any_valid = false;

    //TODO: Im wondering if we can randomly sample a subset of cameras each time, rather than all of them. How big of a differece would it make? Ablate it
    for (uint c = 0; c < num_cameras; c++)
    {
        CameraData cam = cameras[c];

        // World -> view (same access convention as project.slang / bin_count)
        float3 vs = mul(cam.view_matrix, float4(mean, 1.0)).xyz;
        float z = vs.z;
        if (z <= 0.2)
            continue;

        // Exact projection to pixels (view -> clip -> ndc -> pixels)
        float4 clip = mul(cam.proj_matrix, float4(vs, 1.0));
        float inv_w = 1.0 / clip.w;
        float2 ndc = clip.xy * inv_w;
        float2 px = (ndc * 0.5 + 0.5) * cam.image_size;

        // In-screen with the reference's 1.15x tangent-space margin
        float W = cam.image_size.x;
        float H = cam.image_size.y;
        bool in_screen =
            px.x >= -0.15 * W && px.x <= 1.15 * W &&
            px.y >= -0.15 * H && px.y <= 1.15 * H;
        if (!in_screen)
            continue;

        min_z = min(min_z, z);
        any_valid = true;
    }

    if (any_valid)
        filter_3D[gid] = min_z * filter_coef;
    // a gaussian seen by no view needs no low-pass (and is a prune candidate anyway)
}
