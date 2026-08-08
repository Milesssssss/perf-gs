#include "../common/types.slang"
#include "../primitives/gaussian/project.slang"

// One threadgroup of THREADS_PER_SPLAT threads per gaussian: each lane strides
// over num_cameras/THREADS_PER_SPLAT cameras computing a local min view-space
// depth, then the group reduces to a single min and lane 0 writes filter_3D[gid]
// once. Total camera-tests are unchanged vs. 1-thread-per-splat, but the per-
// splat latency is cut ~THREADS_PER_SPLAT-fold when the grid under-occupies the
// device (few live gaussians).
static const uint THREADS_PER_SPLAT = 32;

groupshared float s_min_z[THREADS_PER_SPLAT];
groupshared uint  s_any[THREADS_PER_SPLAT];   // bool as uint for the OR reduction

[shader("compute")]
[numthreads(THREADS_PER_SPLAT, 1, 1)]
void computeMip3DFilter(
    uint3 group_id: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> cameras,
    RWStructuredBuffer<float> filter_3D,     // [max_splats], output
    uniform uint num_cameras,
    uniform float filter_coef,               // sqrt(0.2) / max_focal
    uniform uint max_splats)
{
    uint gid = group_id.x;            // one group == one gaussian
    uint lane = gtid.x;               // 0 .. THREADS_PER_SPLAT-1
    if (gid >= max_splats) return;

    // Uniform across the group (gid is per-group), so these early-outs never
    // split the group ahead of a barrier.
    if (gid >= num_gaussians_counter[0])
    {
        if (lane == 0)
            filter_3D[gid] = 0.0;     // culled / unseen slot: no smoothing
        return;
    }

    float3 mean = gaussians[gid].mean;

    float min_z = 1e30;
    bool any_valid = false;

    // Strided so consecutive lanes touch consecutive cameras (coalesced reads).
    for (uint c = lane; c < num_cameras; c += THREADS_PER_SPLAT)
    {
        CameraData cam = cameras[c];

        // Shared mean projection (single source of truth in project.slang).
        float3 vs;
        float2 px;
        float clip_w;
        projectMeanToPixel(mean, cam.view_matrix, cam.proj_matrix, cam.image_size, vs, px, clip_w);
        float z = vs.z;
        if (z <= 0.2)
            continue;

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

    // Groupshared tree reduction
    s_min_z[lane] = min_z;
    s_any[lane] = any_valid ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = THREADS_PER_SPLAT / 2; stride > 0; stride >>= 1)
    {
        if (lane < stride)
        {
            s_min_z[lane] = min(s_min_z[lane], s_min_z[lane + stride]);
            s_any[lane]   = s_any[lane] | s_any[lane + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0)
    {
        // a gaussian seen by no view needs no low-pass (and is a prune candidate)
        filter_3D[gid] = (s_any[0] != 0u) ? (s_min_z[0] * filter_coef) : 0.0;
    }
}
