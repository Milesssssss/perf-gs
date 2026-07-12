// Cluster-Cull-Compact (Laine et al 2011), adapted for 3DGS training.
//
// Runs BEFORE bin_count each frame. Two entry points:
//
//   1. buildClusterAABB
//      One workgroup per cluster (CLUSTER_SIZE = 128 threads / cluster).
//      Each thread bounds its own gaussian by (mean +/- k_sigma * |R| * scale)
//      — the 3-sigma AABB of the rotated ellipsoid — then a groupshared
//      min/max reduction produces the cluster AABB. Threads whose slot >=
//      num_gaussians (tail of the last cluster) contribute the neutral
//      element (+inf min, -inf max) so they don't corrupt the reduce.
//
//   2. cullClustersFrustum
//      One thread per cluster. Derives 6 frustum planes from
//      mvp = proj * view via Gribb/Hartmann. Runs a conservative
//      AABB-vs-6-planes p-vertex test: for each plane, pick the AABB
//      corner most along the plane normal; if it's still on the negative
//      side of the plane the whole box is outside -> mark not visible.
//      Never a false negative (culls only what is guaranteed offscreen).
//
// Slot indexing is NEVER modified: cluster_visible[c] is a lookup used
// by bin_count's early-out, so backward's grad scatters and Adam's
// per-slot updates continue to operate on the canonical layout. No
// compaction, no per-splat remap table.

#include "../common/types.slang"
#include "../common/math.slang"

static const uint CLUSTER_SIZE = 128;
static const float K_SIGMA = 3.0;  // 3-sigma ellipsoid extent

// Per-cluster reduction scratch (min/max across 128 gaussian AABBs).
groupshared float3 g_bmin[CLUSTER_SIZE];
groupshared float3 g_bmax[CLUSTER_SIZE];

// Compute the world-space AABB of a single gaussian's k-sigma ellipsoid:
//   |R * diag(scale) * x| for |x| <= k  is bounded axis-wise by
//   row-sum-of-absolute-values of (R * diag(scale)) * k. This is exact-
//   conservative (a tight enclosing AABB of the ellipsoid).
void splatAABB(Gaussian3D g, float k, out float3 bmin, out float3 bmax)
{
    float3x3 R = matrixFromQuaternion(g.rot_quat);
    float3 s = exp(g.log_scale);
    // extent along axis i = sum_j |R[i][j]| * s[j] * k
    float3 col0 = float3(R[0][0], R[1][0], R[2][0]) * s.x;
    float3 col1 = float3(R[0][1], R[1][1], R[2][1]) * s.y;
    float3 col2 = float3(R[0][2], R[1][2], R[2][2]) * s.z;
    float3 extent = (abs(col0) + abs(col1) + abs(col2)) * k;
    bmin = g.mean - extent;
    bmax = g.mean + extent;
}

[shader("compute")]
[numthreads(CLUSTER_SIZE, 1, 1)] 
void buildClusterAABB(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint3 lid: SV_GroupThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint>       num_gaussians_counter,
    RWStructuredBuffer<float4>   cluster_aabb_min,
    RWStructuredBuffer<float4>   cluster_aabb_max,
    uniform uint max_splats,
    uniform uint num_clusters)
{
    uint c = gid.x;
    if (c >= num_clusters) return;
    uint t = lid.x;
    uint slot = c * CLUSTER_SIZE + t;

    uint n = num_gaussians_counter[0];
    float3 bmin, bmax;
    if (slot < n && slot < max_splats) {
        splatAABB(gaussians[slot], K_SIGMA, bmin, bmax);
    } else {
        // Neutral element for min/max reduce — never contributes to the
        // cluster AABB, so tail-of-last-cluster threads are inert.
        bmin = float3( 1e30,  1e30,  1e30);
        bmax = float3(-1e30, -1e30, -1e30);
    }
    g_bmin[t] = bmin;
    g_bmax[t] = bmax;
    GroupMemoryBarrierWithGroupSync();

    // Tree reduction across the 128-thread workgroup. Halving stride each
    // step: 128 -> 64 -> 32 -> ... -> 1. Thread 0 holds the final AABB.
    for (uint stride = CLUSTER_SIZE / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            g_bmin[t] = min(g_bmin[t], g_bmin[t + stride]);
            g_bmax[t] = max(g_bmax[t], g_bmax[t + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (t == 0) {
        cluster_aabb_min[c] = float4(g_bmin[0], 0.0);
        cluster_aabb_max[c] = float4(g_bmax[0], 0.0);
    }
}

// Extract the 6 frustum planes from mvp = proj * view in Gribb/Hartmann
// form. planes[i].xyz is the (unnormalized) plane normal pointing INTO
// the frustum, planes[i].w is the offset; a point p is inside plane i iff
// dot(planes[i].xyz, p) + planes[i].w >= 0.
//
// Row indexing convention: Slang float4x4 default is row-major; MVP row i
// is float4(mvp[i][0], mvp[i][1], mvp[i][2], mvp[i][3]). We store planes
// packed as row3 +/- row_k.
void extractFrustumPlanes(float4x4 mvp, out float4 planes[6])
{
    float4 r0 = float4(mvp[0][0], mvp[0][1], mvp[0][2], mvp[0][3]);
    float4 r1 = float4(mvp[1][0], mvp[1][1], mvp[1][2], mvp[1][3]);
    float4 r2 = float4(mvp[2][0], mvp[2][1], mvp[2][2], mvp[2][3]);
    float4 r3 = float4(mvp[3][0], mvp[3][1], mvp[3][2], mvp[3][3]);
    planes[0] = r3 + r0;   // left    (x + w >= 0)
    planes[1] = r3 - r0;   // right   (w - x >= 0)
    planes[2] = r3 + r1;   // bottom  (y + w >= 0)
    planes[3] = r3 - r1;   // top     (w - y >= 0)
    // Vulkan clip-space z is [0, w]. near plane: z >= 0 => r2 . p >= 0.
    planes[4] = r2;
    // far plane: z <= w  => (r3 - r2) . p >= 0.
    planes[5] = r3 - r2;
}

// Conservative AABB-vs-frustum test. Returns true if AABB is at least
// partially inside all 6 half-spaces (i.e. might be visible). Never
// returns false for a visible box.
bool aabbInFrustum(float3 bmin, float3 bmax, float4 planes[6])
{
    for (uint i = 0; i < 6; ++i) {
        // "Positive-vertex" (aka p-vertex): the AABB corner farthest along
        // the plane normal. If even THAT corner is behind the plane, every
        // corner is behind -> AABB fully outside -> cull.
        float3 pv;
        pv.x = (planes[i].x >= 0.0) ? bmax.x : bmin.x;
        pv.y = (planes[i].y >= 0.0) ? bmax.y : bmin.y;
        pv.z = (planes[i].z >= 0.0) ? bmax.z : bmin.z;
        if (dot(planes[i].xyz, pv) + planes[i].w < 0.0) return false;
    }
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cullClustersFrustum(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<float4>     cluster_aabb_min,
    StructuredBuffer<float4>     cluster_aabb_max,
    StructuredBuffer<CameraData> camera,
    RWStructuredBuffer<uint>     cluster_visible,
    uniform uint num_clusters)
{
    uint c = dtid.x;
    if (c >= num_clusters) return;

    // mvp = proj * view. Slang uses row-major matrices and mul(A, B)
    // returns A*B, so this composes correctly for world -> clip.
    float4x4 mvp = mul(camera[0].proj_matrix, camera[0].view_matrix);
    float4 planes[6];
    extractFrustumPlanes(mvp, planes);

    float3 bmin = cluster_aabb_min[c].xyz;
    float3 bmax = cluster_aabb_max[c].xyz;
    cluster_visible[c] = aabbInFrustum(bmin, bmax, planes) ? 1u : 0u;
}
