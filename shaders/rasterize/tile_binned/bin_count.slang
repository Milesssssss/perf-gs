// Phase 1 of the uncapped binning pipeline: per-gaussian tile-touch count.
//
// For each live gaussian:
//   1. Project to 2D (same as old bin.slang).
//   2. Near-plane cull, SnugBox, clamp to frame.
//   3. Use AccuTile to count the exact number of tiles the ellipse touches.
//   4. Write projected_2d (so forward/backward can reuse it) and
//      tile_count_per_gaussian[gid].
//
// The caller then prefix-scans tile_count_per_gaussian to get write
// offsets, and dispatches bin_emit to actually write (key, value) pairs.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/snugbox.slang"
#include "../../primitives/gaussian/accutile.slang"

//TODO: I believe this and bin_emit.slang could be merged into a single kernel, mot likely

static const uint TILE_SIZE = 16;
static const uint CLUSTER_SIZE = 128;  // must match cluster_cull.slang

[shader("compute")]
[numthreads(256, 1, 1)]
void binCount(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    RWStructuredBuffer<Projected2DGaussian> projected_2d,
    RWStructuredBuffer<BlendData> blend_data,          // [max_splats], compact blend gather record
    RWStructuredBuffer<uint> tile_count_per_gaussian,  // [max_splats]
    StructuredBuffer<uint> cluster_visible,            // [ceil(max_splats/128)]
    StructuredBuffer<float> filter_3D,                 // [max_splats], mip 3D size (0=off)
    uniform uint mip_enabled,                          // Mip-Splatting on/off
    uniform float kernel_size,                         // 2D dilation (0.3 vanilla / 0.1 mip)
    uniform float z_near,
    uniform uint max_splats)
{
    uint gid = dtid.x;
    if (gid >= max_splats) return;

    // Default: zero tiles (culled). Write now so the scan sees a clean
    // value even if we early-return.
    tile_count_per_gaussian[gid] = 0;

    if (gid >= num_gaussians_counter[0]) return;

    // Cluster-frustum cull: if the entire 128-splat cluster's AABB is
    // outside the view frustum, its members contribute no tile pairs.
    // cluster_visible is seeded to all-ones when cluster-cull is off, so
    // this check is a no-op unless the cluster-cull kernels wrote it.
    // Wave-uniform read: Morton keeps 128-splat clusters aligned to two
    // consecutive 256-thread workgroups, so a workgroup sees at most two
    // cluster_visible values and every warp reads a single uniform value —
    // zero branch divergence cost.
    //
    // Placed BEFORE projection + blend pack so culled clusters skip the
    // ~15 ALU ops of projectGaussian3DTo2D AND the two 60 B sector writes
    // (projected_2d + blend_data). bin_emit also reads cluster_visible and
    // skips those slots, so projected_2d[gid] / blend_data[gid] for culled
    // slots are never read by any downstream kernel — stale contents are
    // fine.
    if (cluster_visible[gid / CLUSTER_SIZE] == 0u) return;

    Gaussian3D g = gaussians[gid];

    float2 mean_2d;
    float3 cov2d;
    float depth_vs;
    float cov2d_det;

    // Mip-Splatting: 3D smoothing size folded into the projected covariance,
    // plus the coef3d*coef2d opacity compensation (1.0 when off), computed
    // DIFFERENTIABLY inside project so the backward can route dL/dcomp to
    // scale/rot/mean. f3d==0 and kernel_size==0.3 reproduce vanilla exactly.
    float f3d = (mip_enabled != 0u) ? filter_3D[gid] : 0.0;
    float comp;

    projectGaussian3DTo2D(
        g.mean, g.log_scale, g.rot_quat,
        camera[0].view_matrix, camera[0].proj_matrix, camera[0].image_size,
        f3d, kernel_size, mip_enabled,
        mean_2d, cov2d, depth_vs, cov2d_det, comp);

    Projected2DGaussian proj;
    proj.mean = mean_2d;
    proj.cov2d = cov2d;
    proj.depth_vs = depth_vs;
    proj.opacity_compensation = comp;
    projected_2d[gid] = proj;

    // Pack the compact blend record here: all four fields are already in hand
    // (mean_2d/cov2d just computed, opacity/sh0 from g), so this is a free
    // write — no extra global loads. The blend backward gathers this 36 B
    // record instead of the 80 B (gaussians + projected_2d) pair, cutting L2
    // pollution on its random-order gather. Written alongside projected_2d so
    // it covers every projected gaussian the blend could reference.
    BlendData bd;
    bd.mean = mean_2d;
    bd.cov2d = cov2d;
    bd.opacity = g.opacity;
    bd.sh0 = g.sh0;
    bd.opacity_compensation = comp;
    blend_data[gid] = bd;

    if (depth_vs < z_near) return;

    float2 frame_dim = camera[0].image_size;

    float2 bbox_min, bbox_max;
    if (!snugBox(mean_2d, cov2d, g.opacity, bbox_min, bbox_max))
        return;

    bbox_min = max(bbox_min, float2(0.0, 0.0));
    bbox_max = min(bbox_max, frame_dim);
    if (bbox_min.x >= bbox_max.x || bbox_min.y >= bbox_max.y)
        return;

    uint2 num_tiles = ((uint2)frame_dim + TILE_SIZE - 1) / TILE_SIZE;
    uint2 tile_min = uint2(bbox_min) / TILE_SIZE;
    uint2 tile_max = uint2(bbox_max) / TILE_SIZE;
    tile_max = min(tile_max, num_tiles - 1);

    // AccuTile: count only the tiles the ellipse actually crosses, not every
    // tile in the SnugBox rectangle. Iterate the shorter axis, summing the
    // per-strip tile range. bin_emit re-runs the IDENTICAL walk (same stored
    // projection) so its write count matches this exactly — see accutile.slang.
    EllipseParams e;
    if (!computeEllipseParams(cov2d, g.opacity, e))
        return;
    float x_d_args, y_d_args;
    computeExtremumOffsets(e, x_d_args, y_d_args);

    uint count = 0u;
    if (accuTileIterateRows(tile_min, tile_max))
    {
        for (uint ty = tile_min.y; ty <= tile_max.y; ++ty)
        {
            uint tx_lo, tx_hi;
            if (accuTileRowTiles(e, mean_2d, bbox_min, bbox_max, y_d_args,
                                 ty, tile_min.x, tile_max.x, tx_lo, tx_hi))
                count += tx_hi - tx_lo + 1u;
        }
    }
    else
    {
        for (uint tx = tile_min.x; tx <= tile_max.x; ++tx)
        {
            uint ty_lo, ty_hi;
            if (accuTileColTiles(e, mean_2d, bbox_min, bbox_max, x_d_args,
                                 tx, tile_min.y, tile_max.y, ty_lo, ty_hi))
                count += ty_hi - ty_lo + 1u;
        }
    }
    tile_count_per_gaussian[gid] = count;
}
