// Unnamed technique, shader fillBuffer
/*$(ShaderResources)*/

static float2 cameraPos = /*$(Variable:cameraPos)*/;
static float2 scale = /*$(Variable:scale)*/;
static float2 frame_dim = /*$(Variable:frameDim)*/;
static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;
static float4x4 world_to_view = transpose(/*$(Variable:ViewMtx)*/);
static float4x4 view_to_screen = transpose(/*$(Variable:ProjMtx)*/);

#define MAX_GAUSSIANS_PER_TILE 1024
#define TILE_SIZE 16
#define SIGMA_CUTOFF 3.0

#include "utils.slang"

struct OrientedBoundingBox
{
    float2 center;
    float2 axis_u;      // First principal axis (normalized)
    float2 axis_v;      // Second principal axis (normalized)
    float extent_u;     // Half-extent along axis_u (3-sigma radius)
    float extent_v;     // Half-extent along axis_v (3-sigma radius)
};

void computeEigen2x2(float3 cov2d, out float2 eigenvec1, out float2 eigenvec2, out float lambda1, out float lambda2)
{
    float a = cov2d.x; // cov_00
    float b = cov2d.y; // cov_01
    float c = cov2d.z; // cov_11

    float trace = a + c;
    float det = a * c - b * b;
    float discriminant = max(0.0, trace * trace * 0.25 - det);
    float sqrt_disc = sqrt(discriminant);

    lambda1 = trace * 0.5 + sqrt_disc;
    lambda2 = trace * 0.5 - sqrt_disc;

    // Compute eigenvector for lambda1
    if (abs(b) > 1e-8)
    {
        eigenvec1 = normalize(float2(lambda1 - c, b));
    }
    else if (a >= c)
    {
        eigenvec1 = float2(1, 0);
    }
    else
    {
        eigenvec1 = float2(0, 1);
    }

    // Second eigenvector is perpendicular
    eigenvec2 = float2(-eigenvec1.y, eigenvec1.x);
}

OrientedBoundingBox computeOBB(float2 mean_2d, float3 cov2d)
{
    OrientedBoundingBox obb;
    obb.center = mean_2d;

    float lambda1, lambda2;
    computeEigen2x2(cov2d, obb.axis_u, obb.axis_v, lambda1, lambda2);

    // 3-sigma extents
    obb.extent_u = SIGMA_CUTOFF * sqrt(max(lambda1, 1e-8));
    obb.extent_v = SIGMA_CUTOFF * sqrt(max(lambda2, 1e-8));

    return obb;
}

void getOBBScreenBounds(OrientedBoundingBox obb, out float2 bbox_min, out float2 bbox_max)
{
    float2 corner_offsets[4];
    corner_offsets[0] = obb.axis_u * obb.extent_u + obb.axis_v * obb.extent_v;
    corner_offsets[1] = obb.axis_u * obb.extent_u - obb.axis_v * obb.extent_v;
    corner_offsets[2] = -obb.axis_u * obb.extent_u + obb.axis_v * obb.extent_v;
    corner_offsets[3] = -obb.axis_u * obb.extent_u - obb.axis_v * obb.extent_v;

    bbox_min = obb.center;
    bbox_max = obb.center;

    [unroll]
    for (int i = 0; i < 4; i++)
    {
        float2 corner = obb.center + corner_offsets[i];
        bbox_min = min(bbox_min, corner);
        bbox_max = max(bbox_max, corner);
    }
}

bool pointInOBB(float2 point, OrientedBoundingBox obb)
{
    float2 d = point - obb.center;
    float proj_u = abs(dot(d, obb.axis_u));
    float proj_v = abs(dot(d, obb.axis_v));
    return proj_u <= obb.extent_u && proj_v <= obb.extent_v;
}

// Check if a line segment intersects the OBB
bool segmentIntersectsOBB(float2 p0, float2 p1, OrientedBoundingBox obb)
{
    float2 local_p0 = float2(dot(p0 - obb.center, obb.axis_u), dot(p0 - obb.center, obb.axis_v));
    float2 local_p1 = float2(dot(p1 - obb.center, obb.axis_u), dot(p1 - obb.center, obb.axis_v));

    float2 box_min = float2(-obb.extent_u, -obb.extent_v);
    float2 box_max = float2(obb.extent_u, obb.extent_v);

    // Liang-Barsky line clipping
    float2 d = local_p1 - local_p0;
    float t_min = 0.0;
    float t_max = 1.0;

    [unroll]
    for (int i = 0; i < 2; i++)
    {
        float p_i = (i == 0) ? local_p0.x : local_p0.y;
        float d_i = (i == 0) ? d.x : d.y;
        float min_i = (i == 0) ? box_min.x : box_min.y;
        float max_i = (i == 0) ? box_max.x : box_max.y;

        if (abs(d_i) < 1e-8)
        {
            if (p_i < min_i || p_i > max_i)
                return false;
        }
        else
        {
            float t1 = (min_i - p_i) / d_i;
            float t2 = (max_i - p_i) / d_i;
            if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
            t_min = max(t_min, t1);
            t_max = min(t_max, t2);
            if (t_min > t_max)
                return false;
        }
    }
    return true;
}

bool obbOverlapsTile(OrientedBoundingBox obb, float2 tile_min, float2 tile_max)
{
    if (obb.center.x >= tile_min.x && obb.center.x <= tile_max.x &&
        obb.center.y >= tile_min.y && obb.center.y <= tile_max.y)
        return true;

    float2 corners[4];
    corners[0] = tile_min;
    corners[1] = float2(tile_max.x, tile_min.y);
    corners[2] = float2(tile_min.x, tile_max.y);
    corners[3] = tile_max;

    [unroll]
    for (int i = 0; i < 4; i++)
    {
        if (pointInOBB(corners[i], obb))
            return true;
    }

    if (segmentIntersectsOBB(corners[0], corners[1], obb)) return true;
    if (segmentIntersectsOBB(corners[1], corners[3], obb)) return true;
    if (segmentIntersectsOBB(corners[3], corners[2], obb)) return true;
    if (segmentIntersectsOBB(corners[2], corners[0], obb)) return true;

    float2 obb_corners[4];
    obb_corners[0] = obb.center + obb.axis_u * obb.extent_u + obb.axis_v * obb.extent_v;
    obb_corners[1] = obb.center + obb.axis_u * obb.extent_u - obb.axis_v * obb.extent_v;
    obb_corners[2] = obb.center - obb.axis_u * obb.extent_u + obb.axis_v * obb.extent_v;
    obb_corners[3] = obb.center - obb.axis_u * obb.extent_u - obb.axis_v * obb.extent_v;

    [unroll]
    for (int j = 0; j < 4; j++)
    {
        if (obb_corners[j].x >= tile_min.x && obb_corners[j].x <= tile_max.x &&
            obb_corners[j].y >= tile_min.y && obb_corners[j].y <= tile_max.y)
            return true;
    }

    return false;
}

/*$(_compute:binGaussiansDebug)*/ (uint3 DTid: SV_DispatchThreadID)
{
    uint gid = DTid.x;
    if (gid >= num_gaussians_counter[0]) return;

    Struct_Gaussian3D gaussian = input_gaussians[gid];

    const uint2 numTiles = ((uint2)frame_dim + TILE_SIZE - 1) / TILE_SIZE;

    Struct_Projected2DGaussian proj_2d;

    //world_to_view[1] = -world_to_view[1];

    projectGaussian3DTo2D_diff(
        gaussian.mean,
        gaussian.log_scale,
        gaussian.rotation_quat,
        world_to_view,
        view_to_screen,
        frame_dim,
        proj_2d.mean,
        proj_2d.cov2d,
        proj_2d.depth_vs
    );
    //proj_2d.depth_vs = -proj_2d.depth_vs;
    projected_gaussians_2d[gid] = proj_2d;

    // Compute OBB from covariance
    OrientedBoundingBox obb = computeOBB(proj_2d.mean, proj_2d.cov2d);

    // Get coarse AABB of the OBB for tile range
    float2 screen_min, screen_max;
    getOBBScreenBounds(obb, screen_min, screen_max);

    screen_min = max(screen_min, float2(0, 0));
    screen_max = min(screen_max, frame_dim);

    if (screen_min.x >= screen_max.x || screen_min.y >= screen_max.y)
        return;

    uint2 tile_min = uint2(screen_min) / TILE_SIZE;
    uint2 tile_max = uint2(screen_max) / TILE_SIZE;
    tile_max = min(tile_max, numTiles - 1);

    // Bin into tiles with proper OBB overlap test
    for (uint ty = tile_min.y; ty <= tile_max.y; ty++)
    {
        for (uint tx = tile_min.x; tx <= tile_max.x; tx++)
        {
            float2 tile_screen_min = float2(tx, ty) * TILE_SIZE;
            float2 tile_screen_max = tile_screen_min + float2(TILE_SIZE, TILE_SIZE);

            if (!obbOverlapsTile(obb, tile_screen_min, tile_screen_max))
                continue;

            uint tileIdx = ty * numTiles.x + tx;

            uint slot;
            InterlockedAdd(screen_tiles_count[tileIdx], 1, slot);

            if (slot < MAX_GAUSSIANS_PER_TILE)
            {
                screen_tiles_indices[tileIdx * MAX_GAUSSIANS_PER_TILE + slot] = gid;
            }
        }
    }
}