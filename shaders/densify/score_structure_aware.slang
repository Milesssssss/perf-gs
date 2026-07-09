// Structure-aware densification: per-Gaussian frequency-violation scoring.
//
// Implements the eta metric of Lyu et al. "Faster 3DGS Convergence via
// Structure-Aware Densification" (arXiv 2604.28016), Sections 3.3-3.4.
//
// One thread per Gaussian. For the current view we:
//   1. Project the Gaussian's 3 principal axis vectors to screen space,
//      v_k = J W R S_k  (Eq. 1). The columns of A = J W R diag(scale) are v_k.
//   2. Sample the aggregated structure tensor S̄ at the projected footprint,
//      jittered within the 2D covariance to reduce edge-alignment sensitivity,
//      and take its principal eigenvalue lambda1 → local min wavelength
//      Lambda_min = 1 / (sqrt(lambda1) + eps).
//   3. eta_k = ||v_k|| / Lambda_min  (Eq. 6). eta_k > 1 ⟹ this axis under-
//      resolves the local texture.
//   4. Accumulate per-axis multiview stats: n_high (eta>1), n_low (eta<0.1),
//      eta_max, and a shared n_total (Eq. of Section 3.4). These are read back
//      / consumed by the CPU decision + anisotropic split.
//
// Runs once per scoring view; buffers persist across views and are cleared by
// clearStructureScores at the window start. NOT the hot training path.

#include "../common/types.slang"
#include "../common/math.slang"

static const float ETA_HIGH = 1.0;    // eta > this ⟹ under-resolved (split vote)
static const float ETA_LOW  = 0.1;    // eta < this ⟹ over-resolved (prune vote)
static const float EPS = 1e-8;
// Screen-space low-pass dilation the FORWARD renderer adds to the 2D covariance
// diagonal (project.slang: cov2d.a/c += 0.3). It floors every splat to ~sqrt(0.3)
// px so it can't fall between pixel centers and alias. Our per-axis extents are
// std-dev-like (px), the dilation is a variance (px^2), so the on-screen extent
// is sqrt(len^2 + COV_DILATION). Without this the eta kernel UNDER-measures small
// splats' footprints vs. what actually gets rasterized (fidelity gap).
static const float COV_DILATION = 0.3;

// Wang-hash uniform[0,1), matching split_and_clone.slang's jitter RNG so the
// sampling is deterministic per (gaussian, view).
uint _wang_hash(uint x)
{
    x = (x ^ 61u) ^ (x >> 16u);
    x *= 9u;
    x = x ^ (x >> 4u);
    x *= 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}
float _rand01(uint seed) { return float(_wang_hash(seed)) / 4294967295.0; }

// Top eigenvalue of the packed symmetric 2x2 tensor (Sxx, Sxy, Syy).
float _principal_eigenvalue(float3 s)
{
    float a = s.x, b = s.y, c = s.z;
    float mid = 0.5 * (a + c);
    float disc = sqrt(max((0.5 * (a - c)) * (0.5 * (a - c)) + b * b, 0.0));
    return mid + disc;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void scoreStructureAware(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    StructuredBuffer<CameraData> camera,
    // Aggregated structure tensor field S̄ for THIS view, packed (Sxx,Sxy,Syy).
    Texture2D<float4> structure_tensor,
    // Per-gaussian on-screen tile count from THIS view's forward pass
    // (bin_count.slang). >0 iff the gaussian passed near-plane + frustum +
    // snugbox culling and has a nonzero footprint — this is our
    // VISIBILITY GATE, matching SADGS's is_active_vis (only accumulate eta for
    // gaussians actually visible/contributing in the rendered frame). Without
    // it, ~96% of raw-init gaussians trigger a split and densification explodes.
    StructuredBuffer<uint> tile_count_per_gaussian,
    // SCALAR multiview classification counts (SADGS classifies on the
    // max-over-axes eta, not per-axis): n_high = views with eta_max_scalar>1,
    // n_low = views with eta_max_scalar<=0.1.
    RWStructuredBuffer<uint> eta_n_high, RWStructuredBuffer<uint> eta_n_low,
    // Per-axis max eta as fixed-point — kept per-axis because the anisotropic
    // split FACTOR n_k = ceil(sqrt(eta_max_k)) needs each axis independently.
    RWStructuredBuffer<uint> eta_max_x, RWStructuredBuffer<uint> eta_max_y, RWStructuredBuffer<uint> eta_max_z,
    // Count of views in which this gaussian was on-screen / valid.
    RWStructuredBuffer<uint> eta_n_total,
    uniform float opacity_threshold,  // SADGS is_active_vis: sigmoid(opacity) > this (0.05)
    uniform uint pass_idx,       // view index, for jitter decorrelation across views
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    if (i >= num_gaussians_counter[0]) return;

    // VISIBILITY GATE (SADGS is_active_vis): skip gaussians that did not
    // rasterize this view (culled / zero footprint) or are near-transparent.
    // tile_count is this view's forward-pass visibility; opacity is the
    // gsplat sigmoid(opacity) > 0.05 gate.
    if (tile_count_per_gaussian[i] == 0u) return;

    Gaussian3D g = gaussians[i];
    if (1.0 / (1.0 + exp(-g.opacity)) <= opacity_threshold) return;

    CameraData cam = camera[0];
    float2 frame_dim = cam.image_size;

    // --- Project mean to view + pixel space (mirrors project.slang). ---
    float3 mean_vs = mul(cam.view_matrix, float4(g.mean, 1.0)).xyz;
    if (mean_vs.z <= 0.0) return;   // behind camera; no observation this view

    float4 mean_clip = mul(cam.proj_matrix, float4(mean_vs, 1.0));
    if (mean_clip.w <= 0.0) return;
    float2 ndc = mean_clip.xy / mean_clip.w;
    float2 mean_px = (ndc * 0.5 + 0.5) * frame_dim;

    // --- Build the projection Jacobian J (same construction as project.slang). ---
    float fx = cam.proj_matrix[0][0] * frame_dim.x * 0.5;
    float fy = cam.proj_matrix[1][1] * frame_dim.y * 0.5;
    float tan_fovx = 0.5 * frame_dim.x / abs(fx);
    float tan_fovy = 0.5 * frame_dim.y / abs(fy);
    float limx = 1.3 * tan_fovx;
    float limy = 1.3 * tan_fovy;
    float txtz = clamp(mean_vs.x / mean_vs.z, -limx, limx);
    float tytz = clamp(mean_vs.y / mean_vs.z, -limy, limy);
    float3x3 J = float3x3(
        fx / mean_vs.z, 0.0, -fx * txtz / mean_vs.z,
        0.0, fy / mean_vs.z, -fy * tytz / mean_vs.z,
        0.0, 0.0, 0.0
    );

    // A = J W R diag(scale). Its columns are the projected axis vectors v_k.
    float3x3 W = float3x3(cam.view_matrix);
    float3x3 R = matrixFromQuaternion(g.rot_quat);
    float3 scale = exp(g.log_scale);
    float3x3 RS = float3x3(
        R[0][0] * scale.x, R[0][1] * scale.y, R[0][2] * scale.z,
        R[1][0] * scale.x, R[1][1] * scale.y, R[1][2] * scale.z,
        R[2][0] * scale.x, R[2][1] * scale.y, R[2][2] * scale.z
    );
    float3x3 A = mul(J, mul(W, RS));
    // Projected 2D extent of each axis (the x,y rows of A; row 2 of J is 0),
    // dilated to match the renderer's rasterized footprint (see COV_DILATION):
    // effective_len = sqrt(len^2 + 0.3), floored just like project.slang.
    float len_vx = sqrt(A[0][0] * A[0][0] + A[1][0] * A[1][0] + COV_DILATION);
    float len_vy = sqrt(A[0][1] * A[0][1] + A[1][1] * A[1][1] + COV_DILATION);
    float len_vz = sqrt(A[0][2] * A[0][2] + A[1][2] * A[1][2] + COV_DILATION);

    // --- Sample S̄ at a jittered footprint point, get local min wavelength. ---
    // Jitter within ~1 projected-axis extent of the center (Section 3.6's
    // stochastic jitter sampling), decorrelated per view via pass_idx.
    uint seed = i * 2654435761u + pass_idx * 40503u;
    float jx = (_rand01(seed) - 0.5);
    float jy = (_rand01(seed + 1u) - 0.5);
    float2 sample_px = mean_px + float2(jx * len_vx, jy * len_vy);

    uint2 dim; structure_tensor.GetDimensions(dim.x, dim.y);
    int2 tc = int2(clamp(sample_px, float2(0.0, 0.0), float2(dim) - 1.0));
    float3 sbar = structure_tensor[tc].xyz;

    float lambda1 = _principal_eigenvalue(sbar);
    float inv_lambda_min = sqrt(max(lambda1, 0.0)) + EPS;  // 1 / Lambda_min

    float eta_x = len_vx * inv_lambda_min;
    float eta_y = len_vy * inv_lambda_min;
    float eta_z = len_vz * inv_lambda_min;
    // SADGS classifies this view on the SCALAR max-over-axes eta.
    float eta_max_scalar = max(eta_x, max(eta_y, eta_z));

    uint dummy;
    InterlockedAdd(eta_n_total[i], 1u, dummy);

    if (eta_max_scalar > ETA_HIGH) InterlockedAdd(eta_n_high[i], 1u, dummy);
    else if (eta_max_scalar <= ETA_LOW) InterlockedAdd(eta_n_low[i], 1u, dummy);

    // Track max eta PER AXIS as fixed-point (drives the split factor).
    InterlockedMax(eta_max_x[i], uint(max(eta_x, 0.0) * FIXED_SCALE), dummy);
    InterlockedMax(eta_max_y[i], uint(max(eta_y, 0.0) * FIXED_SCALE), dummy);
    InterlockedMax(eta_max_z[i], uint(max(eta_z, 0.0) * FIXED_SCALE), dummy);
}

// Zero the structure-score accumulators at the start of a scoring window.
[shader("compute")]
[numthreads(256, 1, 1)]
void clearStructureScores(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<uint> eta_n_high, RWStructuredBuffer<uint> eta_n_low,
    RWStructuredBuffer<uint> eta_max_x, RWStructuredBuffer<uint> eta_max_y, RWStructuredBuffer<uint> eta_max_z,
    RWStructuredBuffer<uint> eta_n_total,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    eta_n_high[i] = 0u; eta_n_low[i] = 0u;
    eta_max_x[i] = 0u; eta_max_y[i] = 0u; eta_max_z[i] = 0u;
    eta_n_total[i] = 0u;
}
