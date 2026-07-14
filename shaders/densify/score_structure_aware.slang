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
// SADGS PARITY (F1): the reference measures raw projected axis length with only
// a +1e-8 numerical guard inside the sqrt (freq_utils.py:342 axis_lengths =
// sqrt(u^2+v^2+1e-8)). A larger floor (we previously used 0.3, matching the
// renderer's cov2d low-pass dilation) biases every axis up by ~sqrt(0.3)=0.55px,
// which nudges borderline splats over the eta>1 split boundary and suppresses
// eta<0.1 prune votes — a growth bias that desyncs from SADGS's calibrated
// eta>1 threshold. Match the reference: numerical guard only.
static const float COV_DILATION = 1e-8;

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

// Box-Muller: two uniforms -> a pair of independent standard-normal samples.
// SADGS jitters the structure-field sample point with a Gaussian drawn from the
// projected 2D covariance (freq_utils.py: eps1,eps2 = randn); the previous
// uniform[-0.5,0.5] box jitter had std 1/sqrt(12) ~ 0.289 in axis units, so it
// stayed far closer to the Gaussian center and under-observed the footprint.
float2 _randn2(uint seed)
{
    float u1 = max(_rand01(seed), 1e-7);
    float u2 = _rand01(seed + 1u);
    float r = sqrt(-2.0 * log(u1));
    float a = 6.28318530718 * u2;
    return float2(r * cos(a), r * sin(a));
}

// Bilinear sample of the packed structure tensor at pixel coordinate `p`
// (integer coords = texel centers), with ZERO-padding outside [0,dim-1] — this
// matches SADGS's F.grid_sample(align_corners=True) default (bilinear interp,
// zeros outside), replacing the previous nearest-neighbour + border-clamp fetch.
// Border clamp biased samples that fell off-image toward the edge texel's
// (often high) structure value; zero-padding treats out-of-frame as no signal.
float3 _sample_structure_bilinear(Texture2D<float4> tex, float2 p, uint2 dim)
{
    float2 pf = floor(p);
    int2 i0 = int2(pf);
    float2 f = p - pf;                 // bilinear weights in [0,1)
    int W = int(dim.x), H = int(dim.y);

    float3 acc = float3(0.0, 0.0, 0.0);
    // 4 taps; each out-of-bounds tap contributes 0 (zero padding).
    [unroll]
    for (int dy = 0; dy < 2; ++dy)
    {
        [unroll]
        for (int dx = 0; dx < 2; ++dx)
        {
            int tx = i0.x + dx;
            int ty = i0.y + dy;
            if (tx < 0 || tx >= W || ty < 0 || ty >= H) continue;
            float wx = (dx == 0) ? (1.0 - f.x) : f.x;
            float wy = (dy == 0) ? (1.0 - f.y) : f.y;
            acc += (wx * wy) * tex[int2(tx, ty)].xyz;
        }
    }
    return acc;
}

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
    // THIS view's per-gaussian 2D-mean gradient (fixed-point pixel-space sum
    // over covered pixels, from backward). SADGS's is_active_vis ALSO requires
    // a high per-view gradient (freq_utils.py:207-219: norm(viewspace grad) >
    // freq_grad_threshold). Without this per-view grad gate, eta is folded in
    // from EVERY visible view, saturating the high fraction (~everywhere >0.8)
    // and inflating the running eta_max. These buffers are the same per-iter
    // grad seeds accumulate_iter folds into the window accumulator; we read
    // them BEFORE accumulate_iter clears them.
    StructuredBuffer<int64_t> grad_2d_x_iter,
    StructuredBuffer<int64_t> grad_2d_y_iter,
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
    // SADGS is_active_vis per-view grad floor (freq_grad_threshold, default
    // 0.00002 on the NDC-converted grad). This value is expressed in the SAME
    // convention as grad_2d_*_iter after NDC scaling (see the gate below), so
    // the host passes it pre-scaled. 0 disables the gate (offline/test path).
    uniform float per_view_grad_thresh,
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

    // PER-VIEW GRADIENT GATE (SADGS is_active_vis, freq_utils.py:207-219): only
    // fold eta in from views where this gaussian has a high screen-space
    // gradient. grad_2d_*_iter hold the per-pixel-summed PIXEL-space grad for
    // THIS view; convert to NDC per-component exactly as accumulate_iter.slang
    // does (dL/d(ndc) = dL/d(pixel) * dim/2) before taking the norm, so the
    // host threshold is comparable to SADGS's freq_grad_threshold. Skipping this
    // is what let eta accumulate from every visible view, saturating high_ratio
    // (~everywhere >0.8) and inflating the running eta_max -> oversized splits.
    if (per_view_grad_thresh > 0.0)
    {
        float gx = fromFixed(grad_2d_x_iter[i]) * (frame_dim.x * 0.5);
        float gy = fromFixed(grad_2d_y_iter[i]) * (frame_dim.y * 0.5);
        if (sqrt(gx * gx + gy * gy) < per_view_grad_thresh) return;
    }

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
    // Projected 2D length of each axis (the x,y rows of A; row 2 of J is 0),
    // with only the +1e-8 numerical guard (COV_DILATION) inside the sqrt — this
    // is the RAW projected axis length the reference measures for eta
    // (freq_utils.py:342 axis_lengths = sqrt(u²+v²+1e-8)), NOT the renderer's
    // +0.3-dilated footprint. (The +0.3 low-pass dilation IS applied, but only
    // to the covariance used for jitter sampling below, so the jitter covers the
    // rasterized footprint while eta measures the ideal projected extent.)
    float len_vx = sqrt(A[0][0] * A[0][0] + A[1][0] * A[1][0] + COV_DILATION);
    float len_vy = sqrt(A[0][1] * A[0][1] + A[1][1] * A[1][1] + COV_DILATION);
    float len_vz = sqrt(A[0][2] * A[0][2] + A[1][2] * A[1][2] + COV_DILATION);

    // --- Sample S̄ at a jittered footprint point, get local min wavelength. ---
    // SADGS (freq_utils.py) draws the sample offset from the full projected 2D
    // covariance Σ₂ = A Aᵀ (its screen x,y block), not a per-axis box: it
    // Cholesky-factors Σ₂ and offsets the center by L·eps, eps ~ N(0,I). This
    // reaches the correlated, off-axis parts of the footprint a per-axis box
    // jitter never sampled. Σ₂ from A's first two rows (row 2 of J is 0), with
    // the same +0.3 low-pass dilation project.slang adds so the sampled extent
    // matches the RASTERIZED footprint rather than the ideal projected one.
    float cov_xx = A[0][0] * A[0][0] + A[0][1] * A[0][1] + A[0][2] * A[0][2] + 0.3;
    float cov_xy = A[0][0] * A[1][0] + A[0][1] * A[1][1] + A[0][2] * A[1][2];
    float cov_yy = A[1][0] * A[1][0] + A[1][1] * A[1][1] + A[1][2] * A[1][2] + 0.3;

    // Cholesky of the 2D covariance (matches SADGS L11/L21/L22):
    //   L11 = sqrt(cov_xx); L21 = cov_xy / L11; L22 = sqrt(cov_yy - L21²).
    float L11 = sqrt(max(cov_xx, 1e-6));
    float L21 = cov_xy / L11;
    float L22 = sqrt(max(cov_yy - L21 * L21, 1e-6));

    uint seed = i * 2654435761u + pass_idx * 40503u;
    float2 eps = _randn2(seed);
    float2 sample_px = mean_px + float2(L11 * eps.x, L21 * eps.x + L22 * eps.y);

    uint2 dim; structure_tensor.GetDimensions(dim.x, dim.y);
    // Bilinear fetch with zero-padding outside the image (SADGS grid_sample
    // align_corners=True default), replacing nearest + border clamp.
    float3 sbar = _sample_structure_bilinear(structure_tensor, sample_px, dim);

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
