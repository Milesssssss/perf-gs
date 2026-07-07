// Shared type definitions for the v0 forward-only pipeline.
// Kept intentionally minimal: no Adam state, no gradients, no SH.

#ifndef TRAINER_TYPES_SLANG
#define TRAINER_TYPES_SLANG

struct Gaussian3D
{
    float3 mean;            // world-space position
    float3 log_scale;       // exp(log_scale) = per-axis scale of the 3D gaussian
    float4 rot_quat;        // rotation quaternion (w, x, y, z)
    float opacity;          // pre-sigmoid opacity logit; sigmoid(opacity) is the
                            // real probability in [0, 1]. Parameterized as a logit
                            // so Adam's opacity LR operates on an unbounded scalar
                            // (matches gsplat; see shaders/primitives/gaussian/eval.slang).
    float3 sh0;             // SH band-0 (DC) color coeff; linear RGB = 0.5 + C0*sh0
                            // (affine, unbounded — see shDC in math.slang). Replaces
                            // the old sigmoid(color_logit) parameterization.
};

struct Projected2DGaussian
{
    float2 mean;            // screen-space center (in pixels)
    float3 cov2d;           // 2x2 covariance packed as (cov_00, cov_01, cov_11)
    float depth_vs;         // view-space depth (positive-z is forward in our convention)
};

// Compact per-gaussian record for the blend backward's HOT random gather.
// The blend backward reads splats in sorted (tile, depth) order, so g_idx is
// a random permutation and gaussians[g_idx] / projected_2d[g_idx] are scatter
// gathers. Those two fat structs total 80 B but the blend uses only these 4
// fields (36 B): gathering the whole 80 B pollutes L2 (~45% useful bytes).
// This packs exactly what the blend needs into one 36 B record so the gather
// is ~100% useful. Written by bin_count (which already has all four values in
// hand), consumed by backward_persplat. NOT read back to the CPU.
struct BlendData
{
    float2 mean;            // == Projected2DGaussian.mean
    float3 cov2d;           // == Projected2DGaussian.cov2d
    float  opacity;         // == Gaussian3D.opacity (pre-sigmoid logit)
    float3 sh0;             // == Gaussian3D.sh0
};

struct CameraData
{
    float4x4 view_matrix;   // world -> view
    float4x4 proj_matrix;   // view  -> clip
    float2 image_size;      // width, height in pixels
};

// Per-gaussian Adam first/second-moment state. AoS matches the read
// pattern: the Adam kernel loads all m/v for one gaussian, steps it,
// writes it back. No atomics needed (one thread per gaussian).
struct AdamState
{
    float3 mean_m;
    float3 mean_v;
    float3 log_scale_m;
    float3 log_scale_v;
    float4 rot_m;
    float4 rot_v;
    float  opacity_m;
    float  opacity_v;
    float3 color_m;
    float3 color_v;
};

// Fixed-point atomic gradient accumulation (int64, no overflow).
static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

int64_t toFixed(float f)   { return int64_t(f * FIXED_SCALE); }
float fromFixed(int64_t i) { return float(i) * FIXED_INV_SCALE; }

#endif // TRAINER_TYPES_SLANG
