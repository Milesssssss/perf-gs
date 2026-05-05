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
    float3 color_logit;     // sigmoid(color_logit) = linear RGB in [0, 1]
};

struct Projected2DGaussian
{
    float2 mean;            // screen-space center (in pixels)
    float3 cov2d;           // 2x2 covariance packed as (cov_00, cov_01, cov_11)
    float depth_vs;         // view-space depth (positive-z is forward in our convention)
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

// Fixed-point atomic gradient accumulation.
// Scale=4096 balances precision vs overflow headroom:
// Max per-pixel contribution ~0.1 → 410 fixed. 500k pixels → 205M < 2.1B (int32 max).
static const float FIXED_SCALE = 4096.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

int toFixed(float f)   { return int(f * FIXED_SCALE); }
float fromFixed(int i) { return float(i) * FIXED_INV_SCALE; }

#endif // TRAINER_TYPES_SLANG
