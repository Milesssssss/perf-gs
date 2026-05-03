// Shared type definitions for the v0 forward-only pipeline.
// Kept intentionally minimal: no Adam state, no gradients, no SH.

#ifndef TRAINER_TYPES_SLANG
#define TRAINER_TYPES_SLANG

struct Gaussian3D
{
    float3 mean;            // world-space position
    float3 log_scale;       // exp(log_scale) = per-axis scale of the 3D gaussian
    float4 rot_quat;        // rotation quaternion (w, x, y, z)
    float opacity;          // [0, 1]
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

#endif // TRAINER_TYPES_SLANG
