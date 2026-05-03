// Shared type definitions for 3DGS training pipeline
// All structs used across shaders are defined here

#ifndef TYPES_SLANG
#define TYPES_SLANG

// ============================================================================
// Constants
// ============================================================================

static const uint TILE_SIZE = 16;
static const uint MAX_GAUSSIANS_PER_TILE = 1024;
static const uint MAX_TILES = 14400;  // (1920/16)^2 for max 1920x1920 resolution

static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

static const float ADAM_BETA1 = 0.9f;
static const float ADAM_BETA2 = 0.999f;
static const float ADAM_EPSILON = 1e-8f;

#define UINT_MAX 0xffffffff

// ============================================================================
// Core Gaussian Structures
// ============================================================================

struct Struct_Gaussian3D
{
    float3 mean;            // 3D position in world space
    float3 log_scale;       // Log of scale (exp(log_scale) = actual scale)
    float4 rotation_quat;   // Quaternion rotation (r, x, y, z format)
    float opacity;          // Opacity [0,1] range
    float3 color_logit;     // Color in logit space (sigmoid(color_logit) = RGB)
};

struct Struct_Projected2DGaussian
{
    float2 mean;            // 2D screen position
    float3 cov2d;           // 2D covariance matrix packed as (cov_00, cov_01, cov_11)
    float depth_vs;         // View-space depth (negated for sorting)
};

// ============================================================================
// Optimizer State
// ============================================================================

struct Struct_AdamState
{
    float3 mean_m;
    float3 mean_v;
    float3 scale_m;
    float3 scale_v;
    float4 rotation_m;
    float4 rotation_v;
    float3 color_m;
    float3 color_v;
    float opacity_m;
    float opacity_v;
};

struct AdamConfig
{
    float lr_mean;
    float lr_scale;
    float lr_rotation;
    float lr_color;
    float lr_opacity;
    float bias_correction1;
    float bias_correction2;
};

struct GaussianGradients
{
    float3 mean;
    float mean2d_mag;       // Not used to update learnable params, just for densification heuristics
    float3 scale;
    float4 rotation;
    float3 color;
    float opacity;
};

// ============================================================================
// Camera Data
// ============================================================================

struct CameraData
{
    float4x4 view_matrix;       // World to view transform
    float4x4 projection_matrix; // View to clip transform
    float2 image_size;          // Width, height of target image
    uint image_index;           // Index into texture array
    float _pad;                 // Padding for alignment
};

// ============================================================================
// Frame Constants (passed per-dispatch)
// ============================================================================

struct FrameConstants
{
    float2 frame_dim;           // Current render resolution
    uint max_splats;            // Maximum number of gaussians
    uint num_gaussians;         // Current active gaussian count
    uint camera_index;          // Index into camera buffer
    uint iteration;             // Current training iteration
    uint batch_iteration;       // iteration / num_views
    
    // Gradient descent flags
    bool descend_mean;
    bool descend_scale;
    bool descend_rotation;
    bool descend_color;
    bool descend_opacity;
    
    // Densification params
    bool enable_densification;
    uint start_densification_iter;
    uint end_densification_iter;
};

// ============================================================================
// Gradient Accumulator (for groupshared reduction in backward pass)
// ============================================================================

struct GradientAccumulator
{
    int grad_mean_x, grad_mean_y, grad_mean_z;
    int grad_mean2d_magnitude;
    int grad_log_scale_x, grad_log_scale_y, grad_log_scale_z;
    int grad_quat_x, grad_quat_y, grad_quat_z, grad_quat_w;
    int grad_color_r, grad_color_g, grad_color_b;
    int grad_opacity;
};

// ============================================================================
// Helper Functions for Fixed-Point Conversion
// ============================================================================

int toFixed(float f)
{
    return int(f * FIXED_SCALE);
}

float fromFixed(int i)
{
    return float(i) * FIXED_INV_SCALE;
}

// ============================================================================
// Adam Helper Functions
// ============================================================================

AdamConfig make_adam_config(
    float lr_mean,
    float lr_scale,
    float lr_rotation,
    float lr_color,
    float lr_opacity,
    uint step
)
{
    AdamConfig config;
    config.lr_mean = lr_mean;
    config.lr_scale = lr_scale;
    config.lr_rotation = lr_rotation;
    config.lr_color = lr_color;
    config.lr_opacity = lr_opacity;
    config.bias_correction1 = 1.0f - pow(ADAM_BETA1, float(step));
    config.bias_correction2 = 1.0f - pow(ADAM_BETA2, float(step));
    return config;
}

Struct_AdamState zero_adam_state()
{
    Struct_AdamState s;
    s.mean_m = float3(0.0f, 0.0f, 0.0f);
    s.mean_v = float3(0.0f, 0.0f, 0.0f);
    s.scale_m = float3(0.0f, 0.0f, 0.0f);
    s.scale_v = float3(0.0f, 0.0f, 0.0f);
    s.rotation_m = float4(0, 0, 0, 0);
    s.rotation_v = float4(0, 0, 0, 0);
    s.color_m = float3(0.0f, 0.0f, 0.0f);
    s.color_v = float3(0.0f, 0.0f, 0.0f);
    s.opacity_m = 0.0f;
    s.opacity_v = 0.0f;
    return s;
}

// ============================================================================
// Random Number Generation
// ============================================================================

float random(uint seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return float(seed) / float(UINT_MAX);
}

#endif // TYPES_SLANG