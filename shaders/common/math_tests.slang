// Thin scalar-parameter wrappers around the math utilities used by the
// gaussian primitive, exposed so SlangPy's functional API can call them
// from Python without us having to author a compute entry point.
//
// Each wrapper takes plain scalars / small vectors, returns a scalar or
// small vector, and has no side effects. The functional API auto-vectorizes
// across input tensor shapes.
//
// After the logit-opacity refactor the shader-side primitives take a
// pre-sigmoid opacity logit. The test wrappers keep the "probability"
// interface for Python callers (matching the brute-force numpy math)
// and apply logit() once before calling into the shader primitives.

#include "math.slang"
#include "../primitives/gaussian/eval.slang"
#include "../primitives/gaussian/snugbox.slang"
#include "../primitives/gaussian/accutile.slang"

static float _opacityToLogit(float p)
{
    // Clamp strictly away from 0 and 1 so log() is finite; matches the
    // clamp applied in write_gaussians.
    p = clamp(p, 1e-5, 1.0 - 1e-5);
    return log(p / (1.0 - p));
}

float sigmoidScalar(float x)
{
    return sigmoid(x);
}

float3x3 rotMatFromQuat(float4 q)
{
    return matrixFromQuaternion(q);
}

float3x3 covFromScaleQuat(float3 scale, float4 quat)
{
    return buildCovariance3D(scale, quat);
}

// Evaluate 2D gaussian alpha at a pixel.
// pixel: 2D position to sample at
// mean_2d: gaussian center in pixel space
// cov2d: (cov_00, cov_01, cov_11) of the 2D covariance
// opacity: probability in (0, 1) — converted to logit internally.
float evalGaussian2D(float2 pixel, float2 mean_2d, float3 cov2d, float opacity)
{
    return evalGaussian2DFromCov(pixel, mean_2d, cov2d, _opacityToLogit(opacity), 1.0);
}

// Returns (bbox_min.x, bbox_min.y, bbox_max.x, bbox_max.y).
// For the opacity-too-low case, returns a degenerate box at the mean.
float4 snugBoxTest(float2 mean_2d, float3 cov2d, float opacity)
{
    float2 mn, mx;
    snugBox(mean_2d, cov2d, _opacityToLogit(opacity), mn, mx);
    return float4(mn.x, mn.y, mx.x, mx.y);
}

// Returns (x_min, x_max). Sentinel (+inf, -inf) when no intersection.
float2 accuTileRowTest(
    float2 mean_2d, float3 cov2d, float opacity,
    float row_y_bot, float row_y_top)
{
    float opacity_logit = _opacityToLogit(opacity);
    EllipseParams e;
    if (!computeEllipseParams(cov2d, opacity_logit, e))
        return float2(1e30, -1e30);
    float x_d_args, y_d_args;
    computeExtremumOffsets(e, x_d_args, y_d_args);
    float2 mn, mx;
    snugBox(mean_2d, cov2d, opacity_logit, mn, mx);
    return accuTileRowXRange(e, mean_2d.y, mn.x, mx.x, y_d_args, mean_2d,
                             row_y_bot, row_y_top);
}
