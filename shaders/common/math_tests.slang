// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Scalar wrappers of the gaussian math for tests through SlangPy's functional API.
// Opacity arguments are probabilities, converted to the primitives' logit here.

#include "math.slang"
#include "../primitives/gaussian/eval.slang"
#include "../primitives/gaussian/snugbox.slang"
#include "../primitives/gaussian/accutile.slang"

static float _opacityToLogit(float p)
{
    // Clamp away from 0 and 1 so log() is finite, as gaussian_records does.
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

// Alpha at `pixel`; cov2d is (cov_00, cov_01, cov_11).
float evalGaussian2D(float2 pixel, float2 mean_2d, float3 cov2d, float opacity)
{
    return evalGaussian2DFromCov(pixel, mean_2d, cov2d, _opacityToLogit(opacity), 1.0);
}

// (min.x, min.y, max.x, max.y); a degenerate box at the mean when opacity is too low.
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
