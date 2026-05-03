// Thin scalar-parameter wrappers around the math utilities used by the
// gaussian primitive, exposed so SlangPy's functional API can call them
// from Python without us having to author a compute entry point.
//
// Each wrapper takes plain scalars / small vectors, returns a scalar or
// small vector, and has no side effects. The functional API auto-vectorizes
// across input tensor shapes.

#include "math.slang"
#include "../primitives/gaussian/eval.slang"

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
// opacity: scalar opacity
float evalGaussian2D(float2 pixel, float2 mean_2d, float3 cov2d, float opacity)
{
    return evalGaussian2DFromCov(pixel, mean_2d, cov2d, opacity);
}
