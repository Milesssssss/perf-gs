// P0 SPIKE (throwaway): prove Slang interface generics + [Differentiable]
// compile & run identically on Vulkan + D3D12 via the SAME load_program /
// create_compute_kernel / dispatch path kernels.py uses.
//
// Two primitives (Gaussian-like, disk-like) conform to ONE differentiable
// interface (ISampler) and are evaluated + differentiated through ONE generic
// core (sampleGeneric<S>), instantiated by per-type compute entry points. If
// the parity test passes, the plan's core mechanism is validated: a
// monomorphized generic rasterize/eval core with zero dynamic dispatch, where
// swapping the primitive is just supplying a different conforming type.
//
// Delete once P2 lands the real IPrimitive core.

#ifndef SPIKE_GENERIC_SLANG
#define SPIKE_GENERIC_SLANG

// The differentiable primitive contract. IDifferentiable inheritance is what
// lets the generic core differentiate w.r.t. the primitive's stored fields.
interface ISampler : IDifferentiable
{
    [Differentiable]
    float eval(no_diff float2 pixel);
}

// Gaussian-like: isotropic exp falloff of a quadratic form (the 3DGS analog).
struct GaussSampler : ISampler
{
    float2 mean;
    float  logScale;
    float  opacity;   // pre-sigmoid logit

    [Differentiable]
    float eval(no_diff float2 pixel)
    {
        float2 d = pixel - mean;
        float s = exp(logScale);
        float inv = 1.0 / (s * s);
        float power = -0.5 * (d.x * d.x + d.y * d.y) * inv;
        return (1.0 / (1.0 + exp(-opacity))) * exp(power);
    }
}

// Disk-like: smoothstep falloff to a hard radius (the planned 2D-disk analog).
struct DiskSampler : ISampler
{
    float2 mean;
    float  logRadius;
    float  opacity;   // pre-sigmoid logit

    [Differentiable]
    float eval(no_diff float2 pixel)
    {
        float r = exp(logRadius);
        float dist = length(pixel - mean);
        float t = clamp(1.0 - dist / r, 0.0, 1.0);
        float falloff = t * t * (3.0 - 2.0 * t);
        return (1.0 / (1.0 + exp(-opacity))) * falloff;
    }
}

// THE single generic differentiable core shared by every primitive. This is
// the shape the real forward/backward rasterizer takes in P2.
[Differentiable]
float sampleGeneric<S : ISampler>(S s, no_diff float2 pixel)
{
    return s.eval(pixel);
}

// ---- per-type compute entry points (thin monomorphized instantiations) ----

[shader("compute")]
[numthreads(64, 1, 1)]
void gaussEval(
    uint3 tid : SV_DispatchThreadID,
    StructuredBuffer<float> params,        // [n*4] mean.x, mean.y, logScale, opacity
    StructuredBuffer<float> pixels,        // [n*2]
    RWStructuredBuffer<float> outAlpha,    // [n]
    RWStructuredBuffer<float> outGrad,     // [n*4] d(alpha)/d(field)
    uint n)
{
    uint i = tid.x;
    if (i >= n) return;

    GaussSampler s;
    s.mean = float2(params[i * 4 + 0], params[i * 4 + 1]);
    s.logScale = params[i * 4 + 2];
    s.opacity = params[i * 4 + 3];
    float2 px = float2(pixels[i * 2 + 0], pixels[i * 2 + 1]);

    outAlpha[i] = sampleGeneric<GaussSampler>(s, px);

    // Backward through the generic core: seed dL/dalpha = 1.
    var dp = diffPair(s);
    bwd_diff(sampleGeneric<GaussSampler>)(dp, px, 1.0);
    outGrad[i * 4 + 0] = dp.d.mean.x;
    outGrad[i * 4 + 1] = dp.d.mean.y;
    outGrad[i * 4 + 2] = dp.d.logScale;
    outGrad[i * 4 + 3] = dp.d.opacity;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void diskEval(
    uint3 tid : SV_DispatchThreadID,
    StructuredBuffer<float> params,        // [n*4] mean.x, mean.y, logRadius, opacity
    StructuredBuffer<float> pixels,        // [n*2]
    RWStructuredBuffer<float> outAlpha,    // [n]
    RWStructuredBuffer<float> outGrad,     // [n*4] d(alpha)/d(field)
    uint n)
{
    uint i = tid.x;
    if (i >= n) return;

    DiskSampler s;
    s.mean = float2(params[i * 4 + 0], params[i * 4 + 1]);
    s.logRadius = params[i * 4 + 2];
    s.opacity = params[i * 4 + 3];
    float2 px = float2(pixels[i * 2 + 0], pixels[i * 2 + 1]);

    outAlpha[i] = sampleGeneric<DiskSampler>(s, px);

    var dp = diffPair(s);
    bwd_diff(sampleGeneric<DiskSampler>)(dp, px, 1.0);
    outGrad[i * 4 + 0] = dp.d.mean.x;
    outGrad[i * 4 + 1] = dp.d.mean.y;
    outGrad[i * 4 + 2] = dp.d.logRadius;
    outGrad[i * 4 + 3] = dp.d.opacity;
}

#endif // SPIKE_GENERIC_SLANG
