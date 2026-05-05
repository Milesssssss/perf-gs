// Utility kernels for float gradient-accumulator buffers.

// Clear a single float grad buffer to zero.
[shader("compute")]
[numthreads(256, 1, 1)]
void clearGradBuffer(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<float> grad_buffer,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    grad_buffer[i] = 0.0;
}

// Saturation check (NaN/Inf detection for float buffers).
[shader("compute")]
[numthreads(256, 1, 1)]
void checkSaturation(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<float> grad_buffer,
    RWStructuredBuffer<uint> saturation_flag,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    float v = grad_buffer[i];
    if (isnan(v) || isinf(v))
    {
        uint dummy;
        InterlockedOr(saturation_flag[0], 1u, dummy);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void clearSaturationFlag(RWStructuredBuffer<uint> saturation_flag)
{
    saturation_flag[0] = 0;
}
