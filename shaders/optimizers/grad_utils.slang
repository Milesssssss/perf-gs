// Utility kernels for int64 gradient-accumulator buffers.

[shader("compute")]
[numthreads(256, 1, 1)]
void clearGradBuffer(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<int64_t> grad_buffer,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    grad_buffer[i] = int64_t(0);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void checkSaturation(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<int64_t> grad_buffer,
    RWStructuredBuffer<uint> saturation_flag,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    // With int64 overflow is essentially impossible, but check for NaN-equivalent.
    int64_t v = grad_buffer[i];
    if (v == int64_t(0x7FFFFFFFFFFFFFFF) || v == -int64_t(0x7FFFFFFFFFFFFFFF) - int64_t(1))
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
