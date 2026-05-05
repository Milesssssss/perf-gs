// Utility kernels for int32 gradient-accumulator buffers.

[shader("compute")]
[numthreads(256, 1, 1)]
void clearGradBuffer(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<int> grad_buffer,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    grad_buffer[i] = 0;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void checkSaturation(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<int> grad_buffer,
    RWStructuredBuffer<uint> saturation_flag,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    int v = grad_buffer[i];
    if (v == 0x7fffffff || v == -0x7fffffff - 1)
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
