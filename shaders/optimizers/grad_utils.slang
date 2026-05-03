// Utility kernels for the gradient-accumulator buffers.
//
// Gradients are int32 fixed-point (PLAN.md Q9 + Q10). We keep 14 SoA
// buffers — one per scalar component of the gaussian parameters — so
// ``InterlockedAdd`` works cross-IHV.

// Clear a single int32 grad buffer to zero.
//
// Dispatched 14 times per training step (once per component). We don't
// fuse them into one mega-kernel because each buffer is a separate
// resource and SlangPy's dispatch marshalls resources by name — the fused
// version would just bloat the signature.
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

// Saturation check: if ANY int32 grad value sits at INT_MIN or INT_MAX
// the fixed-point accumulator likely wrapped. Writes 1 into a shared
// flag. Python reads the flag after the step and warns.
//
// Dispatched once per component buffer; all 14 share the same flag.
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
