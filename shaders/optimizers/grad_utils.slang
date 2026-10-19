// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Utility kernels for int64 gradient-accumulator buffers + Adam moment state.
#include "../common/types.slang"

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

// Zero a plain float buffer (SH features_rest / adam_rest init). count is the
// TOTAL scalar length, not a gaussian count.
[shader("compute")]
[numthreads(256, 1, 1)]
void clearFloatBuffer(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<float> buffer,
    uniform uint count)
{
    uint i = dtid.x;
    if (i >= count) return;
    buffer[i] = 0.0;
}

// Zero a plain uint buffer (fp16-packed SH features_rest / adam_rest init). A
// zero uint unpacks to two 0.0 fp16 halves. count is the TOTAL uint length.
[shader("compute")]
[numthreads(256, 1, 1)]
void clearUintBuffer(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<uint> buffer,
    uniform uint count)
{
    uint i = dtid.x;
    if (i >= count) return;
    buffer[i] = 0u;
}

// Zero all per-gaussian Adam first/second moments device-side. Replaces a host
// round-trip (readback-to-size + host zero upload) at set-gaussians time. Uses
// Slang's zero-init cast so it never enumerates AdamState's fields (which are
// generated from the contract) — a new moment field needs no edit here.
[shader("compute")]
[numthreads(256, 1, 1)]
void clearAdamState(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<AdamState> adam_state,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    adam_state[i] = (AdamState)0;
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
        InterlockedOr(saturation_flag[0], 1u);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void clearSaturationFlag(RWStructuredBuffer<uint> saturation_flag)
{
    saturation_flag[0] = 0;
}
