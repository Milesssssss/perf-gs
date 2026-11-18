// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

[shader("compute")]
[numthreads(16, 16, 1)]
void gatherTarget(uint3 tid : SV_DispatchThreadID, Texture2D<float4> source, RWTexture2D<float4> target,
                  uniform uint2 size, uniform uint2 phase, uniform uint stride)
{
    if (all(tid.xy < size))
        target[tid.xy] = source.Load(int3(tid.xy * stride + phase, 0));
}
