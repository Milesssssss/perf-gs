// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Writes a constant to every pixel to verify the output_image binding and dispatch plumbing.

[shader("compute")]
[numthreads(16, 16, 1)]
void debugWrite(uint3 dtid: SV_DispatchThreadID, RWTexture2D<float4> output_image)
{
    output_image[dtid.xy] = float4(0.3, 0.7, 0.1, 1.0);
}
