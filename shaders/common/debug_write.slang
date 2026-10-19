// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Trivial compute kernel: write a constant to every pixel to confirm
// the output_image binding and dispatch plumbing work at all. Resources
// live on the entry-point signature as parameters, which is SlangPy's
// idiomatic binding surface.

[shader("compute")]
[numthreads(16, 16, 1)]
void debugWrite(uint3 dtid: SV_DispatchThreadID, RWTexture2D<float4> output_image)
{
    output_image[dtid.xy] = float4(0.3, 0.7, 0.1, 1.0);
}
