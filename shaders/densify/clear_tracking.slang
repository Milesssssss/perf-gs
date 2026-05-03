// Clear the v2 densification tracking buffers. Called by Python at the
// start of each densification window (every N iters). Both buffers are
// SoA per-gaussian; one atomic-accumulated fixed-point, one raw uint.

[shader("compute")]
[numthreads(256, 1, 1)]
void clearTracking(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<int> grad_mean_2d_mag_accum,
    RWStructuredBuffer<uint> pixel_coverage_accum,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    grad_mean_2d_mag_accum[i] = 0;
    pixel_coverage_accum[i] = 0u;
}
