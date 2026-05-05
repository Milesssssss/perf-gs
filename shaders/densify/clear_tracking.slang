// Clear the densification tracking buffers (float version).

[shader("compute")]
[numthreads(256, 1, 1)]
void clearTracking(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<float> grad_mean_2d_mag_accum,
    RWStructuredBuffer<uint> visibility_count_accum,
    RWStructuredBuffer<float> grad_2d_x_iter,
    RWStructuredBuffer<float> grad_2d_y_iter,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    grad_mean_2d_mag_accum[i] = 0.0;
    visibility_count_accum[i] = 0u;
    grad_2d_x_iter[i] = 0.0;
    grad_2d_y_iter[i] = 0.0;
}
