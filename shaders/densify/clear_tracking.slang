// Clear the densification tracking buffers.

[shader("compute")]
[numthreads(256, 1, 1)]
void clearTracking(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<int64_t> grad_mean_2d_mag_accum,
    RWStructuredBuffer<uint> visibility_count_accum,
    RWStructuredBuffer<int64_t> grad_2d_x_iter,
    RWStructuredBuffer<int64_t> grad_2d_y_iter,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    grad_mean_2d_mag_accum[i] = int64_t(0);
    visibility_count_accum[i] = 0u;
    grad_2d_x_iter[i] = int64_t(0);
    grad_2d_y_iter[i] = int64_t(0);
}
