// Clear the v2 densification tracking buffers. Called by Python at the
// start of each densification window (every N iters).
//
// Buffers:
//   grad_mean_2d_mag_accum  - fixed-point sum of per-iter ‖grad_2d‖
//   visibility_count_accum  - number of iters each gaussian was visible
//   grad_2d_x_iter          - per-iter vector accumulator (x component)
//   grad_2d_y_iter          - per-iter vector accumulator (y component)
//
// The per-iter vectors are normally cleared inside accumulate_iter at
// the end of each backward. We clear them here too as an idempotent
// reset — cheap and keeps state well-defined across window boundaries.

[shader("compute")]
[numthreads(256, 1, 1)]
void clearTracking(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<int> grad_mean_2d_mag_accum,
    RWStructuredBuffer<uint> visibility_count_accum,
    RWStructuredBuffer<int> grad_2d_x_iter,
    RWStructuredBuffer<int> grad_2d_y_iter,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;
    grad_mean_2d_mag_accum[i] = 0;
    visibility_count_accum[i] = 0u;
    grad_2d_x_iter[i] = 0;
    grad_2d_y_iter[i] = 0;
}
