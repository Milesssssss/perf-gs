// Per-iter densification tracking fold (int64 fixed-point).

#include "../common/types.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void accumulateIter(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<int64_t> grad_2d_x_iter,
    RWStructuredBuffer<int64_t> grad_2d_y_iter,
    RWStructuredBuffer<int64_t> grad_mean_2d_mag_accum,
    RWStructuredBuffer<uint> visibility_count_accum,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;

    int64_t ix = grad_2d_x_iter[i];
    int64_t iy = grad_2d_y_iter[i];

    if (ix == int64_t(0) && iy == int64_t(0)) return;

    float fx = fromFixed(ix);
    float fy = fromFixed(iy);
    float m = sqrt(fx * fx + fy * fy);

    InterlockedAdd(grad_mean_2d_mag_accum[i], toFixed(m));
    uint udummy;
    InterlockedAdd(visibility_count_accum[i], 1u, udummy);

    grad_2d_x_iter[i] = int64_t(0);
    grad_2d_y_iter[i] = int64_t(0);
}
