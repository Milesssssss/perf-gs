// Per-iter densification tracking fold (int32 fixed-point).

#include "../common/types.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void accumulateIter(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<int> grad_2d_x_iter,
    RWStructuredBuffer<int> grad_2d_y_iter,
    RWStructuredBuffer<int> grad_mean_2d_mag_accum,
    RWStructuredBuffer<uint> visibility_count_accum,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;

    int ix = grad_2d_x_iter[i];
    int iy = grad_2d_y_iter[i];

    if (ix == 0 && iy == 0) return;

    float fx = fromFixed(ix);
    float fy = fromFixed(iy);
    float m = sqrt(fx * fx + fy * fy);

    InterlockedAdd(grad_mean_2d_mag_accum[i], toFixed(m));
    uint dummy;
    InterlockedAdd(visibility_count_accum[i], 1u, dummy);

    grad_2d_x_iter[i] = 0;
    grad_2d_y_iter[i] = 0;
}
