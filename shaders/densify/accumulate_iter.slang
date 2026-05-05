// Per-iter densification tracking fold (float version).
//
// The backward rasterizer atomically accumulates absgrad into
// (grad_2d_x_iter, grad_2d_y_iter) as float. This kernel folds the
// per-iter norm into the window-level accumulator.

#include "../common/types.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void accumulateIter(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<float> grad_2d_x_iter,
    RWStructuredBuffer<float> grad_2d_y_iter,
    RWStructuredBuffer<float> grad_mean_2d_mag_accum,
    RWStructuredBuffer<uint> visibility_count_accum,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;

    float fx = grad_2d_x_iter[i];
    float fy = grad_2d_y_iter[i];

    if (fx == 0.0 && fy == 0.0) return;

    float m = sqrt(fx * fx + fy * fy);

    InterlockedAdd(grad_mean_2d_mag_accum[i], m);
    uint dummy;
    InterlockedAdd(visibility_count_accum[i], 1u, dummy);

    grad_2d_x_iter[i] = 0.0;
    grad_2d_y_iter[i] = 0.0;
}
