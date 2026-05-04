// Per-iter → per-window fold for densification tracking.
//
// The backward rasterizer atomically accumulates the 2D mean gradient
// VECTOR (not per-pixel magnitude) into (grad_2d_x_iter, grad_2d_y_iter)
// for each gaussian. After the backward completes, this kernel runs
// once per gaussian to:
//
//   1. Compute the norm of the iter-summed 2D gradient:
//        m = ‖(grad_2d_x_iter[g], grad_2d_y_iter[g])‖
//   2. Add m to the window-level fixed-point buffer (grad_mean_2d_mag_accum).
//   3. If the gaussian was visible this iter (non-zero iter vector),
//      increment visibility_count_accum[g] by 1.
//   4. Zero the per-iter vector buffers for the next step.
//
// This matches gsplat's DefaultStrategy: see their default.py,
// ``state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))`` and
// ``state["count"].index_add_(0, gs_ids, ones)`` where ``grads`` is
// the per-iter per-gaussian (already pixel-summed) screen-space
// gradient. Our ``grad_2d_*_iter`` buffers play the role of that
// pixel-summed gradient.

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

    // Skip untouched gaussians entirely — don't bump visibility, don't
    // touch mag_accum. Saves work and keeps ``count`` honest.
    if (ix == 0 && iy == 0) return;

    float fx = fromFixed(ix);
    float fy = fromFixed(iy);
    float m = sqrt(fx * fx + fy * fy);

    InterlockedAdd(grad_mean_2d_mag_accum[i], toFixed(m));
    uint dummy;
    InterlockedAdd(visibility_count_accum[i], 1u, dummy);

    // Clear per-iter vectors so the next backward starts from zero.
    grad_2d_x_iter[i] = 0;
    grad_2d_y_iter[i] = 0;
}
