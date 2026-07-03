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
    uniform float2 frame_dim,
    uniform uint max_splats)
{
    uint i = dtid.x;
    if (i >= max_splats) return;

    int64_t ix = grad_2d_x_iter[i];
    int64_t iy = grad_2d_y_iter[i];

    if (ix == int64_t(0) && iy == int64_t(0)) return;

    // Convert the per-iter screen-space gradient from PIXEL units to NDC
    // ([-1,1]) units BEFORE taking the norm, matching gsplat's densification
    // signal (grads[...,0] *= width/2, grads[...,1] *= height/2). Our mean_2d
    // is in pixels, so dp_mean_2d.d is dL/d(pixel); since pixel=(ndc+1)*dim/2,
    // dL/d(ndc) = dL/d(pixel) * dim/2. This makes the signal resolution-
    // independent so a fixed grad threshold is portable across image sizes.
    // The x/y factors differ when W != H, so the scaling MUST be applied
    // per-component here, before the two are collapsed into a magnitude.
    // The loss-mean factor 1/(H*W*channels) is NOT applied here (it would
    // underflow the 2^16 fixed-point accumulator); it is folded into the
    // densification threshold on the host side instead.
    float fx = fromFixed(ix) * (frame_dim.x * 0.5);
    float fy = fromFixed(iy) * (frame_dim.y * 0.5);
    float m = sqrt(fx * fx + fy * fy);

    InterlockedAdd(grad_mean_2d_mag_accum[i], toFixed(m));
    uint udummy;
    InterlockedAdd(visibility_count_accum[i], 1u, udummy);

    grad_2d_x_iter[i] = int64_t(0);
    grad_2d_y_iter[i] = int64_t(0);
}
