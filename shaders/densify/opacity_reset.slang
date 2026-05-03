// Periodic opacity reset (3DGS paper trick): force every gaussian's
// opacity to a small value (~0.01), so the next prune sweep drops any
// gaussian that was only contributing via opacity, not via useful
// geometry. The trainer then has to re-earn the opacity on real
// geometry.

#include "../common/types.slang"

[shader("compute")]
[numthreads(256, 1, 1)]
void resetOpacity(
    uint3 dtid: SV_DispatchThreadID,
    RWStructuredBuffer<Gaussian3D> gaussians,
    StructuredBuffer<uint> num_gaussians_counter,
    uniform float reset_value)
{
    uint i = dtid.x;
    if (i >= num_gaussians_counter[0]) return;
    gaussians[i].opacity = reset_value;
}
