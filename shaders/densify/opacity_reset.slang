// Periodic opacity reset (3DGS paper trick): clamp every gaussian's
// opacity logit DOWNWARDS to logit(reset_value). Gaussians whose opacity
// is already below the cap keep their current (lower) value. After the
// next prune sweep, gaussians that were only contributing via opacity
// (not useful geometry) get dropped.
//
// Matches gsplat's ``reset_opa``: ``torch.clamp(p, max=logit(value))``.
// Overwriting unconditionally (as an earlier version did) is stricter
// and loses information about already-low-opacity gaussians that the
// optimizer had deliberately driven down.
//
// ``reset_value`` is a probability on the Python side; converted to
// logit here via log(p / (1 - p)).

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
    // Cap at numerical extremes to keep log finite; same rationale as the
    // probability→logit conversion in the .ply loader.
    float p = clamp(reset_value, 1e-5, 1.0 - 1e-5);
    float logit_cap = log(p / (1.0 - p));
    gaussians[i].opacity = min(gaussians[i].opacity, logit_cap);
}
