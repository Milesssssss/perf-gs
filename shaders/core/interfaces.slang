// Core contracts for the generic differentiable blendable-primitive rasterizer.
//
// A "blend record" is the compact, per-pixel-evaluable form a primitive takes
// inside the tile blend loop. It is IDifferentiable so the backward rasterizer
// can flow per-pixel gradients into the primitive's stored fields via bwd_diff
// (validated on vulkan+d3d12 by the P0 spike, shaders/_spike/generic_spike.slang).
//
// Each primitive supplies only the two differentiable per-pixel quantities the
// blend loop needs: coverage() (its alpha / falloff) and color(). HOW those
// accumulate front-to-back — the extinction / compositing — is the Compositor's
// job (core/compositor.slang), kept orthogonal so either can vary on its own.

#ifndef TRAINER_CORE_INTERFACES_SLANG
#define TRAINER_CORE_INTERFACES_SLANG

interface IBlendable : IDifferentiable
{
    // Per-pixel coverage/alpha BEFORE the blend loop's ALPHA_MAX clamp and
    // ALPHA_MIN skip. For a 3D gaussian this is sigmoid(opacity) * comp *
    // exp(-0.5 * dᵀ Σ⁻¹ d); a 2D disk would return a smoothstep falloff.
    [Differentiable] float coverage(no_diff float2 pixel);

    // View-independent linear-RGB color the covered pixel receives. For a
    // gaussian this is the SH band-0 (DC) activation shDC(sh0).
    [Differentiable] float3 color();
}

#endif // TRAINER_CORE_INTERFACES_SLANG
