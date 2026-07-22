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

// A differentiable, BLENDABLE PRIMITIVE: the compact per-pixel-evaluable form a
// primitive takes inside the tile blend loop. `: IDifferentiable` lets the
// backward flow per-pixel gradients into the primitive's stored fields via
// bwd_diff. A primitive supplies only the two per-pixel quantities the blend
// needs — coverage() and color(); HOW they composite front-to-back is the
// Compositor's orthogonal job (core/compositor.slang). This is THE concept the
// whole rasterizer is generic over — nothing here is Gaussian-specific (a 2D
// disk conforms the same way, see shaders/_spike/generic_spike.slang).
interface IBlendablePrimitive : IDifferentiable
{
    // Per-pixel coverage/alpha BEFORE the blend loop's ALPHA_MAX clamp and
    // ALPHA_MIN skip. For a 3D gaussian this is sigmoid(opacity) * comp *
    // exp(-0.5 * dᵀ Σ⁻¹ d); a 2D disk would return a smoothstep falloff.
    [Differentiable] float coverage(no_diff float2 pixel);

    // View-independent linear-RGB color the covered pixel receives. For a
    // gaussian this is the SH band-0 (DC) activation shDC(sh0).
    [Differentiable] float3 color();
}

// Free-function wrapper for a primitive's coverage so its REVERSE-MODE
// derivative is nameable via bwd_diff. bwd_diff of an interface METHOD can't be
// taken directly, but wrapping the call in a generic free function can — the
// validated pattern (shaders/_spike/generic_spike.slang sampleGeneric). The
// backward rasterizer seeds dL/dcoverage into the primitive's differential via
// bwd_diff(primitiveCoverage<P>). Monomorphized per primitive, so zero cost.
[Differentiable]
float primitiveCoverage<P : IBlendablePrimitive>(P prim, no_diff float2 pixel)
{
    return prim.coverage(pixel);
}

// A buffer-backed SOURCE of blendable primitives: given a primitive index,
// gather that primitive's compact record from whatever buffers the primitive
// keeps. The forward (and, later, backward) blend loop is written once, generic
// over this interface, so adding a primitive means only supplying a conformer +
// a thin entry point that constructs it. Monomorphized per source type (zero
// dynamic dispatch, validated by shaders/_spike/generic_spike.slang), so the
// split costs nothing at runtime.
interface IPrimitiveSource
{
    // The concrete primitive type this source yields (e.g. GaussianPrimitive).
    associatedtype Primitive : IBlendablePrimitive;

    // Gather + assemble the primitive record at `idx`. Not differentiable: the
    // storage load is a plain gather; the diff path runs through the returned
    // primitive's coverage()/color().
    Primitive primitiveAt(uint idx);
}

#endif // TRAINER_CORE_INTERFACES_SLANG
