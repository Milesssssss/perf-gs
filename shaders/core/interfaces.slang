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

#include "../common/types.slang"   // CameraData (projection contract)

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

// The STORED, optimizable geometry of a primitive that knows how to PROJECT
// itself to screen — the "3D model" side (IBlendablePrimitive is the projected
// "2D" side). `: IDifferentiable` lets the projection backward flow dL/d(screen)
// into the stored params via bwd_diff. A gaussian's model is mean + log_scale +
// rot_quat; the projected mean2d/cov2d/comp receive gradient seeds, while
// depth/det are forward-only (binning) and get zero seeds.
interface IProjectableModel : IDifferentiable
{
    [Differentiable]
    void projectInto(no_diff CameraData cam,
                     no_diff float filter_3D, no_diff float kernel_size, no_diff uint mip_enabled,
                     out float2 mean2d, out float3 cov2d, out float depth_vs,
                     out float cov2d_det, out float opacity_compensation);
}

// Free-function wrapper so the projection's REVERSE derivative is nameable via
// bwd_diff (interface methods aren't; same pattern as primitiveCoverage). The
// projection backward seeds the out-param differentials and reads the model's
// differential (dp_model.d.{mean,log_scale,rot_quat}). Validated by
// shaders/_spike/proj_spike.slang.
[Differentiable]
void projectModel<M : IProjectableModel>(
    M model, no_diff CameraData cam,
    no_diff float filter_3D, no_diff float kernel_size, no_diff uint mip_enabled,
    out float2 mean2d, out float3 cov2d, out float depth_vs,
    out float cov2d_det, out float opacity_compensation)
{
    model.projectInto(cam, filter_3D, kernel_size, mip_enabled,
                      mean2d, cov2d, depth_vs, cov2d_det, opacity_compensation);
}

// The projected primitive's SCREEN FOOTPRINT: the tile-binning contract. Given
// a projected primitive, it answers which tiles the primitive touches so the
// generic count/emit walks (rasterize/tile_binned/tile_walk.slang) can size the
// sort and place one (tile, depth, primitive) pair per touched tile. NON-
// differentiable — this is the integer tile walk, orthogonal to the per-pixel
// blend. A gaussian conforms with a SnugBox AABB + AccuTile ellipse crossing
// (primitives/gaussian/footprint.slang); a 2D disk would supply a circle bbox +
// circle-tile crossing with no change to the walks or the binning kernels.
interface IScreenFootprint
{
    // Unclamped screen-space AABB of the footprint. Returns false when the
    // primitive has no extent (fully culled); out params are then undefined.
    bool bounds(out float2 out_min, out float2 out_max);

    // Pick the walk axis: true iterates ROWS (per-row column spans via
    // rowTiles), false iterates COLUMNS (per-column row spans via colTiles).
    // Choosing the shorter axis keeps the per-strip crossing tests minimal.
    bool iterateRows(uint2 tile_min, uint2 tile_max);

    // For tile row `ty`, the inclusive column range [tx_lo, tx_hi] the footprint
    // actually crosses within [tile_min_x, tile_max_x]. False if the row is
    // untouched (skip it).
    bool rowTiles(float2 bbox_min, float2 bbox_max, uint ty,
                  uint tile_min_x, uint tile_max_x, out uint tx_lo, out uint tx_hi);

    // Column twin of rowTiles: the inclusive row range [ty_lo, ty_hi] crossed in
    // tile column `tx` within [tile_min_y, tile_max_y]. False if untouched.
    bool colTiles(float2 bbox_min, float2 bbox_max, uint tx,
                  uint tile_min_y, uint tile_max_y, out uint ty_lo, out uint ty_hi);
}

#endif // TRAINER_CORE_INTERFACES_SLANG
