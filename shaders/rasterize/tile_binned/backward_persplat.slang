// Transposed (per-splat) backward rasterizer.
//
// The per-pixel backward (backward.slang) parallelizes over pixels and
// serializes the splat chain, so all 256 pixels touching a splat contend
// on that splat's fat gradient accumulator (~16 atomics each). This kernel
// transposes the loop nest: the SPLAT is the unit of parallelism. A lane
// owns a splat, streams over all 256 tile pixels accumulating its splat's
// gradient in registers, then issues ONE set of atomics for the whole
// tile. The contended per-pixel accumulator leaves the hot path.
//
// The lane can't walk a pixel's front-to-back chain to get the pre-splat
// transmittance T_i, so it reconstructs it from the forward checkpoints
// (checkpoint.slang): read the bucket's (T, C_in_front) checkpoint and
// re-blend the splats between the checkpoint and this lane's splat via a
// wave prefix-product of the per-splat transmittance step (T_step = 1-alpha)
// and prefix-sum of c*alpha*T. All forward multiplies of (1-alpha) — no
// division by near-zero.
//
// Layout: [numthreads(256)], one workgroup per tile = 8 warps of 32 lanes.
// Warp w processes bucket (base_bucket + w); lane L in that warp owns splat
// (base_bucket + w) * P + L. The bucket loop advances by 8 buckets/iter.
//
// Gradient math is EXACTLY equivalent to backward.slang; see the C_behind
// telescoping identity in the pixel loop below.

#include "../../common/types.slang"
#include "../../common/math.slang"
#include "../../primitives/gaussian/project.slang"
#include "../../primitives/gaussian/eval.slang"
#include "../../primitives/gaussian/blend.slang"
#include "../../core/compositor.slang"
#include "checkpoint.slang"

#include "generated/execution_contract.slang"
static const uint TILE_SIZE = CONTRACT_TILE_SIZE;
static const uint TILE_AREA = TILE_SIZE * TILE_SIZE;   // 256
static const uint NUM_WARPS = TILE_AREA / 32u;         // 8 (assumes wave=32)

static const float ALPHA_MAX = 0.99;
static const float ALPHA_MIN_TRACK = 1.0 / 255.0;

// Backward blend-scan mode (compile-time toggle). Both modes produce
// numerically-equivalent gradients; the difference is how the exclusive
// prefixes over the 32-lane bucket are computed.
//
//   0 = FUSED (default, current fast path).
//       ONE Hillis-Steele scan over the affine-map pair (T_step, dLdC_weight)
//       yields both exclusive prefixes T_prefix (transmittance product) and
//       dLdC_C_in_front_prefix (dLdC-weighted running color sum) simultaneously.
//       Single ~6-deep shuffle dependency chain. See blendPrefixExclusive.
//
//   1 = LEGACY (pre-fusion baseline; kept for A/B comparison).
//       Separate WavePrefixProduct(T_step) + three WavePrefixSum(contrib.xyz)
//       calls, i.e. FOUR back-to-back wave scans; the sum's input depends on
//       the product's output, so the ALU dependency chain is ~2x deeper (~10)
//       and Nsight flagged this as the short-scoreboard stall the fused path
//       was designed to eliminate.
//
// See commit 6be3f5e ("wave op speedup") for the fusion history.
#ifndef BACKWARD_BLEND_SCAN_MODE
#define BACKWARD_BLEND_SCAN_MODE 0
#endif

// Fused exclusive prefix scan of the alpha-blend LINEAR recurrence across a
// 32-lane bucket. Splat i advances the running state (T, S) by the affine map
//     T -> T * T_step_i,   S -> S + dLdC_weight_i * T
// where T_step = 1 - alpha is the per-splat transmittance factor (T_new =
// T_old * T_step, all multiplies, never divides) and dLdC_weight is the
// per-splat weight for the running sum (see call site for its definition).
// These maps compose associatively:
//     combine(earlier=(a1,b1), later=(a2,b2)) = (a1*a2, b1 + a1*b2)
// so ONE Hillis-Steele scan carrying the pair yields BOTH exclusive prefixes:
//     T_prefix        = prod_{j<i} T_step_j          (was WavePrefixProduct)
//     dLdC_C_in_front_prefix = sum_{j<i}  dLdC_weight_j * T_prefix_j
//                                                    (was the T_checkpoint-scaled WavePrefixSum)
// The old code ran WavePrefixProduct and then a SEQUENTIALLY-DEPENDENT
// WavePrefixSum (the sum's input needed the product's output T_i) — two
// back-to-back ~5-deep shuffle chains ≈ a 10-deep dependency chain, which is
// what Nsight flags as short-scoreboard (shuffle-latency) stalls on both
// lines. Fusing makes it ONE ~6-deep chain for the same shuffle count.
// All-multiply of T_step (never divides by it), so opaque splats (T_step->0)
// stay stable — identical numerical footing to the forward's (1-alpha) product.
void blendPrefixExclusive(float T_step, float dLdC_weight, uint lane,
                          out float T_prefix, out float dLdC_C_in_front_prefix)
{
    // Inclusive scan of the affine maps: lane i holds the composed map over
    // splats [0..i]. a = product coeff, b = weighted-sum coeff. Local names
    // (a,b) match the associativity formula in the docstring above.
    float a = T_step;
    float b = dLdC_weight;
    [unroll]
    for (uint d = 1u; d < RASTER_BUCKET_SIZE; d <<= 1u)
    {
        // Earlier segment sits d lanes below. Clamp the source in-range (its
        // value is unused when lane < d); all 32 lanes are active here (the
        // scan region is warp-uniform), so the reads are always valid.
        uint src = (lane >= d) ? (lane - d) : 0u;
        float a_lo = WaveReadLaneAt(a, src);
        float b_lo = WaveReadLaneAt(b, src);
        if (lane >= d)
        {
            b = b_lo + a_lo * b;   // b1 + a1*b2  (before overwriting a)
            a = a_lo * a;          // a1 * a2
        }
    }
    // Exclusive = the INCLUSIVE value of lane i-1; lane 0 gets the identity
    // map (product 1, sum 0).
    uint prev = (lane >= 1u) ? (lane - 1u) : 0u;
    float a_prev = WaveReadLaneAt(a, prev);
    float b_prev = WaveReadLaneAt(b, prev);
    T_prefix        = (lane >= 1u) ? a_prev : 1.0;
    dLdC_C_in_front_prefix = (lane >= 1u) ? b_prev : 0.0;
}

// Stochastic pixel subsampling (SGD over the tile's pixels). When > 1, each
// backward visits only 1/STRIDE of the tile's pixels, cutting the per-pixel
// eval + wave-scan work. To stay an UNBIASED gradient estimator:
//   - the visited subset alternates each iteration via `pixel_phase` (host
//     toggles 0..STRIDE-1), so every pixel contributes in expectation;
//   - the accumulated per-splat gradient is scaled by STRIDE to compensate
//     for summing only 1/STRIDE of the pixels (keeps the LR unchanged).
// STRIDE == 1 (default) is the exact full-pixel backward: the skip folds
// away and the scale is 1.0, so there is zero overhead. Only the per-PIXEL
// cost is reduced; the per-splat projection backward + atomics run once per
// splat regardless, so speedup is sub-linear in STRIDE.
//
// Compile-time overridable (like BACKWARD_BLEND_SCAN_MODE) so the host can
// sweep it via a shader define (create_device(defines=...)) without editing
// this file. Higher STRIDE also shrinks STAGED_PIXELS (below) -> less LDS ->
// higher occupancy.
#ifndef BACKWARD_PIXEL_STRIDE
#define BACKWARD_PIXEL_STRIDE 1u
#endif

// Per-pixel constants staged in LDS once (avoids re-reading global textures
// on every bucket). dL_dC (rgb) + dL_dA (a) + T_final + C_total (rgb) = 8
// floats/pixel * 256 = 2048 floats = 8 KB.
groupshared float4 gs_dLdC[TILE_AREA];   // rgb = dL_dC, a = dL_dA
groupshared float  gs_Tfinal[TILE_AREA];
groupshared float3 gs_Ctotal[TILE_AREA];
groupshared uint   gs_processed[TILE_AREA];
groupshared uint   gs_max_processed;     // deepest splat any pixel processed

// Per-bucket checkpoints staged in LDS. The pixel loop needs one warp-uniform
// checkpoint per pixel: raster_checkpoints[cp_bucket*256 + p]. Issued inline it
// was 256 broadcast global loads/bucket on the critical path -> LSU-bound. Each
// of the NUM_WARPS warps instead batch-loads its bucket's entries here, then
// reads from LDS.
//
// Only the STRIDE-subsampled pixels are ever read (the pixel loop `continue`s
// on the rest before touching this array), so we stage ONLY those: TILE_AREA /
// STRIDE entries per warp, compacted by k = (p - phase) / STRIDE. At STRIDE=4
// this is 64 not 256 -> 4 KB not 16 KB, which is what lifts the shared-memory
// occupancy cap. Folds to the full 256 at STRIDE=1 (phase=0, k=p).
static const uint STAGED_PIXELS = TILE_AREA / BACKWARD_PIXEL_STRIDE;
//static const uint STAGED_PIXELS = TILE_AREA / 1u;

groupshared uint2  gs_checkpoints[NUM_WARPS][STAGED_PIXELS]; //Transmitt, R,G and B are already compressed to fp16. The limiting factor in this kernl is the groupshared memory.
//on a 4080 we have ~100kb of gs mem. Each thread group is using ~25kb, meaning we can have 4 threagroups per SM, each TG is 256 threads == 8 warps. 8*4=32. The max warps in flight is 48 for a 4080, so
// we cannot go over an occupancy of 2/3. Nsight captures in fact show 50/60% warp occupancy.
// we most likely cannot compress the gs checkpoints more

[shader("compute")]
[numthreads(TILE_AREA, 1, 1)]
void backwardRasterizePerSplat(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    // Compact per-gaussian blend record (mean, cov2d, opacity, sh0), packed by
    // bin_count. Replaces the fat gaussians[]/projected_2d[] gather: the blend
    // reads splats in random (sorted) order, so gathering only the 36 B it uses
    // instead of 80 B roughly halves the bytes and lifts the L2 hit rate.
    StructuredBuffer<BlendData> blend_data,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,
    StructuredBuffer<uint> bucket_offsets,
    StructuredBuffer<uint2> raster_checkpoints,
    StructuredBuffer<CameraData> camera,
    RWTexture2D<float4> output_image,
    RWTexture2D<uint> gaussian_count_per_pixel,
    RWTexture2D<float4> dL_dC_texture,
    // 2D gradient seeds, summed over all tiles a gaussian touches. The 3D
    // projection backward (mean/scale/rot) is deferred to projection_backward.
    // slang, which runs it ONCE per gaussian on these sums (projection is
    // linear in the seed: sum_k J^T g_k == J^T sum_k g_k). Opacity/color are
    // 2D-level already, so they stay here as final grads.
    RWStructuredBuffer<int64_t> grad_mean2d_x,
    RWStructuredBuffer<int64_t> grad_mean2d_y,
    RWStructuredBuffer<int64_t> grad_cov2d_a,
    RWStructuredBuffer<int64_t> grad_cov2d_b,
    RWStructuredBuffer<int64_t> grad_cov2d_c,
    RWStructuredBuffer<int64_t> grad_comp,   // Mip: dL/d(opacity_compensation)
    RWStructuredBuffer<int64_t> grad_opacity,
    RWStructuredBuffer<int64_t> grad_color_r,
    RWStructuredBuffer<int64_t> grad_color_g,
    RWStructuredBuffer<int64_t> grad_color_b,
    RWStructuredBuffer<int64_t> grad_2d_x_iter,
    RWStructuredBuffer<int64_t> grad_2d_y_iter,
    // Per-iteration phase in [0, BACKWARD_PIXEL_STRIDE) selecting which pixel
    // subset this backward visits. Host toggles it each iter so the SGD is
    // unbiased. Ignored when BACKWARD_PIXEL_STRIDE == 1.
    uniform uint pixel_phase,
    // Compositor extinction hardness k (1 = alpha-over). See core/compositor.slang.
    uniform float extinction_hardness)
{
    uint tid = gtid.x;
    uint2 frame_dim_px = uint2(camera[0].image_size);
    uint2 num_tiles = (frame_dim_px + TILE_SIZE - 1) / TILE_SIZE;
    // 1D dispatch: one 256-thread workgroup per tile. gid.x is the LINEAR
    // tile index; recover 2D tile coords from it (do NOT assume a 2D grid
    // like the per-pixel backward does).
    uint tile_idx = gid.x;
    uint tile_x = tile_idx % num_tiles.x;
    uint tile_y = tile_idx / num_tiles.x;
    if (tile_y >= num_tiles.y) return;

    uint t_start = tile_start[tile_idx];
    uint t_end   = tile_end[tile_idx];
    uint tile_count = (t_end > t_start) ? (t_end - t_start) : 0u;
    if (tile_count == 0u) return;

    uint tile_bucket_base = bucket_offsets[tile_idx];

    float4x4 world_to_view = camera[0].view_matrix;
    float4x4 view_to_clip  = camera[0].proj_matrix;
    float2 frame_dim = camera[0].image_size;

    // --- Stage per-pixel constants into LDS once ---
    // tid indexes a pixel-in-tile [0,256). Its screen coords:
    uint tile_origin_x = tile_x * TILE_SIZE;
    uint tile_origin_y = tile_y * TILE_SIZE;
    uint2 my_pixel = uint2(tile_origin_x + (tid % TILE_SIZE),
                           tile_origin_y + (tid / TILE_SIZE));
    bool my_pixel_valid = (my_pixel.x < frame_dim_px.x && my_pixel.y < frame_dim_px.y);
    if (my_pixel_valid)
    {
        gs_dLdC[tid]      = dL_dC_texture[my_pixel];
        // T_final = 1 - alpha (forward wrote output_image.a = 1-T).
        float4 out_px     = output_image[my_pixel];
        gs_Tfinal[tid]    = 1.0 - out_px.a;
        gs_Ctotal[tid]    = out_px.rgb;
        gs_processed[tid] = gaussian_count_per_pixel[my_pixel];
    }
    else
    {
        gs_dLdC[tid]      = float4(0.0);
        gs_Tfinal[tid]    = 1.0;
        gs_Ctotal[tid]    = float3(0.0);
        gs_processed[tid] = 0u;   // no splat contributes to an OOB pixel
    }
    GroupMemoryBarrierWithGroupSync();

    // Deepest splat any pixel in this tile actually processed. Pixels
    // early-break (forward T < 1e-4), so max_processed <= tile_count and is
    // usually far smaller in dense tiles. Buckets beyond ceil(max/P) have
    // zero contributing pixels -> skip them entirely (saturation early-out).
    if (tid == 0u) gs_max_processed = 0u;
    GroupMemoryBarrierWithGroupSync();
    InterlockedMax(gs_max_processed, gs_processed[tid]);
    GroupMemoryBarrierWithGroupSync();
    uint max_processed = gs_max_processed;

    uint lane = WaveGetLaneIndex();          // splat within the bucket [0,32)
    uint warp_idx = tid / WaveGetLaneCount(); // which bucket (offset from base)
    // Truncate to the buckets some pixel actually reached (vs. tile_count).
    uint num_buckets = (max_processed + RASTER_BUCKET_SIZE - 1u) / RASTER_BUCKET_SIZE;

    //example: 96gs, 32 per bucket -> 3 buckets. warp_idx=0,1,2 valid; 3..7 skip.

    // --- Bucket loop: 8 warps handle 8 consecutive buckets per iteration ---
    [loop]
    for (uint base_bucket = 0u; base_bucket < num_buckets; base_bucket += NUM_WARPS)
    {
        uint bucket = base_bucket + warp_idx;
        bool bucket_valid = (bucket < num_buckets);

        // This lane's splat (global index within the tile's sorted list).
        uint local_splat = bucket * RASTER_BUCKET_SIZE + lane;
        bool splat_valid = bucket_valid && (local_splat < tile_count);

        // Load this lane's splat once (registers). Masked lanes hold junk
        // but never write output; they act as scan identities via alpha=0.
        // Random (sorted-order) gather, so AoS wins over SoA — but we gather
        // only the compact BlendData (mean/cov2d/opacity/sh0) this pass needs,
        // not the full 80 B of gaussians[]+projected_2d[]. g_idx still comes
        // from the cheap uint sorted_gaussian_ids gather (needed for the atomic
        // scatter targets below).
        uint g_idx = 0u;
        BlendData blend_only_data;
        if (splat_valid)
        {
            g_idx = sorted_gaussian_ids[t_start + local_splat];
            blend_only_data = blend_data[g_idx];
        }

        // Fat gradient accumulators for THIS lane's splat, summed over pixels.
        float2 acc_mean2d = float2(0.0);
        float3 acc_cov2d  = float3(0.0);
        float  acc_opacity = 0.0;
        float  acc_comp = 0.0;   // Mip: sum_pixels dL/d(opacity_compensation)
        float3 acc_color  = float3(0.0);
        float  acc_absx = 0.0;
        float  acc_absy = 0.0;

        // Build the primitive record once per lane via the SAME
        // IBlendablePrimitive conformer the forward uses (GaussianPrimitive), so
        // coverage()/color() + their bwd_diff define the gaussian's
        // differentiable behavior in ONE place. Masked lanes get a zero prim
        // (finite color()/coverage(); they're scan identities via alpha=0).
        GaussianPrimitive prim = splat_valid
            ? GaussianPrimitive.make(blend_only_data.mean, blend_only_data.cov2d,
                                     blend_only_data.opacity, blend_only_data.sh0,
                                     blend_only_data.opacity_compensation)
            : GaussianPrimitive.make(float2(0.0), float3(0.0), 0.0, float3(0.0), 0.0);
        float3 c_i = prim.color();

        uint cp_bucket = tile_bucket_base + bucket;
        // First splat index of this bucket. A pixel contributes to this
        // bucket only if its ray reached at least this splat
        // (processed[p] > bucket_first_splat); otherwise ALL 32 lanes are
        // masked for that pixel and it produces zero gradient. `bucket` is
        // warp-uniform, so this skip is warp-uniform -> the wave scans below
        // stay valid (all 32 lanes continue together).
        uint bucket_first_splat = bucket * RASTER_BUCKET_SIZE;

        // --- Coalesced batch-stage this warp's 256 checkpoints into LDS ---
        // The pixel loop reads one warp-uniform checkpoint per pixel. Issued
        // inline that was 256 global loads/bucket on the critical path (all 32
        // lanes hammering the same address -> LSU-issue-bound). Instead, the 32
        // lanes cooperatively load the bucket's 256 (T,C) entries in 8 fully
        // coalesced passes (lane L loads e = L, L+32, ...), then the pixel loop
        // reads from LDS. The bucket loop trip count is block-uniform, so these
        // group barriers are legal inside it.
        //   barrier #1 (WAR): don't overwrite entries the previous iteration's
        //     pixel loop may still be reading. Redundant but harmless on iter 0.
        //   barrier #2 (RAW): pixel-loop reads must see the staged writes. The
        //     read is cross-lane (lane L reads entry p written by lane p%32),
        //     so a warp-uniform read alone is not safe without this sync.
        GroupMemoryBarrierWithGroupSync();
        if (bucket_valid)
        {
            uint cp_base = checkpointIndex(cp_bucket, 0u);
            uint phase = pixel_phase % BACKWARD_PIXEL_STRIDE;  // 0 at STRIDE=1
            // Compact: LDS slot k holds pixel p = phase + k*STRIDE, i.e. exactly
            // the pixels the loop below visits. Global reads are STRIDE-strided
            // (mild coalescing loss) but this is once/bucket and shrinks LDS to
            // STAGED_PIXELS, which is the point.
            for (uint k = lane; k < STAGED_PIXELS; k += RASTER_BUCKET_SIZE)
                gs_checkpoints[warp_idx][k] = raster_checkpoints[cp_base + phase + k * BACKWARD_PIXEL_STRIDE];
        }
        GroupMemoryBarrierWithGroupSync();

        // --- Pixel loop: all 32 lanes lockstep so wave scans are valid ---
        [loop]
        for (uint p = 0u; p < TILE_AREA; ++p)
        {
            // Per-bucket saturation skip: if pixel p's ray terminated before
            // this bucket, no lane contributes -> skip the eval + wave ops
            // entirely. Warp-uniform (bucket + gs_processed[p] are uniform
            // across the 32 lanes), so this does not desync the wave scans.
            uint processed = gs_processed[p];
            if (processed <= bucket_first_splat)
                continue;

            // Stochastic pixel subsampling: visit only pixels whose index is
            // congruent to the current phase mod STRIDE. Depends solely on p
            // and pixel_phase (both warp-uniform), so all 32 lanes skip
            // together -> wave scans below stay valid. Folds away at STRIDE=1.
            if (BACKWARD_PIXEL_STRIDE > 1u &&
                (p % BACKWARD_PIXEL_STRIDE) != (pixel_phase % BACKWARD_PIXEL_STRIDE))
                continue;

            // Early-break mask: pixel p only contributes gradient to splats
            // it actually processed (local_splat < processed[p]). Masked
            // lanes contribute alpha=0 -> scan identity (T_step=1, cαT=0).
            bool contributes = splat_valid && (local_splat < processed);

            // Read the bucket checkpoint for pixel p (in-front-of-bucket
            // (T, C)) from LDS (batch-staged above). Only reached for visited
            // pixels (p % STRIDE == phase), which are compacted to slot
            // (p - phase) / STRIDE; folds to p at STRIDE=1. Only meaningful
            // if the pixel reached this bucket; masked lanes ignore.
            uint checkpoint_slot = (p - (pixel_phase % BACKWARD_PIXEL_STRIDE))
                                   / BACKWARD_PIXEL_STRIDE;
            float T_checkpoint; float3 C_checkpoint;
            unpackCheckpoint(gs_checkpoints[warp_idx][checkpoint_slot], T_checkpoint, C_checkpoint);

            // Pixel center for pixel-in-tile p (computed unconditionally so
            // it's in scope for the eval backward below).
            float2 pixel_center = float2(tile_origin_x + (p % TILE_SIZE),
                                         tile_origin_y + (p / TILE_SIZE)) + 0.5;

            float alpha = 0.0;
            float alpha_grad_scale = 0.0;
            if (contributes)
            {
                float alpha_raw = prim.coverage(pixel_center);
                alpha = min(alpha_raw, ALPHA_MAX);
                alpha_grad_scale = (alpha_raw <= ALPHA_MAX) ? 1.0 : 0.0;
                // Alpha-min skip — MUST be bit-identical to the forward
                // (forward.slang ALPHA_MIN = 1/255 == ALPHA_MIN_TRACK). A splat
                // the forward skipped (alpha < 1/255) did NOT blend or advance
                // T there, so here it must be a full no-op too: alpha=0 makes
                // T_step=1 and alpha*T=0 (scan identity) and zeroes its
                // gradient. Without this, the backward would blend a splat the
                // forward skipped -> T reconstruction diverges -> wrong grads.
                if (alpha < ALPHA_MIN_TRACK)
                {
                    alpha = 0.0;
                    alpha_grad_scale = 0.0;
                }
            }

            // Per-splat transmittance step: T advances by *= T_step across this
            // splat. Identity (1) when the lane is masked (alpha=0) so the scan
            // naturally skips masked lanes. hardness is a uniform: 1 = alpha-over
            // (bit-identical); k > 1 sharpens extinction (see core/compositor.slang).
            Compositor comp;
            comp.hardness = extinction_hardness;
            float T_step = comp.transmittance(alpha);

            // dL_dC (and C_checkpoint/C_total/T_final/dL_dA) are PER-PIXEL, hence
            // warp-uniform here: all 32 lanes share pixel p; lanes differ only
            // in which splat they own.
            float3 dL_dC = gs_dLdC[p].rgb;              // warp-uniform broadcast

            // Scan region: compute T_i (pre-splat transmittance) and either
            // scalar (fused) or vector (legacy) exclusive prefixes needed by
            // the dL_dalpha formula in the `if (contributes)` block below.
            // See BACKWARD_BLEND_SCAN_MODE define at the top for a description
            // of the two modes.
#if BACKWARD_BLEND_SCAN_MODE == 0
            // FUSED: one Hillis-Steele scan of the affine-map pair
            // (T_step, dLdC_weight). C_in_front and contrib are consumed ONLY
            // as dot(dL_dC, .) below (via C_behind -> dC_dalpha -> dL_dalpha),
            // so with warp-uniform dL_dC the dot commutes THROUGH the prefix
            // sum:
            //   dot(dL_dC, prefixsum(c_i*alpha*T_i)) ==
            //     prefixsum(dot(dL_dC,c_i)*alpha*T_i)
            // The fused scan returns:
            //   T_prefix        = prod_{j<i} T_step_j
            //                     -> T_i     = T_checkpoint * T_prefix
            //   dLdC_C_in_front_prefix = sum_{j<i}  dLdC_weight_j * T_prefix_j
            //                     -> dLdC(C_in_front_excl) = dot(dL_dC,C_checkpoint)
            //                                              + T_checkpoint * dLdC_C_in_front_prefix
            float dLdC_weight = dot(dL_dC, c_i) * comp.weight(alpha);   // 0 when masked
            float T_prefix, dLdC_C_in_front_prefix;
            blendPrefixExclusive(T_step, dLdC_weight, lane, T_prefix, dLdC_C_in_front_prefix);
            float T_i     = T_checkpoint * T_prefix;                   // pre-splat transmittance
            float alpha_T = comp.weight(alpha) * T_i;          // this splat's blend weight
            // This splat's contribution to dot(dL_dC, C_in_front_incl):
            //   dLdC_contrib = dot(dL_dC, c_i * alpha * T_i)
            //               = dot(dL_dC, c_i) * alpha_T
            //               = dLdC_weight * T_i.
            float dLdC_contrib = dLdC_weight * T_i;
            // dot(dL_dC, C_in_front_excl(i)) = dot(dL_dC, C_checkpoint) + prefixsum.
            float dLdC_C_in_front = dot(dL_dC, C_checkpoint) + T_checkpoint * dLdC_C_in_front_prefix;   // exclusive
#else
            // LEGACY: WavePrefixProduct(T_step) followed by three vector
            // WavePrefixSum(contrib.xyz). The sum's input depends on T_i (the
            // product's output) -> sequential ~2x-deeper shuffle chain.
            // Kept for A/B comparison against the fused path.
            float T_prefix = WavePrefixProduct(T_step);        // exclusive
            float T_i     = T_checkpoint * T_prefix;
            float alpha_T = comp.weight(alpha) * T_i;          // this splat's blend weight
            // Color contribution c*alpha*T for the prefix-sum of C_in_front.
            // Use the SAME luminance-free vector sum the forward used.
            float3 contrib = c_i * alpha_T;                    // 0 when masked
            // C_in_front(i) = C_checkpoint + sum of contrib for lower lanes.
            float3 C_in_front = C_checkpoint + float3(
                WavePrefixSum(contrib.x),
                WavePrefixSum(contrib.y),
                WavePrefixSum(contrib.z));
#endif

            if (contributes)
            {
                float  dL_dA = gs_dLdC[p].a;
                float  T_final = gs_Tfinal[p];
                float3 C_total = gs_Ctotal[p];

                // Derivative denominator for the dC/dα formula (same numeric
                // value as T_step; kept separate to document intent — one is
                // "the transmittance factor across this splat", the other is
                // "the 1-α in the dC/dα quotient").
                float oneMinusAlpha = 1.0 - alpha;

                // Color grad seed: dL_dc_i = dL_dC * (alpha * T_i). shDC is
                // affine (Jacobian = const SH_C0), so instead of calling
                // bwd_diff(shDC) per pixel we accumulate the seed here and
                // scale by SH_C0 once after the pixel loop. Exact, 256x fewer
                // autodiff calls per bucket.
                acc_color += dL_dC * alpha_T;

#if BACKWARD_BLEND_SCAN_MODE == 0
                // FUSED: use the scalar dLdC_C_in_front + dLdC_contrib produced by
                // the fused scan.
                //   dot(dL_dC, C_behind(i)), with
                //   C_behind = C_total - C_in_front(i) - c_i*alpha_i*T_i:
                float dLdC_Cbehind = dot(dL_dC, C_total) - dLdC_C_in_front - dLdC_contrib;
                // Per-splat dLoss/dcoverage; the compositor owns the alpha-over
                // vs hardened-extinction math (bit-identical at hardness=1).
                float dL_dalpha = comp.coverageGrad(alpha, dot(dL_dC, c_i), T_i,
                                                    dLdC_Cbehind, dL_dA, T_final);
#else
                // LEGACY: use the vector C_in_front + contrib produced by the two
                // separate scans.
                // C_behind(i) = C_total - C_in_front(i) - c_i*alpha_i*T_i
                //            = sum_{j>i, j<processed} c_j*alpha_j*T_j.
                float3 C_behind = C_total - C_in_front - contrib;
                // Route the coverage-gradient through the SAME Compositor as the
                // fused path so both scan modes honor extinction_hardness
                // (bit-identical at hardness=1: coverageGrad reduces to the
                // c_i*T_i - C_behind/(1-alpha) telescoping this replaced).
                float dL_dalpha = comp.coverageGrad(alpha, dot(dL_dC, c_i), T_i,
                                                    dot(dL_dC, C_behind), dL_dA, T_final);
#endif
                float dL_d_alpha_raw = dL_dalpha * alpha_grad_scale;

                var dp_prim = diffPair(prim);

                bwd_diff(primitiveCoverage<GaussianPrimitive>)(dp_prim, pixel_center, dL_d_alpha_raw);

                acc_mean2d += dp_prim.d.mean;
                acc_cov2d  += dp_prim.d.cov2d;
                acc_opacity += dp_prim.d.opacity;
                acc_comp   += dp_prim.d.opacity_compensation;
                // acc_color seed accumulated above (SH_C0 applied post-loop).

                if (alpha >= ALPHA_MIN_TRACK)
                {
                    acc_absx += abs(dp_prim.d.mean.x);
                    acc_absy += abs(dp_prim.d.mean.y);
                }
            }
        }

        if (splat_valid)
        {
            // Apply shDC's constant Jacobian once for the whole bucket: the
            // pixel loop accumulated the raw color-grad seed sum(dL_dC * alpha_T);
            // shDC = 0.5 + SH_C0*sh0, so d(loss)/d(sh0) = SH_C0 * that sum.
            // Replaces 256 per-pixel bwd_diff(shDC) calls. Exact.
            acc_color *= SH_C0;

            // SGD scale: we summed only 1/STRIDE of the pixels, so scale the
            // accumulated gradient up by STRIDE to make it an unbiased
            // estimate of the full-pixel sum (keeps the LR unchanged). All
            // six accumulators — including the densification-tracking abs
            // sums — are scaled together so downstream thresholds see the
            // same magnitude as the full backward. Folds to *1.0 at STRIDE=1.
            if (BACKWARD_PIXEL_STRIDE > 1u)
            {
                float s = float(BACKWARD_PIXEL_STRIDE);
                acc_mean2d *= s;
                acc_cov2d  *= s;
                acc_opacity *= s;
                acc_comp   *= s;
                acc_color  *= s;
                acc_absx *= s;
                acc_absy *= s;
            }

            // Emit the SUMMED 2D gradients (mean2d, cov2d) for this splat's
            // contribution to this tile. projection_backward.slang applies the
            // 3D projection backward once per gaussian on the full sum. The
            // heavy quaternion/cov Jacobian no longer runs K times per gaussian
            // in this hot loop.
            InterlockedAdd(grad_mean2d_x[g_idx], toFixed(acc_mean2d.x));
            InterlockedAdd(grad_mean2d_y[g_idx], toFixed(acc_mean2d.y));
            InterlockedAdd(grad_cov2d_a[g_idx], toFixed(acc_cov2d.x));
            InterlockedAdd(grad_cov2d_b[g_idx], toFixed(acc_cov2d.y));
            InterlockedAdd(grad_cov2d_c[g_idx], toFixed(acc_cov2d.z));
            InterlockedAdd(grad_comp[g_idx], toFixed(acc_comp));
            InterlockedAdd(grad_opacity[g_idx], toFixed(acc_opacity));
            InterlockedAdd(grad_color_r[g_idx], toFixed(acc_color.x));
            InterlockedAdd(grad_color_g[g_idx], toFixed(acc_color.y));
            InterlockedAdd(grad_color_b[g_idx], toFixed(acc_color.z));
            InterlockedAdd(grad_2d_x_iter[g_idx], toFixed(acc_absx));
            InterlockedAdd(grad_2d_y_iter[g_idx], toFixed(acc_absy));
        }

        // Warps advance to their next bucket; a barrier isn't needed because
        // each warp's scans are self-contained and gs_* staging is read-only
        // here. But different warps finish their pixel loops independently,
        // and the next iteration re-reads only read-only LDS -> safe.
    }
}


// ===========================================================================
// FLAT (warp-per-bucket) backward. Same gradient math as
// backwardRasterizePerSplat; only the work assignment differs.
//
// Layout: [numthreads(256)] = 8 INDEPENDENT warp-buckets per workgroup. Warp w
// (tid / wave) handles the flat live-bucket slot gid.x*8 + w; lane L owns splat
// local_bucket*32 + L. There is NO bucket loop (one bucket per warp) and NO
// tile-wide per-pixel LDS staging: the per-pixel constants (dL_dC / T_final /
// C_total / processed) are read INLINE from the textures as warp-uniform
// broadcasts. Dropping the ~8 KB tile-wide staging leaves only the per-warp
// checkpoint stage in LDS, which raises occupancy vs the per-tile kernel (which
// is LDS-bound at ~50-60%). The tradeoff: a pixel's constants are re-read once
// per live bucket it contributes to (no cross-bucket amortization), but those
// are cached, warp-uniform texture reads.
//
// The dispatch is INDIRECT over the compacted live-bucket list built by
// live_buckets.slang (tileMaxProcessed -> scan -> scatterLiveBuckets ->
// prepareBucketDispatchArgs). live_buckets[slot] = (tile, local_bucket), so
// there is no in-kernel tile search and saturated buckets are never launched.
// ===========================================================================
[shader("compute")]
[numthreads(TILE_AREA, 1, 1)]
void backwardRasterizePerBucket(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<BlendData> blend_data,
    StructuredBuffer<uint> tile_start,
    StructuredBuffer<uint> tile_end,
    StructuredBuffer<uint> sorted_gaussian_ids,
    StructuredBuffer<uint> bucket_offsets,
    StructuredBuffer<uint2> live_buckets,        // (tile, local_bucket) per slot
    StructuredBuffer<uint> live_bucket_total,    // [1]: number of live buckets
    StructuredBuffer<uint2> raster_checkpoints,
    StructuredBuffer<CameraData> camera,
    RWTexture2D<float4> output_image,
    RWTexture2D<uint> gaussian_count_per_pixel,
    RWTexture2D<float4> dL_dC_texture,
    RWStructuredBuffer<int64_t> grad_mean2d_x,
    RWStructuredBuffer<int64_t> grad_mean2d_y,
    RWStructuredBuffer<int64_t> grad_cov2d_a,
    RWStructuredBuffer<int64_t> grad_cov2d_b,
    RWStructuredBuffer<int64_t> grad_cov2d_c,
    RWStructuredBuffer<int64_t> grad_comp,
    RWStructuredBuffer<int64_t> grad_opacity,
    RWStructuredBuffer<int64_t> grad_color_r,
    RWStructuredBuffer<int64_t> grad_color_g,
    RWStructuredBuffer<int64_t> grad_color_b,
    RWStructuredBuffer<int64_t> grad_2d_x_iter,
    RWStructuredBuffer<int64_t> grad_2d_y_iter,
    uniform uint pixel_phase,
    uniform float extinction_hardness)
{
    uint tid = gtid.x;
    uint lane = WaveGetLaneIndex();            // splat within the bucket [0,32)
    uint warp_idx = tid / WaveGetLaneCount();  // which warp-bucket in the WG [0,8)

    // Global live-bucket slot for this warp. Guard the tail block's extra warps
    // (slot >= live_total): they must still reach the staging barrier below, so
    // we only *return* AFTER it.
    uint slot = gid.x * NUM_WARPS + warp_idx;
    uint live_total = live_bucket_total[0];
    bool bucket_valid = (slot < live_total);

    uint2 frame_dim_px = uint2(camera[0].image_size);
    uint2 grid_tiles = (frame_dim_px + TILE_SIZE - 1) / TILE_SIZE;

    uint tile_idx = 0u, local_bucket = 0u;
    uint t_start = 0u, tile_count = 0u;
    uint tile_origin_x = 0u, tile_origin_y = 0u;
    uint cp_bucket = 0u, bucket_first_splat = 0u;
    if (bucket_valid)
    {
        uint2 lb = live_buckets[slot];
        tile_idx = lb.x;
        local_bucket = lb.y;
        uint tile_x = tile_idx % grid_tiles.x;
        uint tile_y = tile_idx / grid_tiles.x;
        tile_origin_x = tile_x * TILE_SIZE;
        tile_origin_y = tile_y * TILE_SIZE;
        t_start = tile_start[tile_idx];
        uint t_end = tile_end[tile_idx];
        tile_count = (t_end > t_start) ? (t_end - t_start) : 0u;
        cp_bucket = bucket_offsets[tile_idx] + local_bucket;
        bucket_first_splat = local_bucket * RASTER_BUCKET_SIZE;
    }

    // --- Stage this warp's checkpoints into per-warp LDS (coalesced) ---
    uint phase = pixel_phase % BACKWARD_PIXEL_STRIDE;   // 0 at STRIDE=1
    if (bucket_valid)
    {
        uint cp_base = checkpointIndex(cp_bucket, 0u);
        for (uint k = lane; k < STAGED_PIXELS; k += RASTER_BUCKET_SIZE)
            gs_checkpoints[warp_idx][k] = raster_checkpoints[cp_base + phase + k * BACKWARD_PIXEL_STRIDE];
    }
    // RAW barrier: the pixel loop's cross-lane checkpoint reads must see the
    // staged writes. ALL 8 warps must reach it (incl. invalid ones), so the
    // slot >= live_total return happens AFTER, never before.
    GroupMemoryBarrierWithGroupSync();
    if (!bucket_valid) return;

    // This lane's splat (index within the tile's sorted list).
    uint local_splat = local_bucket * RASTER_BUCKET_SIZE + lane;
    bool splat_valid = (local_splat < tile_count);
    uint g_idx = 0u;
    BlendData blend_only_data;
    if (splat_valid)
    {
        g_idx = sorted_gaussian_ids[t_start + local_splat];
        blend_only_data = blend_data[g_idx];
    }

    float2 acc_mean2d = float2(0.0);
    float3 acc_cov2d  = float3(0.0);
    float  acc_opacity = 0.0;
    float  acc_comp = 0.0;
    float3 acc_color  = float3(0.0);
    float  acc_absx = 0.0;
    float  acc_absy = 0.0;

    // Build the primitive record once per lane via the SAME IBlendablePrimitive
    // conformer the forward uses (GaussianPrimitive), so coverage()/color() +
    // their bwd_diff define the gaussian's differentiable behavior in ONE place.
    // Masked lanes get a zero prim (finite results; scan identities via alpha=0).
    GaussianPrimitive prim = splat_valid ? GaussianPrimitive.make(blend_only_data.mean, blend_only_data.cov2d,
                                 blend_only_data.opacity, blend_only_data.sh0,
                                 blend_only_data.opacity_compensation) : GaussianPrimitive.make(float2(0.0), float3(0.0), 0.0, float3(0.0), 0.0);
    float3 c_i = prim.color();

    // --- Pixel loop: all 32 lanes lockstep so the wave scans stay valid ---
    [loop]
    for (uint p = 0u; p < TILE_AREA; ++p)
    {
        uint2 pixel = uint2(tile_origin_x + (p % TILE_SIZE),
                            tile_origin_y + (p / TILE_SIZE));
        bool pix_valid = (pixel.x < frame_dim_px.x && pixel.y < frame_dim_px.y);
        // Inline per-pixel processed depth (was gs_processed[p]). OOB -> 0.
        uint processed = pix_valid ? gaussian_count_per_pixel[pixel] : 0u;
        // Saturation skip: no lane contributes if the ray ended before this
        // bucket. Warp-uniform, so the wave scans stay in lockstep.
        if (processed <= bucket_first_splat)
            continue;

        // Stochastic pixel subsampling (folds away at STRIDE=1). Warp-uniform.
        if (BACKWARD_PIXEL_STRIDE > 1u && (p % BACKWARD_PIXEL_STRIDE) != phase)
            continue;

        bool contributes = splat_valid && (local_splat < processed);

        uint checkpoint_slot = (p - phase) / BACKWARD_PIXEL_STRIDE;
        float T_checkpoint; float3 C_checkpoint;
        unpackCheckpoint(gs_checkpoints[warp_idx][checkpoint_slot], T_checkpoint, C_checkpoint);

        float2 pixel_center = float2(pixel) + 0.5;

        // Inline per-pixel dL_dC (was gs_dLdC[p]). In-bounds here (processed>0
        // implies pix_valid). Warp-uniform broadcast.
        float4 dLdC4 = dL_dC_texture[pixel];
        float3 dL_dC = dLdC4.rgb;

        float alpha = 0.0;
        float alpha_grad_scale = 0.0;
        if (contributes)
        {
            float alpha_raw = prim.coverage(pixel_center);
            alpha = min(alpha_raw, ALPHA_MAX);
            alpha_grad_scale = (alpha_raw <= ALPHA_MAX) ? 1.0 : 0.0;
            if (alpha < ALPHA_MIN_TRACK)
            {
                alpha = 0.0;
                alpha_grad_scale = 0.0;
            }
        }

        // Transmittance factor via the Compositor. hardness is a uniform:
        // 1 = alpha-over (bit-identical); k > 1 sharpens extinction. Shared
        // with the forward + per-tile backward. See core/compositor.slang.
        Compositor comp;
        comp.hardness = extinction_hardness;
        float T_step = comp.transmittance(alpha);

#if BACKWARD_BLEND_SCAN_MODE == 0
        float dLdC_weight = dot(dL_dC, c_i) * comp.weight(alpha);
        float T_prefix, dLdC_C_in_front_prefix;
        blendPrefixExclusive(T_step, dLdC_weight, lane, T_prefix, dLdC_C_in_front_prefix);
        float T_i     = T_checkpoint * T_prefix;
        float alpha_T = comp.weight(alpha) * T_i;
        float dLdC_contrib = dLdC_weight * T_i;
        float dLdC_C_in_front = dot(dL_dC, C_checkpoint) + T_checkpoint * dLdC_C_in_front_prefix;
#else
        float T_prefix = WavePrefixProduct(T_step);
        float T_i     = T_checkpoint * T_prefix;
        float alpha_T = comp.weight(alpha) * T_i;
        float3 contrib = c_i * alpha_T;
        float3 C_in_front = C_checkpoint + float3(
            WavePrefixSum(contrib.x),
            WavePrefixSum(contrib.y),
            WavePrefixSum(contrib.z));
#endif

        if (contributes)
        {
            float  dL_dA = dLdC4.a;
            // Inline per-pixel T_final / C_total (was gs_Tfinal/gs_Ctotal).
            float4 out_px  = output_image[pixel];
            float  T_final = 1.0 - out_px.a;
            float3 C_total = out_px.rgb;
            float  oneMinusAlpha = 1.0 - alpha;

            acc_color += dL_dC * alpha_T;

#if BACKWARD_BLEND_SCAN_MODE == 0
            float dLdC_Cbehind = dot(dL_dC, C_total) - dLdC_C_in_front - dLdC_contrib;
            float dL_dalpha = comp.coverageGrad(alpha, dot(dL_dC, c_i), T_i, dLdC_Cbehind, dL_dA, T_final);
#else
            float3 C_behind = C_total - C_in_front - contrib; //remember that C_total = C_in_front + contrib_this_prim + C_behind, easy
            // Route the coverage-gradient through the SAME Compositor as the fused
            // path so both scan modes honor extinction_hardness (bit-identical at
            // hardness=1: coverageGrad reduces to c_i*T_i - C_behind/(1-alpha)).
            float dL_dalpha = comp.coverageGrad(alpha, dot(dL_dC, c_i), T_i, dot(dL_dC, C_behind), dL_dA, T_final);
#endif
            float dL_d_alpha_raw = dL_dalpha * alpha_grad_scale;

            var dp_prim = diffPair(prim);

            bwd_diff(primitiveCoverage<GaussianPrimitive>)(dp_prim, pixel_center, dL_d_alpha_raw);

            acc_mean2d += dp_prim.d.mean;
            acc_cov2d  += dp_prim.d.cov2d;
            acc_opacity += dp_prim.d.opacity;
            acc_comp   += dp_prim.d.opacity_compensation;

            if (alpha >= ALPHA_MIN_TRACK)
            {
                acc_absx += abs(dp_prim.d.mean.x);
                acc_absy += abs(dp_prim.d.mean.y);
            }
        }
    }

    // --- Per-splat projection seed + atomics (one lane per splat) ---
    if (splat_valid)
    {
        //rendered color is produced by taking the sh0 coeff, multiplying it by SH_C0 and adding 0.5. So the grad wrt to
        //Loss is SH_C0*dL_dC_i, where dL_dC_i is the accumulated seed sum(dL_dC * alpha_T) from the pixel loop above.
        acc_color *= SH_C0;
        
        if (BACKWARD_PIXEL_STRIDE > 1u)
        {
            float s = float(BACKWARD_PIXEL_STRIDE);
            acc_mean2d *= s;
            acc_cov2d  *= s;
            acc_opacity *= s;
            acc_comp   *= s;
            acc_color  *= s;
            acc_absx *= s;
            acc_absy *= s;
        }
        InterlockedAdd(grad_mean2d_x[g_idx], toFixed(acc_mean2d.x));
        InterlockedAdd(grad_mean2d_y[g_idx], toFixed(acc_mean2d.y));
        InterlockedAdd(grad_cov2d_a[g_idx], toFixed(acc_cov2d.x));
        InterlockedAdd(grad_cov2d_b[g_idx], toFixed(acc_cov2d.y));
        InterlockedAdd(grad_cov2d_c[g_idx], toFixed(acc_cov2d.z));
        InterlockedAdd(grad_comp[g_idx], toFixed(acc_comp));
        InterlockedAdd(grad_opacity[g_idx], toFixed(acc_opacity));
        InterlockedAdd(grad_color_r[g_idx], toFixed(acc_color.x));
        InterlockedAdd(grad_color_g[g_idx], toFixed(acc_color.y));
        InterlockedAdd(grad_color_b[g_idx], toFixed(acc_color.z));
        InterlockedAdd(grad_2d_x_iter[g_idx], toFixed(acc_absx));
        InterlockedAdd(grad_2d_y_iter[g_idx], toFixed(acc_absy));
    }
}
