// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Importance-guided sparse backward sampling — the GPU half.
//
// Runs BETWEEN the (dense) image-space loss pass and the Gaussian backward, and
// leaves both untouched:
//
//   loss kernels     write h_i = dL/dC_i for the WHOLE image, mask off. That is
//                    the pre-existing dense path, byte-identical.
//   these kernels    build p_i from h_i, sample M_i = 1[U_i < p_i], rewrite
//                    dL_dC in place as M_i h_i / p_i, and emit a SCREEN-SIZED
//                    mask table in exactly the packed layout common/stbn.slang
//                    already reads.
//   backward         unchanged. It is handed the screen table with
//                    stbn_dims = padded frame dims and stbn_slice = 0, and its
//                    per-tile count already comes from the table (a global STBN
//                    threshold varies it too), so nothing there knows the
//                    density became nonuniform.
//
// That last point is the reason for the whole design: the existing 3-field
// uint32 layout carries a SET, not a ranking or a threshold, so a nonuniform
// marginal is expressible in it without a single change to the consumer.
//
// Math, and why each step is the way it is: see
// src/trainer/backend/render/adaptive_stbn.py, which is the host reference this
// file reimplements in float32 and which tests/test_adaptive_stbn.py pins it to.
//
// Dispatch order (adaptive_sampler.py drives it):
//   1  importanceReduce    full-res -> coarse    s = ||h||_2, box-averaged
//   2  importanceBlurX     coarse  -> coarse     separable Gaussian, clamped
//   3  importanceBlurY     coarse  -> coarse
//   4  importanceStats     coarse  -> partials   sum / min / max
//   5  importanceResolve   partials -> scalars   mean, bin range; clears state
//   6  importanceHistogram full-res              writes a_i, histograms it
//   7  solveLambda         histogram -> lambda   bisection on the budget
//   8  applyAdaptive       full-res              p, U, M, h/p, packed table
//
// Steps 2-5 and 7 touch only the coarse grid or a 257-entry histogram, so the
// per-iteration cost is dominated by the three full-resolution passes.

#include "../common/stbn.slang"
#include "../common/fixed_point.slang"

// ---------------------------------------------------------------------------
// Layouts shared with the host (adaptive_sampler.py mirrors every constant)
// ---------------------------------------------------------------------------

// `adaptive_scalars` slots. One small float buffer instead of five one-element
// buffers, so a stage reads its inputs from a line it already has.
static const uint ADAPTIVE_SCALAR_MEAN      = 0u;  // mean(s~) over coarse cells
static const uint ADAPTIVE_SCALAR_A_MIN     = 1u;  // min over the coarse a field
static const uint ADAPTIVE_SCALAR_A_MAX     = 2u;
static const uint ADAPTIVE_SCALAR_LAMBDA    = 3u;
static const uint ADAPTIVE_SCALAR_BIN_LO    = 4u;
static const uint ADAPTIVE_SCALAR_BIN_WIDTH = 5u;
static const uint ADAPTIVE_SCALAR_DEGENERATE = 6u; // 1 => a is constant
static const uint ADAPTIVE_SCALAR_COUNT     = 8u;

// Histogram of the FULL-RESOLUTION a field. Bin i covers
// [bin_lo + i*w, bin_lo + (i+1)*w) with w = (1/rho - bin_lo)/BINS; bin BINS is
// the overflow, a_i >= 1/rho, whose pixels saturate at EVERY lambda >= rho and
// therefore need no value, only a count.
//
// Counts alone would leave a bin-width quantization error in the budget (each
// unsaturated bin would contribute lambda * count * centre instead of
// lambda * sum). Carrying the SUM as well makes every fully-unsaturated bin
// exact, so the only residual error is the single bin the threshold falls
// inside: measured 1e-8 relative on the budget at 256 bins, against 3e-4 for a
// coarse-grid solve. The sums are fixed point because a float atomic add is not
// available and would not be deterministic anyway.
//
// The scale is 2^12, not something larger, because the per-GROUP partial is
// accumulated in a 32-bit groupshared counter: 64-bit atomics on GROUPSHARED
// memory need shaderSharedInt64Atomics / SM 6.6 and are not universally
// available, while 64-bit atomics on a device buffer (which the gradient
// accumulators already rely on) are. The bound is 256 pixels per group times the
// largest non-overflow a, which is 1/rho, so at the smallest usable rho (1/64)
// the worst partial is 256 * 64 * 2^12 = 6.9e7 — comfortably inside uint32,
// where 2^20 would have overflowed it. Quantization at 2^12 costs ~0.1 absolute
// on a sum of ~1e6, i.e. nothing the budget can notice.
static const uint ADAPTIVE_BINS = 256u;
static const float ADAPTIVE_HIST_SCALE = 4096.0;            // 2^12
static const float ADAPTIVE_HIST_INV_SCALE = 1.0 / ADAPTIVE_HIST_SCALE;

// Per-iteration diagnostics (section 4 of the design: mean/min/max p, saturated
// fraction, expected vs actual sample count). Accumulated with atomics during
// applyAdaptive and read back OUT of the hot path — never per iteration, or the
// host drain would dominate the timing this feature exists to improve.
static const uint ADAPTIVE_STAT_SUM_P     = 0u;   // fixed point, 2^30
static const uint ADAPTIVE_STAT_SELECTED  = 1u;
static const uint ADAPTIVE_STAT_SATURATED = 2u;
static const uint ADAPTIVE_STAT_LAMBDA    = 3u;   // fixed point, 2^30
static const uint ADAPTIVE_STAT_I64_STRIDE = 4u;
static const uint ADAPTIVE_STAT_MIN_P   = 0u;     // asuint(p), monotone for p>=0
static const uint ADAPTIVE_STAT_MAX_P   = 1u;
static const uint ADAPTIVE_STAT_PIXELS  = 2u;
static const uint ADAPTIVE_STAT_U32_STRIDE = 4u;
static const float ADAPTIVE_P_SCALE = 1073741824.0;         // 2^30

// Where U comes from. Mirrors UNIFORM_* / U_MODE_* in adaptive_stbn.py.
static const uint ADAPTIVE_U_TABLE = 0u;   // a scalar blue-noise volume
static const uint ADAPTIVE_U_HASH  = 1u;   // fresh IID per visit

static const uint ADAPTIVE_BISECTIONS = 60u;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Thomas Wang's 32-bit hash. adaptive_stbn.hash_uniform is the same function in
// numpy, so the `iid` regime's host diagnostics sample the same stream the
// trainer does rather than a look-alike from a different RNG.
uint adaptiveHash(uint key)
{
    key = (key ^ 61u) ^ (key >> 16u);
    key *= 9u;
    key = key ^ (key >> 4u);
    key *= 0x27D4EB2Du;
    key = key ^ (key >> 15u);
    return key;
}

// 24 mantissa bits, so the result is exactly representable and strictly < 1.
float adaptiveUnitHash(uint2 pixel, uint seed)
{
    uint key = (pixel.x * 0x9E3779B1u) ^ (pixel.y * 0x85EBCA77u) ^ seed;
    return float(adaptiveHash(key) >> 8u) * (1.0 / 16777216.0);
}

// a_i from the smoothed importance. A UNIFORM rescale of a is a no-op on p
// (lambda absorbs it exactly, since lambda is solved for the budget), so the
// only thing `mean` has to be is positive — its accuracy affects the reported
// diagnostics, never the sampling.
float adaptiveStrength(float smoothed, float mean, float alpha)
{
    return (1.0 - alpha) + alpha * smoothed / (mean + 1e-12);
}

// Bilinear reconstruction of the coarse field, half-pixel centred and edge
// clamped — the same convention as adaptive_stbn.bilinear_upsample. Done by
// hand rather than with a sampler so the host reference can match it bit for
// bit enough to compare, and so the coarse field can live in a plain buffer.
float adaptiveSampleCoarse(StructuredBuffer<float> coarse, uint2 coarse_dim,
                           uint2 pixel, uint downsample)
{
    if (downsample <= 1u)
        return coarse[min(pixel.y, coarse_dim.y - 1u) * coarse_dim.x
                      + min(pixel.x, coarse_dim.x - 1u)];
    float inv = 1.0 / float(downsample);
    float cx = clamp((float(pixel.x) + 0.5) * inv - 0.5, 0.0, float(coarse_dim.x - 1u));
    float cy = clamp((float(pixel.y) + 0.5) * inv - 0.5, 0.0, float(coarse_dim.y - 1u));
    uint x0 = uint(floor(cx));
    uint y0 = uint(floor(cy));
    uint x1 = min(x0 + 1u, coarse_dim.x - 1u);
    uint y1 = min(y0 + 1u, coarse_dim.y - 1u);
    float fx = cx - float(x0);
    float fy = cy - float(y0);
    float v00 = coarse[y0 * coarse_dim.x + x0];
    float v01 = coarse[y0 * coarse_dim.x + x1];
    float v10 = coarse[y1 * coarse_dim.x + x0];
    float v11 = coarse[y1 * coarse_dim.x + x1];
    return lerp(lerp(v00, v01, fx), lerp(v10, v11, fx), fy);
}

// ---------------------------------------------------------------------------
// 1. s_i = ||h_i||_2, box-averaged onto the coarse grid
// ---------------------------------------------------------------------------
//
// RGB only: dL_dC.a is dL/dA (alpha supervision), which is not part of the dC
// the rasterizer Jacobian acts on, so folding it in would change what the proxy
// proxies for.
//
// Edge cells average over their VALID pixels only. Dividing by downsample^2
// regardless would depress importance at the right/bottom edge purely because
// the image does not divide evenly there, and that deficit would survive into p.
[shader("compute")]
[numthreads(16, 16, 1)]
void importanceReduce(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> dL_dC,
    RWStructuredBuffer<float> coarse_out,
    uniform uint2 frame_dim,
    uniform uint2 coarse_dim,
    uniform uint downsample)
{
    uint2 cell = dtid.xy;
    if (cell.x >= coarse_dim.x || cell.y >= coarse_dim.y) return;

    uint x_begin = cell.x * downsample;
    uint y_begin = cell.y * downsample;
    uint x_end = min(x_begin + downsample, frame_dim.x);
    uint y_end = min(y_begin + downsample, frame_dim.y);

    float total = 0.0;
    uint count = 0u;
    for (uint y = y_begin; y < y_end; ++y)
    {
        for (uint x = x_begin; x < x_end; ++x)
        {
            total += length(dL_dC[uint2(x, y)].rgb);
            count += 1u;
        }
    }
    coarse_out[cell.y * coarse_dim.x + cell.x] = (count > 0u)
        ? total / float(count) : 0.0;
}

// ---------------------------------------------------------------------------
// 2/3. Separable Gaussian on the coarse grid
// ---------------------------------------------------------------------------
//
// Taps are evaluated in the loop rather than read from a table: at the sigmas
// this runs at (under 4 coarse cells) the tap count is small, the grid is tiny,
// and an exp() per tap is far cheaper than the plumbing for an uploaded kernel
// that would then have to be kept in sync with the host's.
//
// Boundaries are CLAMPED, matching adaptive_stbn._blur_axis. Zero padding would
// make a border region look unimportant because it is a border, which would then
// be a permanently lower p there — a systematic sampling deficit dressed up as
// an importance measurement.
float adaptiveBlurTap(StructuredBuffer<float> source, uint2 coarse_dim,
                      int2 base, int2 step, float sigma, int radius)
{
    if (radius <= 0)
        return source[uint(base.y) * coarse_dim.x + uint(base.x)];
    float inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);
    float total = 0.0;
    float weight_total = 0.0;
    for (int offset = -radius; offset <= radius; ++offset)
    {
        int2 at = base + step * offset;
        at.x = clamp(at.x, 0, int(coarse_dim.x) - 1);
        at.y = clamp(at.y, 0, int(coarse_dim.y) - 1);
        float weight = exp(-float(offset * offset) * inv_two_sigma_sq);
        total += weight * source[uint(at.y) * coarse_dim.x + uint(at.x)];
        weight_total += weight;
    }
    return total / weight_total;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void importanceBlurX(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<float> source,
    RWStructuredBuffer<float> destination,
    uniform uint2 coarse_dim,
    uniform float sigma,
    uniform int radius)
{
    uint2 cell = dtid.xy;
    if (cell.x >= coarse_dim.x || cell.y >= coarse_dim.y) return;
    destination[cell.y * coarse_dim.x + cell.x] = adaptiveBlurTap(
        source, coarse_dim, int2(cell), int2(1, 0), sigma, radius);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void importanceBlurY(
    uint3 dtid: SV_DispatchThreadID,
    StructuredBuffer<float> source,
    RWStructuredBuffer<float> destination,
    uniform uint2 coarse_dim,
    uniform float sigma,
    uniform int radius)
{
    uint2 cell = dtid.xy;
    if (cell.x >= coarse_dim.x || cell.y >= coarse_dim.y) return;
    destination[cell.y * coarse_dim.x + cell.x] = adaptiveBlurTap(
        source, coarse_dim, int2(cell), int2(0, 1), sigma, radius);
}

// ---------------------------------------------------------------------------
// 4/5. sum / min / max of the smoothed coarse field
// ---------------------------------------------------------------------------

static const uint ADAPTIVE_REDUCE_THREADS = 256u;

groupshared float gs_sum[ADAPTIVE_REDUCE_THREADS];
groupshared float gs_min[ADAPTIVE_REDUCE_THREADS];
groupshared float gs_max[ADAPTIVE_REDUCE_THREADS];

[shader("compute")]
[numthreads(256, 1, 1)]
void importanceStats(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<float> smoothed,
    RWStructuredBuffer<float> partials,        // 3 floats per group
    uniform uint cell_count,
    uniform uint group_count)
{
    uint tid = gtid.x;
    uint index = dtid.x;
    bool live = index < cell_count;
    float value = live ? smoothed[index] : 0.0;
    gs_sum[tid] = value;
    // Inactive lanes must not win a min/max, so seed them with the identity.
    gs_min[tid] = live ? value : 1e30;
    gs_max[tid] = live ? value : -1e30;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = ADAPTIVE_REDUCE_THREADS / 2u; stride > 0u; stride >>= 1u)
    {
        if (tid < stride)
        {
            gs_sum[tid] += gs_sum[tid + stride];
            gs_min[tid] = min(gs_min[tid], gs_min[tid + stride]);
            gs_max[tid] = max(gs_max[tid], gs_max[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u && gid.x < group_count)
    {
        partials[gid.x * 3u + 0u] = gs_sum[0];
        partials[gid.x * 3u + 1u] = gs_min[0];
        partials[gid.x * 3u + 2u] = gs_max[0];
    }
}

// Second (single-group) level: finishes the reduction, derives the histogram
// range, and CLEARS the histogram + this iteration's stats slot. Folding the
// clears in here rather than into their own dispatches is worth it: they are two
// more launches per training iteration otherwise, and this group is running
// alone anyway.
[shader("compute")]
[numthreads(256, 1, 1)]
void importanceResolve(
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<float> partials,
    RWStructuredBuffer<float> scalars,
    RWStructuredBuffer<uint> hist_count,
    RWStructuredBuffer<int64_t> hist_sum,
    RWStructuredBuffer<int64_t> stats_i64,
    RWStructuredBuffer<uint> stats_u32,
    uniform uint group_count,
    uniform uint cell_count,
    uniform uint pixel_count,
    uniform uint stats_slot,
    uniform float alpha,
    uniform float rho)
{
    uint tid = gtid.x;

    float sum = 0.0;
    float lo = 1e30;
    float hi = -1e30;
    for (uint i = tid; i < group_count; i += ADAPTIVE_REDUCE_THREADS)
    {
        sum += partials[i * 3u + 0u];
        lo = min(lo, partials[i * 3u + 1u]);
        hi = max(hi, partials[i * 3u + 2u]);
    }
    gs_sum[tid] = sum;
    gs_min[tid] = lo;
    gs_max[tid] = hi;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = ADAPTIVE_REDUCE_THREADS / 2u; stride > 0u; stride >>= 1u)
    {
        if (tid < stride)
        {
            gs_sum[tid] += gs_sum[tid + stride];
            gs_min[tid] = min(gs_min[tid], gs_min[tid + stride]);
            gs_max[tid] = max(gs_max[tid], gs_max[tid + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Clear the histogram and this slot's stats while the reduction settles.
    for (uint bin = tid; bin <= ADAPTIVE_BINS; bin += ADAPTIVE_REDUCE_THREADS)
    {
        hist_count[bin] = 0u;
        if (bin < ADAPTIVE_BINS)
            hist_sum[bin] = int64_t(0);
    }
    if (tid < ADAPTIVE_STAT_I64_STRIDE)
        stats_i64[stats_slot * ADAPTIVE_STAT_I64_STRIDE + tid] = int64_t(0);
    if (tid < ADAPTIVE_STAT_U32_STRIDE)
        stats_u32[stats_slot * ADAPTIVE_STAT_U32_STRIDE + tid] =
            (tid == ADAPTIVE_STAT_MIN_P) ? 0xFFFFFFFFu : 0u;

    if (tid == 0u)
    {
        float mean = (cell_count > 0u) ? gs_sum[0] / float(cell_count) : 0.0;
        float a_min = adaptiveStrength(gs_min[0], mean, alpha);
        float a_max = adaptiveStrength(gs_max[0], mean, alpha);
        // The full-resolution a field is a bilinear reconstruction of the coarse
        // one, i.e. a convex combination of coarse values, so it cannot leave
        // [a_min, a_max]. That is what makes these safe histogram bounds.
        //
        // The bin range stops at 1/rho rather than at a_max: any a >= 1/rho
        // saturates at every lambda >= rho, so those pixels go in one overflow
        // bin and need no resolution. Spending all 256 bins on [a_min, 1/rho]
        // instead of on a range a hot spot can stretch to 50 is what keeps the
        // quantization error at 1e-8 instead of 1e-5.
        float bin_hi = 1.0 / max(rho, 1e-6);
        float bin_lo = min(a_min, bin_hi);
        scalars[ADAPTIVE_SCALAR_MEAN] = mean;
        scalars[ADAPTIVE_SCALAR_A_MIN] = a_min;
        scalars[ADAPTIVE_SCALAR_A_MAX] = a_max;
        scalars[ADAPTIVE_SCALAR_BIN_LO] = bin_lo;
        scalars[ADAPTIVE_SCALAR_BIN_WIDTH] =
            max(bin_hi - bin_lo, 1e-12) / float(ADAPTIVE_BINS);
        // alpha == 0 makes a identically 1. Flagged so solveLambda can answer
        // analytically (lambda = rho/a) instead of bisecting: rho is a dyadic
        // k/64 here, so the analytic answer makes p_i EXACTLY k/64 and 1/p_i
        // exactly the TILE_AREA/k the uniform path applies. That equality is
        // what makes the alpha=0 arm a bitwise reproduction of the existing
        // uniform mask AND of its gradient, not merely a close one.
        scalars[ADAPTIVE_SCALAR_DEGENERATE] = (a_max - a_min <= 0.0) ? 1.0 : 0.0;
        stats_u32[stats_slot * ADAPTIVE_STAT_U32_STRIDE + ADAPTIVE_STAT_PIXELS] =
            pixel_count;
    }
}

// ---------------------------------------------------------------------------
// 6. a_i at full resolution, plus its histogram
// ---------------------------------------------------------------------------
//
// `a_texture` is materialized rather than recomputed inside applyAdaptive so
// that the histogram and the sampling see BIT-IDENTICAL values. If they could
// differ, the budget the solver enforces would not be the budget the mask
// realizes, and the difference would show up as an unexplained gap between the
// expected and actual sample counts.
groupshared uint gs_hist_count[ADAPTIVE_BINS + 1u];
groupshared uint gs_hist_sum[ADAPTIVE_BINS];    // 2^12 fixed point, see above

[shader("compute")]
[numthreads(256, 1, 1)]
void importanceHistogram(
    uint3 dtid: SV_DispatchThreadID,
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<float> smoothed,
    StructuredBuffer<float> scalars,
    RWTexture2D<float> a_texture,
    RWStructuredBuffer<uint> hist_count,
    RWStructuredBuffer<int64_t> hist_sum,
    uniform uint2 frame_dim,
    uniform uint2 coarse_dim,
    uniform uint downsample,
    uniform float alpha)
{
    uint tid = gtid.x;
    for (uint bin = tid; bin <= ADAPTIVE_BINS; bin += 256u)
    {
        gs_hist_count[bin] = 0u;
        if (bin < ADAPTIVE_BINS)
            gs_hist_sum[bin] = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint index = dtid.x;
    uint total = frame_dim.x * frame_dim.y;
    if (index < total)
    {
        uint2 pixel = uint2(index % frame_dim.x, index / frame_dim.x);
        float mean = scalars[ADAPTIVE_SCALAR_MEAN];
        float a = adaptiveStrength(
            adaptiveSampleCoarse(smoothed, coarse_dim, pixel, downsample),
            mean, alpha);
        a_texture[pixel] = a;

        float bin_lo = scalars[ADAPTIVE_SCALAR_BIN_LO];
        float bin_width = scalars[ADAPTIVE_SCALAR_BIN_WIDTH];
        int bin = int(floor((a - bin_lo) / bin_width));
        if (bin >= int(ADAPTIVE_BINS))
        {
            InterlockedAdd(gs_hist_count[ADAPTIVE_BINS], 1u);
        }
        else
        {
            uint slot = uint(max(bin, 0));
            InterlockedAdd(gs_hist_count[slot], 1u);
            // ROUNDED, not truncated. uint() floors, which biases every value
            // down by half a quantum; over the whole sum that is a systematic
            // -1/(2*SCALE) relative deficit in sum(a), which lambda then
            // compensates by inflating every p by the same 1.2e-4. Measured
            // exactly that before the +0.5 (mean p 0.250031 against a target of
            // 0.25) — small, but a systematic budget overshoot, not noise.
            InterlockedAdd(gs_hist_sum[slot], uint(a * ADAPTIVE_HIST_SCALE + 0.5));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // One global atomic per (group, non-empty bin) instead of one per pixel.
    for (uint bin = tid; bin <= ADAPTIVE_BINS; bin += 256u)
    {
        uint count = gs_hist_count[bin];
        if (count != 0u)
            InterlockedAdd(hist_count[bin], count);
        if (bin < ADAPTIVE_BINS)
        {
            uint partial = gs_hist_sum[bin];
            if (partial != 0u)
                InterlockedAdd(hist_sum[bin], int64_t(partial));
        }
    }
}

// ---------------------------------------------------------------------------
// 7. lambda: sum_i min(1, lambda a_i) = rho N
// ---------------------------------------------------------------------------
//
// The sum is continuous and non-decreasing in lambda whatever the field looks
// like, so bisection is unconditionally correct here — no shape assumption, which
// a Newton step on this piecewise-linear function would need. Bracketed by
// [0, 1/a_min]: zero at the bottom, and at the top every pixel has p=1 so the
// sum is N >= rho N. 60 steps resolve lambda to ~1e-17 of the bracket.
//
// Evaluated from the histogram, so each step is a 257-entry reduction rather
// than a pass over the image: 60 image passes per training iteration would cost
// more than the backward this whole feature is meant to shorten.
groupshared uint gs_bin_count[ADAPTIVE_BINS + 1u];
groupshared float gs_bin_sum[ADAPTIVE_BINS];
groupshared float gs_partial[ADAPTIVE_REDUCE_THREADS];

[shader("compute")]
[numthreads(256, 1, 1)]
void solveLambda(
    uint3 gtid: SV_GroupThreadID,
    StructuredBuffer<uint> hist_count,
    StructuredBuffer<int64_t> hist_sum,
    RWStructuredBuffer<float> scalars,
    RWStructuredBuffer<int64_t> stats_i64,
    uniform uint pixel_count,
    uniform uint stats_slot,
    uniform float rho)
{
    uint tid = gtid.x;
    for (uint bin = tid; bin <= ADAPTIVE_BINS; bin += 256u)
    {
        gs_bin_count[bin] = hist_count[bin];
        if (bin < ADAPTIVE_BINS)
            gs_bin_sum[bin] = float(hist_sum[bin]) * ADAPTIVE_HIST_INV_SCALE;
    }
    GroupMemoryBarrierWithGroupSync();

    float bin_lo = scalars[ADAPTIVE_SCALAR_BIN_LO];
    float bin_width = scalars[ADAPTIVE_SCALAR_BIN_WIDTH];
    float a_min = scalars[ADAPTIVE_SCALAR_A_MIN];
    float a_max = scalars[ADAPTIVE_SCALAR_A_MAX];
    float target = rho * float(pixel_count);

    float lo = 0.0;
    float hi = 1.0 / max(a_min, 1e-12);
    if (scalars[ADAPTIVE_SCALAR_DEGENERATE] > 0.5)
    {
        // Constant field: answer exactly, see importanceResolve.
        if (tid == 0u)
        {
            float lambda = rho / max(a_max, 1e-12);
            scalars[ADAPTIVE_SCALAR_LAMBDA] = lambda;
            stats_i64[stats_slot * ADAPTIVE_STAT_I64_STRIDE + ADAPTIVE_STAT_LAMBDA] =
                int64_t(lambda * ADAPTIVE_P_SCALE);
        }
        return;
    }

    for (uint step = 0u; step < ADAPTIVE_BISECTIONS; ++step)
    {
        float mid = 0.5 * (lo + hi);
        // sum over bins of (count if the bin saturates, else lambda * sum).
        // The overflow bin always saturates.
        float threshold = 1.0 / max(mid, 1e-30);
        float partial = (tid == 0u) ? float(gs_bin_count[ADAPTIVE_BINS]) : 0.0;
        for (uint bin = tid; bin < ADAPTIVE_BINS; bin += 256u)
        {
            float centre = bin_lo + (float(bin) + 0.5) * bin_width;
            partial += (centre >= threshold)
                ? float(gs_bin_count[bin])
                : mid * gs_bin_sum[bin];
        }
        gs_partial[tid] = partial;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = ADAPTIVE_REDUCE_THREADS / 2u; stride > 0u; stride >>= 1u)
        {
            if (tid < stride)
                gs_partial[tid] += gs_partial[tid + stride];
            GroupMemoryBarrierWithGroupSync();
        }
        if (gs_partial[0] < target) lo = 0.5 * (lo + hi);
        else                        hi = 0.5 * (lo + hi);
        GroupMemoryBarrierWithGroupSync();
    }

    if (tid == 0u)
    {
        float lambda = 0.5 * (lo + hi);
        scalars[ADAPTIVE_SCALAR_LAMBDA] = lambda;
        stats_i64[stats_slot * ADAPTIVE_STAT_I64_STRIDE + ADAPTIVE_STAT_LAMBDA] =
            int64_t(lambda * ADAPTIVE_P_SCALE);
    }
}

// ---------------------------------------------------------------------------
// 8. Sample, reweight, and emit the packed table
// ---------------------------------------------------------------------------
//
// One 64-thread group per SCREEN TILE, because the packed layout is per tile:
// slot, compacted list and count are all tile-local, so the group that decides
// the tile's 64 flags is exactly the group that can compact them.
//
// Note what this does NOT do: it does not give the tile a quota. Every pixel's
// decision is its own Bernoulli draw against its own p_i, so a tile's count is
// free to vary (which is already true of a global STBN threshold — the backward
// reads the count from the table for exactly this reason). Allocating k pixels
// per tile would be a different sampling process and would destroy both the
// marginal p_i and the spatial arrangement.
groupshared uint gs_flag[CONTRACT_TILE_AREA];
groupshared uint gs_scan[CONTRACT_TILE_AREA];
groupshared uint gs_compact[CONTRACT_TILE_AREA];
groupshared float gs_stat_sum_p[CONTRACT_TILE_AREA];
groupshared uint gs_stat_min[CONTRACT_TILE_AREA];
groupshared uint gs_stat_max[CONTRACT_TILE_AREA];
groupshared uint gs_stat_sat[CONTRACT_TILE_AREA];

[shader("compute")]
[numthreads(CONTRACT_TILE_AREA, 1, 1)]
void applyAdaptive(
    uint3 gid: SV_GroupID,
    uint3 gtid: SV_GroupThreadID,
    RWTexture2D<float4> dL_dC,
    RWTexture2D<float> a_texture,
    StructuredBuffer<float> scalars,
    // The scalar blue-noise volume as a UNIFORM in [0,1) — per-slice global
    // normalized ranks, see adaptive_stbn.uniform_volume. Bound (as a 1-element
    // dummy) even in the hash mode, because a declared parameter must be.
    StructuredBuffer<float> u_table,
    RWStructuredBuffer<uint> mask_table,
    RWStructuredBuffer<int64_t> stats_i64,
    RWStructuredBuffer<uint> stats_u32,
    uniform uint2 frame_dim,
    uniform uint2 num_tiles,
    uniform uint2 u_dims,        // (width, height) of one u_table slice
    uniform uint u_slice,
    uniform uint u_mode,
    uniform uint u_seed,
    uniform uint stats_slot,
    uniform float rho,
    // 1 = the proposed estimator, h/p (unbiased for the dense gradient).
    // 0 = the labelled diagnostic control: h/rho, i.e. sampled from p but NOT
    //     inverse-probability weighted, so it optimizes an error-weighted
    //     objective on purpose. The 1/rho keeps the overall gradient magnitude
    //     matched to the unbiased arm — without it the control would also change
    //     the effective learning rate and the meaning of the densifier's
    //     ABSOLUTE --grad-thresh, confounding the one thing it is asking about.
    uniform uint unbiased)
{
    uint tid = gtid.x;
    uint tile_idx = gid.x;
    uint tile_x = tile_idx % num_tiles.x;
    uint tile_y = tile_idx / num_tiles.x;
    if (tile_y >= num_tiles.y) return;

    uint2 pixel = uint2(tile_x * STBN_TILE_WIDTH + (tid % STBN_TILE_WIDTH),
                        tile_y * STBN_TILE_HEIGHT + (tid / STBN_TILE_WIDTH));
    bool valid = pixel.x < frame_dim.x && pixel.y < frame_dim.y;

    float lambda = scalars[ADAPTIVE_SCALAR_LAMBDA];
    float p = 0.0;
    bool selected = false;
    if (valid)
    {
        p = clamp(lambda * a_texture[pixel], 0.0, 1.0);
        float u = (u_mode == ADAPTIVE_U_TABLE)
            ? u_table[(u_slice * u_dims.y + (pixel.y % u_dims.y)) * u_dims.x
                      + (pixel.x % u_dims.x)]
            : adaptiveUnitHash(pixel, u_seed);
        // The ONLY place the uniform is used. It is not warped, offset, scaled or
        // ranked — the importance map moved the threshold, and that is the whole
        // mechanism.
        selected = u < p;
    }

    gs_flag[tid] = selected ? 1u : 0u;
    gs_stat_sum_p[tid] = valid ? p : 0.0;
    gs_stat_min[tid] = valid ? asuint(p) : 0xFFFFFFFFu;
    gs_stat_max[tid] = valid ? asuint(p) : 0u;
    gs_stat_sat[tid] = (valid && p >= 1.0) ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();

    // Inclusive Hillis-Steele scan over the 64 flags: slot = inclusive - 1, and
    // the last lane's inclusive total is the tile's count.
    gs_scan[tid] = gs_flag[tid];
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1u; offset < STBN_TILE_AREA; offset <<= 1u)
    {
        uint add = (tid >= offset) ? gs_scan[tid - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        gs_scan[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint count = gs_scan[STBN_TILE_AREA - 1u];
    uint slot = gs_scan[tid] - gs_flag[tid];       // exclusive prefix

    gs_compact[tid] = 0u;
    GroupMemoryBarrierWithGroupSync();
    if (selected)
        gs_compact[slot] = tid;
    GroupMemoryBarrierWithGroupSync();

    // Same packing as stbn.pack_selection: slot (0xFF = unselected) | the
    // slot-th selected pixel << 8 | the tile's count << 16. Slots at or past
    // `count` are never read by the backward, so 0 there is fine.
    //
    // blocks_x == num_tiles.x because the table is allocated at the tile-padded
    // frame size, so stbnBlockBase(tile origin, padded dims, 0) is tile_idx *
    // TILE_AREA. Written directly rather than through the helper to make that
    // identity explicit at the one place it matters.
    mask_table[tile_idx * STBN_TILE_AREA + tid] =
          (selected ? slot : STBN_SLOT_NONE)
        | (gs_compact[tid] << STBN_COMPACT_SHIFT)
        | (count << STBN_COUNT_SHIFT);

    // h'_i = M_i h_i / p_i. Unselected pixels must be WRITTEN zero, not skipped:
    // dL_dC is fully rewritten each iteration and the backward's licence to skip
    // a pixel is precisely that its gradient is already zero.
    if (valid)
    {
        if (selected)
        {
            float scale = (unbiased != 0u) ? (1.0 / max(p, 1e-12))
                                           : (1.0 / max(rho, 1e-12));
            dL_dC[pixel] = dL_dC[pixel] * scale;
        }
        else
        {
            dL_dC[pixel] = float4(0.0);
        }
    }

    // Diagnostics, reduced in the group so this costs 5 global atomics per tile
    // rather than per pixel.
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = STBN_TILE_AREA / 2u; stride > 0u; stride >>= 1u)
    {
        if (tid < stride)
        {
            gs_stat_sum_p[tid] += gs_stat_sum_p[tid + stride];
            gs_stat_min[tid] = min(gs_stat_min[tid], gs_stat_min[tid + stride]);
            gs_stat_max[tid] = max(gs_stat_max[tid], gs_stat_max[tid + stride]);
            gs_stat_sat[tid] += gs_stat_sat[tid + stride];
            gs_flag[tid] += gs_flag[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u)
    {
        uint i64_base = stats_slot * ADAPTIVE_STAT_I64_STRIDE;
        uint u32_base = stats_slot * ADAPTIVE_STAT_U32_STRIDE;
        InterlockedAdd(stats_i64[i64_base + ADAPTIVE_STAT_SUM_P],
                       int64_t(gs_stat_sum_p[0] * ADAPTIVE_P_SCALE));
        InterlockedAdd(stats_i64[i64_base + ADAPTIVE_STAT_SELECTED],
                       int64_t(gs_flag[0]));
        InterlockedAdd(stats_i64[i64_base + ADAPTIVE_STAT_SATURATED],
                       int64_t(gs_stat_sat[0]));
        // asuint of a non-negative float is monotone in the float, so an integer
        // min/max on the bit pattern is a float min/max — and 32-bit atomic
        // min/max are available where a 64-bit one may not be.
        InterlockedMin(stats_u32[u32_base + ADAPTIVE_STAT_MIN_P], gs_stat_min[0]);
        InterlockedMax(stats_u32[u32_base + ADAPTIVE_STAT_MAX_P], gs_stat_max[0]);
    }
}
