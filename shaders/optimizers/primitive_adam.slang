// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_OPTIMIZER_PRIMITIVE_ADAM_SLANG
#define TRAINER_OPTIMIZER_PRIMITIVE_ADAM_SLANG

#include "../common/fixed_point.slang"

// Generic Adam over an arbitrary primitive's flat scalar parameter vector.
//
// A primitive exposes its optimizable scalars as a flat PrimitiveValues<N>
// (packed via ChannelCursor) behind IPrimitiveParameterStorage<N>; its Adam
// state lives in channel-major PLANES (AdamPlanes<N>) indexed
// [channel * stride + primitive]. AdamPass.step is the ONE monomorphized
// update loop — swapping the primitive only means supplying a new storage
// conformer, never touching the Adam math.

// --- Flat parameter vector -------------------------------------------------

// The primitive's optimizable scalars, laid out contiguously. `data[c]` is the
// value of channel c; a primitive packs/unpacks it through ChannelCursor so the
// per-field layout is declared in exactly one place.
struct PrimitiveValues<let N : int>
{
    float data[N];
}

// Sequential reader/writer over a PrimitiveValues<N> data array. `offset`
// advances as fields are consumed, so a primitive lists its fields once
// (write3 mean, write4 quat, ...) and the channel indices follow structurally.
struct ChannelCursor<let N : int>
{
    uint offset;

    [mutating] void write1(inout float data[N], float v)
    {
        data[offset] = v;
        offset += 1u;
    }
    [mutating] void write2(inout float data[N], float2 v)
    {
        data[offset + 0u] = v.x;
        data[offset + 1u] = v.y;
        offset += 2u;
    }
    [mutating] void write3(inout float data[N], float3 v)
    {
        data[offset + 0u] = v.x;
        data[offset + 1u] = v.y;
        data[offset + 2u] = v.z;
        offset += 3u;
    }
    [mutating] void write4(inout float data[N], float4 v)
    {
        data[offset + 0u] = v.x;
        data[offset + 1u] = v.y;
        data[offset + 2u] = v.z;
        data[offset + 3u] = v.w;
        offset += 4u;
    }

    [mutating] float read1(inout float data[N])
    {
        float v = data[offset];
        offset += 1u;
        return v;
    }
    [mutating] float2 read2(inout float data[N])
    {
        float2 v = float2(data[offset + 0u], data[offset + 1u]);
        offset += 2u;
        return v;
    }
    [mutating] float3 read3(inout float data[N])
    {
        float3 v = float3(data[offset + 0u], data[offset + 1u], data[offset + 2u]);
        offset += 3u;
        return v;
    }
    [mutating] float4 read4(inout float data[N])
    {
        float4 v = float4(data[offset + 0u], data[offset + 1u],
                          data[offset + 2u], data[offset + 3u]);
        offset += 4u;
        return v;
    }
}

// The parameter STORAGE seam: map a primitive index <-> its flat value vector.
// The conformer owns the concrete buffer(s) (e.g. RWStructuredBuffer<Gaussian3D>)
// and preserves any non-optimized fields across a load/store round-trip.
interface IPrimitiveParameterStorage<let N : int>
{
    PrimitiveValues<N> loadValues(uint index);
    void storeValues(uint index, PrimitiveValues<N> values);
}

// --- Optimizer state + config ---------------------------------------------

// Adam hyperparameters shared by every channel. The bias-correction
// denominators are precomputed on the host (they depend only on the step count).
struct AdamConfig
{
    float bias_correction1;
    float bias_correction2;
}

// Per-channel learning rates, supplied as a uniform. The host expands a
// primitive's grouped LRs (e.g. one LR per gaussian field) into this flat
// per-channel vector so the generic step stays primitive-agnostic.
struct ChannelLearningRates<let N : int>
{
    float rates[N];
}

// Channel-major Adam state PLANES: gradients (int64 fixed-point, atomically
// summed by the backward pass), first and second moments, all indexed
// [channel * channel_stride + primitive_index]. One buffer per quantity keeps
// each channel's column contiguous across primitives.
struct AdamPlanes<let N : int>
{
    RWStructuredBuffer<int64_t> gradients;
    RWStructuredBuffer<float>   moment1;
    RWStructuredBuffer<float>   moment2;
    uint channel_stride;

    uint address(uint channel, uint index)
    {
        return channel * channel_stride + index;
    }
    float gradient(uint channel, uint index, float inv_pixels)
    {
        return fromFixed(gradients[address(channel, index)]) * inv_pixels;
    }
}

// --- The Adam update -------------------------------------------------------

static const float ADAM_BETA1   = 0.9;
static const float ADAM_BETA2   = 0.999;
static const float ADAM_EPSILON = 1e-8;

// One monomorphized Adam step for one primitive, generic over its parameter
// storage. Reads the flat values + per-channel gradients, runs the standard
// Adam recurrence per channel, writes the values back and clears the gradients.
// A single NaN/Inf in any channel skips the whole primitive.
struct AdamPass
{
    static void step<let N : int, P : IPrimitiveParameterStorage<N>>(
        P storage, AdamPlanes<N> planes, ChannelLearningRates<N> lrs,
        uint index, AdamConfig cfg, float inv_pixels)
    {
        // Guard first: bail before touching any moment if a gradient is
        // non-finite (matches the reference kernel's all-or-nothing skip).
        [unroll]
        for (uint c = 0u; c < uint(N); ++c)
        {
            float g = planes.gradient(c, index, inv_pixels);
            if (isnan(g) || isinf(g)) return;
        }

        PrimitiveValues<N> values = storage.loadValues(index);

        [unroll]
        for (uint c = 0u; c < uint(N); ++c)
        {
            uint addr = planes.address(c, index);
            float g = fromFixed(planes.gradients[addr]) * inv_pixels;

            float m = planes.moment1[addr];
            float v = planes.moment2[addr];
            m = ADAM_BETA1 * m + (1.0 - ADAM_BETA1) * g;
            v = ADAM_BETA2 * v + (1.0 - ADAM_BETA2) * g * g;

            float m_hat = m / cfg.bias_correction1;
            float v_hat = v / cfg.bias_correction2;
            values.data[c] = values.data[c]
                           - lrs.rates[c] * m_hat / (sqrt(v_hat) + ADAM_EPSILON);

            planes.moment1[addr] = m;
            planes.moment2[addr] = v;
            planes.gradients[addr] = int64_t(0);
        }

        storage.storeValues(index, values);
    }
}

#endif // TRAINER_OPTIMIZER_PRIMITIVE_ADAM_SLANG
