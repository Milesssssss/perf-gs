// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// Forward-pass transmittance/color checkpoints for the transposed
// (per-splat) backward.

#ifndef TRAINER_RASTER_CHECKPOINT_SLANG
#define TRAINER_RASTER_CHECKPOINT_SLANG

// Splats per bucket P (RASTER_BUCKET_SIZE) must equal the 32-lane wave width: the backward maps
// local_splat = bucket*P + lane, so P > 32 silently drops splats. Shrink checkpoints via max_sort_pairs.
#include "common/execution.slang"

// Per-bucket checkpoint stride, one (T, C) entry per tile pixel; must match buffers.py's
// raster_checkpoints allocation (max_buckets * TILE_AREA).
static const uint CHECKPOINT_TILE_AREA = TILE_AREA;

// Pack (a, b) two floats into one uint as two fp16 halves (a low, b high).
uint packHalf2(float a, float b)
{
    return (f32tof16(a) & 0xFFFFu) | (f32tof16(b) << 16u);
}

float2 unpackHalf2(uint packed)
{
    return float2(f16tof32(packed & 0xFFFFu), f16tof32(packed >> 16u));
}

// A checkpoint entry is (T, R, G, B) -> uint2 (.x = pack(T,R), .y = pack(G,B)).
uint2 packCheckpoint(float T, float3 rgb)
{
    return uint2(packHalf2(T, rgb.x), packHalf2(rgb.y, rgb.z));
}

void unpackCheckpoint(uint2 e, out float T, out float3 rgb)
{
    float2 tr = unpackHalf2(e.x);
    float2 gb = unpackHalf2(e.y);
    T = tr.x;
    rgb = float3(tr.y, gb.x, gb.y);
}

// Ragged checkpoint buffer index; global_bucket = bucket_offsets[tile] + local_bucket.
uint checkpointIndex(uint global_bucket, uint pixel_in_tile)
{
    return global_bucket * CHECKPOINT_TILE_AREA + pixel_in_tile;
}

#endif // TRAINER_RASTER_CHECKPOINT_SLANG
