// Forward-pass transmittance/color checkpoints for the transposed
// (per-splat) backward.

#ifndef TRAINER_RASTER_CHECKPOINT_SLANG
#define TRAINER_RASTER_CHECKPOINT_SLANG

// Splats per bucket. Tied to a 32-lane wave: the transposed backward
// processes one bucket as one 32-lane warp, using a single WavePrefixProduct
// / WavePrefixSum across the bucket to reconstruct (T, C). P MUST equal the
// wave width (32) — the backward's `local_splat = bucket*P + lane` with
// lane in [0,32) only covers 32 splats per bucket, so P>32 would silently
// drop the rest. Shrinking the checkpoint buffer must be done by capping
// max_sort_pairs, NOT by raising P.
// MUST match _RASTER_BUCKET_SIZE in src/trainer/buffers.py.
#include "generated/execution_contract.slang"
static const uint RASTER_BUCKET_SIZE = CONTRACT_RASTER_BUCKET_SIZE;

// Pixels per tile = the per-bucket checkpoint stride (one (T,C) entry per pixel
// per bucket). = TILE_WIDTH*TILE_HEIGHT (non-square tiles allowed). Must stay in
// lockstep with buffers.py's raster_checkpoints allocation (max_buckets*TILE_AREA).
static const uint CHECKPOINT_TILE_AREA = CONTRACT_TILE_AREA;

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

// Flat index of the (global_bucket, pixel_in_tile) entry in the ragged
// checkpoint buffer. global_bucket = bucket_offsets[tile] + local_bucket.
uint checkpointIndex(uint global_bucket, uint pixel_in_tile)
{
    return global_bucket * CHECKPOINT_TILE_AREA + pixel_in_tile;
}

#endif // TRAINER_RASTER_CHECKPOINT_SLANG
