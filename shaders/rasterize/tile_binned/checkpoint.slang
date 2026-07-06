// Forward-pass transmittance/color checkpoints for the transposed
// (per-splat) backward.
//
// The transposed backward makes the SPLAT the unit of parallelism (a lane
// owns a splat, streams over the tile's 256 pixels). A splat-owning lane
// can't walk a pixel's front-to-back chain, so it can't recover the
// pre-splat transmittance T_i the way the per-pixel backward does. We fix
// that with checkpointing (à la Taming-3DGS / activation checkpointing):
// the forward records, once every RASTER_BUCKET_SIZE splats, the running
// (T, C_in_front) for each pixel. The backward reads the checkpoint at the
// start of a splat's bucket and re-blends only the ≤P-1 splats between the
// checkpoint and the splat — bounded recompute, all forward multiplies of
// (1-alpha) (no division by near-zero, so opaque splats need no clamp).
//
// Storage is a RAGGED (CSR) buffer: buckets are indexed by a per-tile
// bucket-offset prefix sum (bucket_offsets[tile] + local_bucket), so the
// buffer scales with actual work (~num_rendered / P) instead of a
// rectangular W*H*MAX_BUCKETS grid. Each bucket stores TILE_AREA (256)
// per-pixel entries; each entry is (T, R, G, B) packed as fp16 into a
// uint2 (8 bytes) — half precision is ample since T and RGB are bounded
// and we only re-multiply a handful of well-conditioned terms after a read.

#ifndef TRAINER_RASTER_CHECKPOINT_SLANG
#define TRAINER_RASTER_CHECKPOINT_SLANG

// Splats per bucket. Memory<->recompute dial: larger P = fewer checkpoints
// (less VRAM). The checkpoint buffer costs (256 pixels * 8 B) / P bytes per
// sorted splat-tile pair, so doubling P halves the buffer. The backward
// processes a bucket as P/32 serial 32-lane chunks (RASTER_BUCKET_CHUNKS),
// carrying the running (T, C) forward from the checkpoint across chunks, so
// P is decoupled from the 32-lane wave and can exceed 32.
// MUST match _RASTER_BUCKET_SIZE in src/trainer/buffers.py + pipeline.py.
// Must be a multiple of 32 (the wave width the backward assumes).
static const uint RASTER_BUCKET_SIZE = 128u;
static const uint RASTER_BUCKET_CHUNKS = RASTER_BUCKET_SIZE / 32u;  // 4

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
    return global_bucket * 256u + pixel_in_tile;
}

#endif // TRAINER_RASTER_CHECKPOINT_SLANG
