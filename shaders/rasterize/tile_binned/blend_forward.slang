#ifndef TRAINER_BLEND_FORWARD_SLANG
#define TRAINER_BLEND_FORWARD_SLANG

#include "../../core/interfaces.slang"
#include "../../core/compositor.slang"
#include "checkpoint.slang"
#include "generated/execution_contract.slang"

static const float BLEND_T_EARLY_BREAK = 1e-4;
static const float BLEND_ALPHA_MAX = 0.99;
static const float BLEND_ALPHA_MIN = 1.0 / 255.0;

// Shared front-to-back tile blend. Primitive-specific code supplies only an
// IPrimitiveSource; compositing, checkpoints, thresholds, and early-out are one
// production implementation monomorphized for every primitive.
void blendForward<S : IPrimitiveSource>(
    S src,
    uint t_start,
    uint primitive_count,
    no_diff float2 pixel_center,
    bool valid_pixel,
    uint pixel_in_tile,
    uint tile_bucket_base,
    StructuredBuffer<uint> sorted_primitive_ids,
    RWStructuredBuffer<uint2> raster_checkpoints,
    float extinction_hardness,
    out float3 color,
    out float transmittance,
    out uint processed)
{
    color = float3(0.0);
    transmittance = 1.0;
    processed = 0u;

    Compositor compositor;
    compositor.hardness = extinction_hardness;

    for (uint local_index = 0u; local_index < primitive_count; ++local_index)
    {
        if (valid_pixel && (local_index % RASTER_BUCKET_SIZE) == 0u)
        {
            uint bucket = tile_bucket_base + local_index / RASTER_BUCKET_SIZE;
            raster_checkpoints[checkpointIndex(bucket, pixel_in_tile)] =
                packCheckpoint(transmittance, color);
        }

        uint primitive_index = sorted_primitive_ids[t_start + local_index];
        let primitive = src.primitiveAt(primitive_index);
        float alpha = min(primitive.coverage(pixel_center), BLEND_ALPHA_MAX);
        if (alpha < BLEND_ALPHA_MIN)
            continue;

        color += primitive.color() * compositor.weight(alpha) * transmittance;
        transmittance *= compositor.transmittance(alpha);
        processed = local_index + 1u;
        if (transmittance < BLEND_T_EARLY_BREAK)
            break;
    }
}

#endif // TRAINER_BLEND_FORWARD_SLANG
