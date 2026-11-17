// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// The tiles a splat's 1/255 ellipse touches (SnugBox + AccuTile), walked in one fixed order so
// bin_count's count and bin_emit's nth tile always agree with the scanned pair offsets.

#ifndef TRAINER_SPLAT_TILES_SLANG
#define TRAINER_SPLAT_TILES_SLANG

#include "../../common/types.slang"
#include "../../primitives/gaussian/snugbox.slang"
#include "../../primitives/gaussian/accutile.slang"
#include "common/execution.slang"

struct SplatTiles
{
    EllipseParams e;
    float2 mean_2d;
    float2 bbox_min;    // SnugBox clamped to the frame
    float2 bbox_max;
    float x_d_args;     // x-offset where the ellipse's y-extrema occur
    float y_d_args;     // y-offset where the ellipse's x-extrema occur
    uint2 tile_min;
    uint2 tile_max;

    // False when the splat never reaches the 1/255 floor or its box misses the frame.
    static bool make(BlendData splat, float2 frame_dim, out SplatTiles st)
    {
        st.mean_2d = splat.mean;
        float cutoff_q = splatCutoffQ(splat.amplitude, splat.opacity);
        if (!snugBox(splat.mean, splat.conic, cutoff_q, st.bbox_min, st.bbox_max))
            return false;
        st.bbox_min = max(st.bbox_min, float2(0.0));
        st.bbox_max = min(st.bbox_max, frame_dim);
        if (st.bbox_min.x >= st.bbox_max.x || st.bbox_min.y >= st.bbox_max.y)
            return false;

        uint2 tile_dim = uint2(TILE_WIDTH, TILE_HEIGHT);
        uint2 num_tiles = ((uint2)frame_dim + tile_dim - 1u) / tile_dim;
        st.tile_min = uint2(st.bbox_min) / tile_dim;
        st.tile_max = min(uint2(st.bbox_max) / tile_dim, num_tiles - 1u);

        if (!computeEllipseParams(splat.conic, cutoff_q, st.e))
            return false;
        computeExtremumOffsets(st.e, st.x_d_args, st.y_d_args);
        return true;
    }

    // Walks the crossed tiles strip by strip along the shorter axis and returns how many there are,
    // stopping at the nth one (nth_tile = (tx, ty)) when nth is below that count.
    uint walk(uint nth, out uint2 nth_tile)
    {
        nth_tile = uint2(0u);
        uint count = 0u;
        if (accuTileIterateRows(tile_min, tile_max))
        {
            for (uint ty = tile_min.y; ty <= tile_max.y; ++ty)
            {
                uint lo, hi;
                if (!accuTileRowTiles(e, mean_2d, bbox_min, bbox_max, y_d_args,
                                      ty, tile_min.x, tile_max.x, lo, hi))
                    continue;
                uint n = hi - lo + 1u;
                if (nth - count < n)
                {
                    nth_tile = uint2(lo + nth - count, ty);
                    return count + n;
                }
                count += n;
            }
        }
        else
        {
            for (uint tx = tile_min.x; tx <= tile_max.x; ++tx)
            {
                uint lo, hi;
                if (!accuTileColTiles(e, mean_2d, bbox_min, bbox_max, x_d_args,
                                      tx, tile_min.y, tile_max.y, lo, hi))
                    continue;
                uint n = hi - lo + 1u;
                if (nth - count < n)
                {
                    nth_tile = uint2(tx, lo + nth - count);
                    return count + n;
                }
                count += n;
            }
        }
        return count;
    }
}

#endif // TRAINER_SPLAT_TILES_SLANG
