// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// The Gaussian's screen footprint (SnugBox AABB + AccuTile ellipse-tile crossing) as an IScreenFootprint,
// so the binning kernels (rasterize/tile_binned/tile_walk.slang) enumerate touched tiles generically.

#ifndef TRAINER_GAUSSIAN_FOOTPRINT_SLANG
#define TRAINER_GAUSSIAN_FOOTPRINT_SLANG

#include "../../core/interfaces.slang"
#include "snugbox.slang"
#include "accutile.slang"

struct GaussianFootprint : IScreenFootprint
{
    EllipseParams e;    // inverse-covariance a,b,c + threshold t
    float2 mean_2d;     // screen center (AccuTile translates strips about it)
    float2 aabb_min;    // SnugBox bbox (unclamped to the frame)
    float2 aabb_max;
    float x_d_args;     // x-offset where the ellipse's y-extrema occur
    float y_d_args;     // y-offset where the ellipse's x-extrema occur
    bool visible;       // SnugBox has extent and EllipseParams well-formed

    // Computes SnugBox, ellipse params and extremum offsets up front so the walk queries are pure reads.
    static GaussianFootprint make(float2 mean_2d, float3 conic, float cutoff_q)
    {
        GaussianFootprint fp;
        fp.mean_2d = mean_2d;

        bool box_ok = snugBox(mean_2d, conic, cutoff_q, fp.aabb_min, fp.aabb_max);

        EllipseParams e;
        bool ell_ok = computeEllipseParams(conic, cutoff_q, e);
        computeExtremumOffsets(e, fp.x_d_args, fp.y_d_args);
        fp.e = e;

        fp.visible = box_ok && ell_ok;
        return fp;
    }

    bool bounds(out float2 out_min, out float2 out_max)
    {
        out_min = aabb_min;
        out_max = aabb_max;
        return visible;
    }

    bool iterateRows(uint2 tile_min, uint2 tile_max)
    {
        return accuTileIterateRows(tile_min, tile_max);
    }

    bool rowTiles(float2 bbox_min, float2 bbox_max, uint ty,
                  uint tile_min_x, uint tile_max_x, out uint tx_lo, out uint tx_hi)
    {
        return accuTileRowTiles(e, mean_2d, bbox_min, bbox_max, y_d_args,
                                ty, tile_min_x, tile_max_x, tx_lo, tx_hi);
    }

    bool colTiles(float2 bbox_min, float2 bbox_max, uint tx,
                  uint tile_min_y, uint tile_max_y, out uint ty_lo, out uint ty_hi)
    {
        return accuTileColTiles(e, mean_2d, bbox_min, bbox_max, x_d_args,
                                tx, tile_min_y, tile_max_y, ty_lo, ty_hi);
    }
}

#endif // TRAINER_GAUSSIAN_FOOTPRINT_SLANG
