// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// AccuTile (Speedy-Splat §4.1.2): exact tile intersection with the 2D Gaussian ellipse,
// iterating the shorter side of the SnugBox tile rectangle and emitting only crossed tiles.

#ifndef TRAINER_GAUSSIAN_ACCUTILE_SLANG
#define TRAINER_GAUSSIAN_ACCUTILE_SLANG

#include "../../common/math.slang"
#include "eval.slang"
#include "common/execution.slang"

struct EllipseParams
{
    float a;  // Σ⁻¹[0][0]
    float b;  // Σ⁻¹[0][1]
    float c;  // Σ⁻¹[1][1]
    float t;  // splatCutoffQ
};

// Same ellipse snugBox bounds: conic is the packed inverse covariance, t the splat kernel's cutoff.
bool computeEllipseParams(float3 conic, float t, out EllipseParams e)
{
    e.a = conic.x;
    e.b = conic.y;
    e.c = conic.z;
    e.t = t;
    return e.t > 0.0;
}

// (x_min, x_max) of the ellipse at y_d = y - μy, or (+inf, -inf) when the row misses it.
float2 ellipseXAtY(EllipseParams e, float y_d)
{
    // a*x² + 2b*y*x + (c*y² - t) = 0
    float disc = (e.b * y_d) * (e.b * y_d) - e.a * (e.c * y_d * y_d - e.t);
    if (disc < 0.0)
    {
        return float2(1e30, -1e30);
    }
    float sq = sqrt(disc);
    float inv_a = 1.0 / max(e.a, 1e-20);
    float root1 = (-e.b * y_d - sq) * inv_a;
    float root2 = (-e.b * y_d + sq) * inv_a;
    return float2(min(root1, root2), max(root1, root2));
}

// Symmetric: given x_d, solve for y_d.
float2 ellipseYAtX(EllipseParams e, float x_d)
{
    float disc = (e.b * x_d) * (e.b * x_d) - e.c * (e.a * x_d * x_d - e.t);
    if (disc < 0.0)
    {
        return float2(1e30, -1e30);
    }
    float sq = sqrt(disc);
    float inv_c = 1.0 / max(e.c, 1e-20);
    float r1 = (-e.b * x_d - sq) * inv_c;
    float r2 = (-e.b * x_d + sq) * inv_c;
    return float2(min(r1, r2), max(r1, r2));
}

// Absolute (x_min, x_max) of the ellipse in a horizontal strip, clamped to the bbox; (+inf, -inf) on a miss.
float2 accuTileRowXRange(
    EllipseParams e,
    float mean_y,
    float bbox_min_x, float bbox_max_x,
    float y_d_args,             // y-offset where x-extrema occur, positive
    float2 mean_2d,
    float row_y_bot_abs, float row_y_top_abs)
{
    float yb = row_y_bot_abs - mean_2d.y;
    float yt = row_y_top_abs - mean_2d.y;

    // No y-radius test: SnugBox already clamped everything into the bbox.
    float2 x_at_bot = ellipseXAtY(e, yb);
    float2 x_at_top = ellipseXAtY(e, yt);

    // Union of the ellipse's x-range across the strip.
    float xmin_abs = mean_2d.x + min(x_at_bot.x, x_at_top.x);
    float xmax_abs = mean_2d.x + max(x_at_bot.y, x_at_top.y);

    // Paper insight: a strip containing an x-extremum (y_d = ±y_d_args) reaches the bbox x-edge.
    bool has_x_extremum_plus  = (yb <= +y_d_args) && (+y_d_args <= yt);
    bool has_x_extremum_minus = (yb <= -y_d_args) && (-y_d_args <= yt);
    if (has_x_extremum_plus || has_x_extremum_minus)
    {
        xmin_abs = min(xmin_abs, bbox_min_x);
        xmax_abs = max(xmax_abs, bbox_max_x);
    }

    // Degenerate: strip doesn't intersect the ellipse at all.
    if (xmin_abs > xmax_abs)
    {
        return float2(1e30, -1e30);
    }

    // Clamp for safety; the gaussian cannot exceed the bbox anyway.
    xmin_abs = max(xmin_abs, bbox_min_x);
    xmax_abs = min(xmax_abs, bbox_max_x);
    return float2(xmin_abs, xmax_abs);
}

// Same by symmetry for columns.
float2 accuTileColYRange(
    EllipseParams e,
    float mean_x,
    float bbox_min_y, float bbox_max_y,
    float x_d_args,
    float2 mean_2d,
    float col_x_left_abs, float col_x_right_abs)
{
    float xl = col_x_left_abs - mean_2d.x;
    float xr = col_x_right_abs - mean_2d.x;
    float2 y_at_left  = ellipseYAtX(e, xl);
    float2 y_at_right = ellipseYAtX(e, xr);
    float ymin_abs = mean_2d.y + min(y_at_left.x, y_at_right.x);
    float ymax_abs = mean_2d.y + max(y_at_left.y, y_at_right.y);

    bool has_ext_plus  = (xl <= +x_d_args) && (+x_d_args <= xr);
    bool has_ext_minus = (xl <= -x_d_args) && (-x_d_args <= xr);
    if (has_ext_plus || has_ext_minus)
    {
        ymin_abs = min(ymin_abs, bbox_min_y);
        ymax_abs = max(ymax_abs, bbox_max_y);
    }

    if (ymin_abs > ymax_abs)
    {
        return float2(1e30, -1e30);
    }
    ymin_abs = max(ymin_abs, bbox_min_y);
    ymax_abs = min(ymax_abs, bbox_max_y);
    return float2(ymin_abs, ymax_abs);
}

// Offsets x_d_args / y_d_args at which the ellipse extrema occur.
void computeExtremumOffsets(EllipseParams e, out float x_d_args, out float y_d_args)
{
    float ac_minus_bb = max(e.a * e.c - e.b * e.b, 1e-20);
    x_d_args = sqrt(max((e.b * e.b * e.t) / (ac_minus_bb * max(e.a, 1e-20)), 0.0));
    y_d_args = sqrt(max((e.b * e.b * e.t) / (ac_minus_bb * max(e.c, 1e-20)), 0.0));
}

// bin_count and bin_emit must enumerate identical tiles (else scan offsets misalign): both use these helpers.

// The rasterizer's tile size: a shared contract with binning, not a tuning knob.
static const uint ACCUTILE_TILE_W = TILE_WIDTH;
static const uint ACCUTILE_TILE_H = TILE_HEIGHT;

// True to iterate rows (loop ty, range tx), false for columns; the shorter axis means fewer strips.
bool accuTileIterateRows(uint2 tile_min, uint2 tile_max)
{
    uint nx = tile_max.x - tile_min.x + 1u;
    uint ny = tile_max.y - tile_min.y + 1u;
    return ny <= nx;
}

// Inclusive tx range the ellipse crosses within tile row `ty`; false if it misses the row.
bool accuTileRowTiles(
    EllipseParams e, float2 mean_2d,
    float2 bbox_min, float2 bbox_max, float y_d_args,
    uint ty, uint tile_min_x, uint tile_max_x,
    out uint tx_lo, out uint tx_hi)
{
    tx_lo = 0u; tx_hi = 0u;
    float row_y_bot = float(ty * ACCUTILE_TILE_H);
    float row_y_top = float((ty + 1u) * ACCUTILE_TILE_H);
    float2 xr = accuTileRowXRange(e, mean_2d.y, bbox_min.x, bbox_max.x,
                                  y_d_args, mean_2d, row_y_bot, row_y_top);
    if (xr.x > xr.y) return false;   // sentinel: no intersection this row
    tx_lo = max(uint(xr.x) / ACCUTILE_TILE_W, tile_min_x);
    tx_hi = min(uint(xr.y) / ACCUTILE_TILE_W, tile_max_x);
    return tx_lo <= tx_hi;
}

// Inclusive ty range the ellipse crosses within tile column `tx`.
bool accuTileColTiles(
    EllipseParams e, float2 mean_2d,
    float2 bbox_min, float2 bbox_max, float x_d_args,
    uint tx, uint tile_min_y, uint tile_max_y,
    out uint ty_lo, out uint ty_hi)
{
    ty_lo = 0u; ty_hi = 0u;
    float col_x_left  = float(tx * ACCUTILE_TILE_W);
    float col_x_right = float((tx + 1u) * ACCUTILE_TILE_W);
    float2 yr = accuTileColYRange(e, mean_2d.x, bbox_min.y, bbox_max.y,
                                  x_d_args, mean_2d, col_x_left, col_x_right);
    if (yr.x > yr.y) return false;
    ty_lo = max(uint(yr.x) / ACCUTILE_TILE_H, tile_min_y);
    ty_hi = min(uint(yr.y) / ACCUTILE_TILE_H, tile_max_y);
    return ty_lo <= ty_hi;
}

#endif // TRAINER_GAUSSIAN_ACCUTILE_SLANG
