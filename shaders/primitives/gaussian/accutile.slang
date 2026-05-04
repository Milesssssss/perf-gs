// AccuTile (Speedy-Splat §4.1.2): exact tile intersection with the 2D
// Gaussian ellipse. Refines SnugBox by iterating along the shorter side
// of the snug bbox's tile-extent rectangle and emitting only the tiles
// the ellipse actually crosses.
//
// The core primitive is ``rangeOverTiles(emit_ctx)``: for a given
// row (or column) of tiles, returns [tile_min, tile_max] — the inclusive
// tile-index range the ellipse touches in that row/column.
//
// Shader callers iterate the shorter side and emit per row/column.

#ifndef TRAINER_GAUSSIAN_ACCUTILE_SLANG
#define TRAINER_GAUSSIAN_ACCUTILE_SLANG

#include "../../common/math.slang"

// Ellipse data in pre-computed form (inverse covariance a,b,c + threshold t).
// Callers get this from snugBoxEllipse (see below).
struct EllipseParams
{
    float a;  // Σ⁻¹[0][0]
    float b;  // Σ⁻¹[0][1]
    float c;  // Σ⁻¹[1][1]
    float t;  // 2·log(255·opacity)
};

// Extract the same ellipse parameters snugBox uses. Keeps the math in one
// place; AccuTile needs them exposed so we don't redo the inverse.
//
// ``opacity_logit`` is the pre-sigmoid parameter; we sigmoid here to get
// the real probability that feeds the alpha threshold.
bool computeEllipseParams(float3 cov2d, float opacity_logit, out EllipseParams e)
{
    float cov00 = cov2d.x;
    float cov01 = cov2d.y;
    float cov11 = cov2d.z;
    float det_cov = max(cov00 * cov11 - cov01 * cov01, 1e-20);
    float inv_det = 1.0 / det_cov;
    e.a =  cov11 * inv_det;
    e.b = -cov01 * inv_det;
    e.c =  cov00 * inv_det;
    float opacity = sigmoid(opacity_logit);
    e.t = 2.0 * log(255.0 * opacity);
    return e.t > 0.0;
}

// Given y_d (= y - μy), solve the ellipse equation for x_d.
// Returns (x_min, x_max). If the row doesn't intersect the ellipse
// (discriminant < 0), returns (+inf, -inf).
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

// Compute the ellipse's x-range within a horizontal strip [y_bot, y_top]
// centered on mean_2d.y. Uses the paper's insight: the x-extrema of the
// ellipse land at y_d = ±y_d_args, so if either extremum y lies in the
// strip, the strip's x-range reaches the bbox's horizontal edge.
//
// ``bbox_min_x, bbox_max_x`` are SnugBox's bbox edges in absolute pixel
// space. ``mean_y`` is the gaussian's screen y. ``y_d_args`` is the y
// offset at which x-extrema occur (computed once per gaussian).
//
// Returns absolute (x_min, x_max) in pixel space, clamped to the bbox.
// When the strip doesn't intersect the ellipse, returns (+inf, -inf).
float2 accuTileRowXRange(
    EllipseParams e,
    float mean_y,
    float bbox_min_x, float bbox_max_x,
    float y_d_args,             // y-offset where x-extrema occur, positive
    float2 mean_2d,             // needed to translate x back to absolute
    float row_y_bot_abs, float row_y_top_abs)
{
    // Strip y_d range.
    float yb = row_y_bot_abs - mean_2d.y;
    float yt = row_y_top_abs - mean_2d.y;

    // Ellipse y-radius. The ellipse's y-extrema are at y_d = ±sqrt(t·c/(ac-b²))
    // but SnugBox already clamped everything into the bbox. If the strip
    // lies entirely outside [mean_y - y_max, mean_y + y_max], no intersection.
    float2 x_at_bot = ellipseXAtY(e, yb);
    float2 x_at_top = ellipseXAtY(e, yt);

    // Union of the ellipse's x-range across the strip.
    float xmin_abs = mean_2d.x + min(x_at_bot.x, x_at_top.x);
    float xmax_abs = mean_2d.x + max(x_at_bot.y, x_at_top.y);

    // If either x-extremum y lies inside the strip, the row's x-range
    // extends to the full bbox x-edge.
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

    // Clamp to bbox for safety (snapping to the bbox the gaussian already
    // cannot exceed).
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

// Compute the x_d_args and y_d_args (y and x offsets at ellipse extrema).
// Helper so callers don't duplicate the snugBox math.
void computeExtremumOffsets(EllipseParams e, out float x_d_args, out float y_d_args)
{
    float ac_minus_bb = max(e.a * e.c - e.b * e.b, 1e-20);
    x_d_args = sqrt(max((e.b * e.b * e.t) / (ac_minus_bb * max(e.a, 1e-20)), 0.0));
    y_d_args = sqrt(max((e.b * e.b * e.t) / (ac_minus_bb * max(e.c, 1e-20)), 0.0));
}

#endif // TRAINER_GAUSSIAN_ACCUTILE_SLANG
