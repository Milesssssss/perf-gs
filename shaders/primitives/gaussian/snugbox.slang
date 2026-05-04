// SnugBox (Speedy-Splat §4.1.1): tight axis-aligned bounding box of the
// ellipse where the Gaussian's alpha = opacity * exp(power) exceeds the
// 1/255 display threshold. Factoring in opacity gives a much tighter
// bbox than the opacity-agnostic 3-sigma-eigenvalue estimate.
//
// Derivation, using inverse covariance Σ⁻¹ = [[a b][b c]]:
//   alpha(p) >= 1/255  <=>  ax_d² + 2b·x_d·y_d + c·y_d² <= t
//                        with t = 2·log(255·σ)
// Differentiating the ellipse y_d = (-b·x_d ± sqrt((b²-ac)·x_d² + tc)) / c
// w.r.t. x_d and solving ∂y/∂x = 0 gives the x_d values at y-extrema:
//   x_d_args = ± sqrt( -b²·t / ((b²-ac)·a) ) = ± sqrt( b²·t / ((ac-b²)·a) )
// Substituting back gives the y-extrema. x-extrema by symmetry.
//
// Returns false when the gaussian has no visible extent (opacity too
// low: t <= 0) — caller should skip binning.

#ifndef TRAINER_GAUSSIAN_SNUGBOX_SLANG
#define TRAINER_GAUSSIAN_SNUGBOX_SLANG

bool snugBox(float2 mean_2d, float3 cov2d, float opacity,
             out float2 bbox_min, out float2 bbox_max)
{
    float cov00 = cov2d.x;
    float cov01 = cov2d.y;
    float cov11 = cov2d.z;
    float det_cov = cov00 * cov11 - cov01 * cov01;
    det_cov = max(det_cov, 1e-20);
    float inv_det = 1.0 / det_cov;
    float a =  cov11 * inv_det;   // Σ⁻¹[0][0]
    float b = -cov01 * inv_det;   // Σ⁻¹[0][1]
    float c =  cov00 * inv_det;   // Σ⁻¹[1][1]

    float t = 2.0 * log(255.0 * opacity);
    if (t <= 0.0)
    {
        bbox_min = mean_2d;
        bbox_max = mean_2d;
        return false;
    }

    // ac - b² = det(Σ⁻¹) > 0 (Σ is positive definite).
    float ac_minus_bb = a * c - b * b;
    ac_minus_bb = max(ac_minus_bb, 1e-20);

    // x at y-extremum: sqrt( b²·t / ((ac-b²)·a) )
    float x_d_args = sqrt(max((b * b * t) / (ac_minus_bb * max(a, 1e-20)), 0.0));
    // radical at the extremum: sqrt( t·(ac-b²)/a )
    float rad_y = sqrt(max(t * ac_minus_bb / max(a, 1e-20), 0.0));
    // y_radius maximises over sign choices of x_d and the radical branch.
    float y_radius = (abs(b) * x_d_args + rad_y) / max(c, 1e-20);

    float y_d_args = sqrt(max((b * b * t) / (ac_minus_bb * max(c, 1e-20)), 0.0));
    float rad_x = sqrt(max(t * ac_minus_bb / max(c, 1e-20), 0.0));
    float x_radius = (abs(b) * y_d_args + rad_x) / max(a, 1e-20);

    bbox_min = mean_2d - float2(x_radius, y_radius);
    bbox_max = mean_2d + float2(x_radius, y_radius);
    return true;
}

#endif // TRAINER_GAUSSIAN_SNUGBOX_SLANG
