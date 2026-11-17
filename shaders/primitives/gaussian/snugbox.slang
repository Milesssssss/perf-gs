// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// SnugBox (Speedy-Splat §4.1.1): opacity-aware tight AABB of the ellipse a·x_d² + 2b·x_d·y_d + c·y_d² <= t
// (Σ⁻¹ = [[a b][b c]]) where alpha exceeds the 1/255 display threshold.

#ifndef TRAINER_GAUSSIAN_SNUGBOX_SLANG
#define TRAINER_GAUSSIAN_SNUGBOX_SLANG

#include "../../common/math.slang"
#include "eval.slang"

// conic is the packed inverse covariance and t comes from splatCutoffQ; false when t <= 0 (the caller
// skips binning).
bool snugBox(float2 mean_2d, float3 conic, float t,
             out float2 bbox_min, out float2 bbox_max)
{
    float a = conic.x;   // Σ⁻¹[0][0]
    float b = conic.y;   // Σ⁻¹[0][1]
    float c = conic.z;   // Σ⁻¹[1][1]

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
