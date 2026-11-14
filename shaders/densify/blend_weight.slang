// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_BLEND_WEIGHT_SLANG
#define TRAINER_BLEND_WEIGHT_SLANG

#include "../common/fixed_point.slang"

// A splat whose blend weight (alpha * T summed over a view's pixels), averaged
// over the views it appeared in during the densify window, stays under
// min_blend_weight pixels barely changes any image. A splat the window never
// showed is kept: no evidence either way. min_blend_weight <= 0 disables.
bool isUnderweight(int64_t blend_weight, uint views, float min_blend_weight)
{
    return min_blend_weight > 0.0 && views > 0u
        && fromFixed(blend_weight) < min_blend_weight * float(views);
}

#endif // TRAINER_BLEND_WEIGHT_SLANG
