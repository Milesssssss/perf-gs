// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_FIXED_POINT_SLANG
#define TRAINER_FIXED_POINT_SLANG

// Fixed-point atomic gradient accumulation (int64, no overflow).
static const float FIXED_SCALE = 65536.0;
static const float FIXED_INV_SCALE = 1.0 / FIXED_SCALE;

int64_t toFixed(float value) { return int64_t(value * FIXED_SCALE); }
float fromFixed(int64_t value) { return float(value) * FIXED_INV_SCALE; }

#endif // TRAINER_FIXED_POINT_SLANG
