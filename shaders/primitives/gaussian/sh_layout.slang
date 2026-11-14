// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

// SH band limits; mirrored by trainer.common.sh.

#ifndef TRAINER_SH_LAYOUT
#define TRAINER_SH_LAYOUT

static const int SH_MAX_DEGREE = 3;
static const int SH_REST_COEFFS = 15;  // per channel, excludes the DC term
static const int SH_REST_FLOATS = 45;  // SH_REST_COEFFS * 3

#endif // TRAINER_SH_LAYOUT
