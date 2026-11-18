# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Dimensionless strength of the cos-fourth encoded-RGB lens response."""
import math

import numpy as np


# Match the FP32 constant folded by Slang for -2.0 / 2.2 (SPIR-V bits 0xbf68ba2e).
COS4_SRGB_EXPONENT = float(np.float32(-2.0) / np.float32(2.2))


def parse_strength(value: str | float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0 or value > np.finfo(np.float32).max:
        raise ValueError("lens strength must be a finite nonnegative float32 value")
    return value
