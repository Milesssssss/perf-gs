# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

from __future__ import annotations

import numpy as np


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB [0, 1] → linear. Returns float32."""
    low = c / 12.92
    high = np.power(np.clip((c + 0.055) / 1.055, 1e-7, None), 2.4)
    return np.where(c <= 0.04045, low, high).astype(np.float32)
