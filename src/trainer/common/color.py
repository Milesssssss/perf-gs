"""sRGB ↔ linear color transfer functions.

The single source of truth for the true piecewise sRGB curve, which was
previously copy-pasted across pipeline.py, v2_train.py, v1_train.py, colmap.py,
and export.py. These are the exact IEC 61966-2-1 transfer functions.

NOTE: this is the *true* sRGB curve (piecewise, exponent 2.4). It is NOT the
same as the cheap ``x ** (1/2.2)`` display-gamma approximation used by
``export.render_to_png`` and ``v1_train._save_srgb_png`` — those are a
deliberately different (faster, less accurate) approximation and are left
alone.

``linear_to_srgb`` is the pure transfer function: it does NOT clamp its input
to [0, 1]. Callers that want a displayable value (e.g. PNG preview) should clip
to [0, 1] themselves before calling; callers that must round-trip unbounded
values (e.g. PLY f_dc export, where linear = 0.5 + C0*sh0 can leave [0, 1])
pass the raw value.
"""

from __future__ import annotations

import numpy as np


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB [0, 1] → linear. Returns float32."""
    low = c / 12.92
    high = np.power(np.clip((c + 0.055) / 1.055, 1e-7, None), 2.4)
    return np.where(c <= 0.04045, low, high).astype(np.float32)


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """Linear → sRGB. Pure transfer function; does not clamp the input range
    (see module docstring). Preserves the input dtype."""
    low = c * 12.92
    high = 1.055 * np.power(np.clip(c, 1e-7, None), 1.0 / 2.4) - 0.055
    return np.where(c <= 0.0031308, low, high)
