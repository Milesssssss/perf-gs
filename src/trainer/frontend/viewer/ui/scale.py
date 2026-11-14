# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Display (HiDPI) scale of the panel font, which sgl rasterizes at the monitor's content scale.

Only meaningful once an ``sgl.Window`` exists: before that the process is DPI-unaware and reports 96 dpi.
"""
from __future__ import annotations

import math
import sys

# ImGui's own baseline: 96 dpi == 100% == the scale :mod:`.text` was measured at.
BASE_DPI = 96.0
# ImGui's default font size in pixels, i.e. what the display scale multiplies.
BASE_FONT_PX = 13.0
# Guard against a nonsense query result driving the layout off a cliff.
MIN_SCALE = 1.0
MAX_SCALE = 4.0


def normalize(dpi: float) -> float:
    """Turn a raw dpi reading into a usable scale factor."""
    scale = float(dpi) / BASE_DPI
    if not (scale == scale) or scale <= 0.0:      # NaN / garbage
        return MIN_SCALE
    return min(MAX_SCALE, max(MIN_SCALE, scale))


def font_scale(dpi_scale: float) -> float:
    """How much wider the live font is than :mod:`.text`'s measured advances.

    The atlas is rasterized at a whole-pixel size, so round up: under-measuring clips text.
    """
    return math.ceil(BASE_FONT_PX * float(dpi_scale)) / BASE_FONT_PX


def query_dpi() -> float | None:
    """The primary monitor's effective dpi as GLFW (and so sgl's font atlas) sees it, or None if unknown."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        # MonitorFromPoint(0,0, MONITOR_DEFAULTTOPRIMARY)
        monitor = user32.MonitorFromPoint(ctypes.wintypes.POINT(0, 0), 1)
        x, y = ctypes.c_uint(), ctypes.c_uint()
        # GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &x, &y)
        if ctypes.windll.shcore.GetDpiForMonitor(
                monitor, 0, ctypes.byref(x), ctypes.byref(y)) == 0 and x.value:
            return float(x.value)
        if hasattr(user32, "GetDpiForSystem"):
            dpi = user32.GetDpiForSystem()
            if dpi:
                return float(dpi)
        dc = user32.GetDC(0)
        try:
            return float(ctypes.windll.gdi32.GetDeviceCaps(dc, 88))  # LOGPIXELSX
        finally:
            user32.ReleaseDC(0, dc)
    except Exception:       # noqa: BLE001 - a UI nicety must never break startup
        return None


def display_scale() -> float:
    """Scale the panel font is actually rendered at (1.0 when unknown)."""
    dpi = query_dpi()
    return MIN_SCALE if dpi is None else font_scale(normalize(dpi))
