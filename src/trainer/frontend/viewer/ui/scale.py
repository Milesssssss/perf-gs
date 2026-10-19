# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Display (HiDPI) scale of the panel font.

WHY THIS EXISTS: sgl builds the ImGui font atlas at the monitor's content scale,
so on a 150% display every glyph is drawn ~1.5x wider than the advances measured
in :mod:`.text` — but ImGui's style metrics (window padding, scrollbar) and sgl's
hard-coded ``PushItemWidth(300)`` stay in raw pixels. Text fitted with the raw
table therefore overflows the panel by the display scale, which is what clipped
the messages mid-word in the real viewer while an offscreen render of the same
panel (font atlas at 1.0x, because no window existed) looked perfect.

The atlas is rasterized at an INTEGER pixel size, so the effective scale is not
exactly the dpi ratio: measured live on a 144 dpi display, a row fitted with the
table's advances came out 1.538x wider, and 20/13 = 1.5385 — ImGui's 13px base
font scaled by 1.5 is 19.5px, rasterized as 20. :func:`font_scale` therefore
rounds the atlas size UP: over-measuring wraps a line one word early, which is
invisible, while under-measuring clips it, which is the bug being fixed.

Note the process is DPI-UNAWARE until an ``sgl.Window`` exists (the query then
returns a flat 96 dpi), so :func:`display_scale` is only meaningful once the
viewer window has been created — call it when building the panel, not at import.
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

    Rounds the rasterized atlas size up to the next whole pixel (see module
    docstring): never under-measure, because that is what clips text.
    """
    return math.ceil(BASE_FONT_PX * float(dpi_scale)) / BASE_FONT_PX


def query_dpi() -> float | None:
    """The primary monitor's effective dpi, or None if it can't be determined.

    Mirrors what GLFW (and therefore sgl's font atlas) uses: the per-monitor
    effective dpi, falling back to the system dpi and then to the legacy device
    caps. Any failure returns None so the caller can keep the 1.0 default.
    """
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
