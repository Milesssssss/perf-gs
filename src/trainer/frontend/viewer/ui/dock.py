# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Edge-docking geometry for the viewer panel, with a user-resizable thickness.

Thicknesses are logical units (pixels at a 100% display); rects are device pixels; ``scale`` converts.
"""
from __future__ import annotations

from dataclasses import dataclass

ANCHORS = ("left", "right", "top", "bottom")

# ImGui style metrics stay raw pixels on HiDPI (unlike the font): subtract them before converting to logical.
PADDING = 8.0
SCROLLBAR = 18.0


@dataclass(frozen=True)
class Rect:
    """Window rect in pixels (x, y = top-left)."""

    x: float
    y: float
    w: float
    h: float

    def approx_eq(self, other: Rect, eps: float) -> bool:
        return (abs(self.x - other.x) <= eps and abs(self.y - other.y) <= eps
                and abs(self.w - other.w) <= eps and abs(self.h - other.h) <= eps)


class Dock:
    """Pins a panel to one window edge and owns its user-resizable thickness."""

    # Only a drag that leaves the panel pinned to its edge counts as a resize.
    PIN_TOL = 2.0
    # Allows mid-drag span drift but rejects collapsed/corner-dragged rects, which would read a bogus size.
    SPAN_TOL = 8.0
    # Below this, writing pos/size would just re-send what ImGui already has.
    WRITE_EPS = 0.5

    def __init__(self, anchor: str = "left", thickness: float = 360.0,
                 min_thickness: float = 320.0, max_fraction: float = 0.6,
                 scale: float = 1.0) -> None:
        if anchor not in ANCHORS:
            raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
        self.anchor = anchor
        self.min_thickness = float(min_thickness)
        self.max_fraction = float(max_fraction)
        self.scale = max(0.25, float(scale))
        self._thickness = float(thickness)
        self._rect = Rect(0.0, 0.0, self.thickness_px, self.thickness_px)

    @property
    def thickness(self) -> float:
        """Panel thickness in logical units (width when docked left/right)."""
        return self._thickness

    @property
    def thickness_px(self) -> float:
        """Panel thickness in device pixels — what ImGui is given."""
        return self._thickness * self.scale

    @property
    def rect(self) -> Rect:
        """Last rect returned by :meth:`resolve`, in device pixels."""
        return self._rect

    @property
    def vertical(self) -> bool:
        """True when the panel is a side bar (thickness is a width)."""
        return self.anchor in ("left", "right")

    def content_width(self) -> float:
        """Usable text width inside the panel in logical units, comparable with :mod:`.text` widths."""
        return max(32.0, self._rect.w - 2.0 * PADDING - SCROLLBAR) / self.scale

    def resolve(self, window_w: float, window_h: float,
                live: Rect | None = None) -> Rect:
        """Return the rect the panel should occupy this frame.

        ``live``: ImGui's current device-pixel rect (a user resize is adopted); None forces the stored thickness.
        """
        w = max(1.0, float(window_w))
        h = max(1.0, float(window_h))
        if live is not None:
            dragged = self._dragged_thickness(live, w, h)
            if dragged is not None:
                self._thickness = dragged / self.scale
        self._thickness = self._clamp(self._thickness,
                                      (w if self.vertical else h) / self.scale)
        t = self.thickness_px
        if self.anchor == "left":
            self._rect = Rect(0.0, 0.0, t, h)
        elif self.anchor == "right":
            self._rect = Rect(max(0.0, w - t), 0.0, t, h)
        elif self.anchor == "top":
            self._rect = Rect(0.0, 0.0, w, t)
        else:  # bottom
            self._rect = Rect(0.0, max(0.0, h - t), w, t)
        return self._rect

    def needs_write(self, live: Rect | None) -> bool:
        """Whether the resolved rect must be pushed to ImGui; skipping no-op writes avoids fighting a drag."""
        return live is None or not live.approx_eq(self._rect, self.WRITE_EPS)

    def _clamp(self, thickness: float, limit: float) -> float:
        """Both arguments and the result are in logical units."""
        largest = max(self.min_thickness, limit * self.max_fraction)
        return float(min(max(thickness, self.min_thickness), largest))

    def _dragged_thickness(self, live: Rect, w: float, h: float) -> float | None:
        """Pixel thickness the user just dragged the free edge to, or None if ``live`` is not such a resize."""
        if self.anchor == "left":
            pinned = abs(live.x) <= self.PIN_TOL
            spans = abs(live.h - h) <= self.SPAN_TOL
            return live.w if (pinned and spans) else None
        if self.anchor == "right":
            pinned = abs((live.x + live.w) - w) <= self.PIN_TOL
            spans = abs(live.h - h) <= self.SPAN_TOL
            return live.w if (pinned and spans) else None
        if self.anchor == "top":
            pinned = abs(live.y) <= self.PIN_TOL
            spans = abs(live.w - w) <= self.SPAN_TOL
            return live.h if (pinned and spans) else None
        pinned = abs((live.y + live.h) - h) <= self.PIN_TOL
        spans = abs(live.w - w) <= self.SPAN_TOL
        return live.h if (pinned and spans) else None
