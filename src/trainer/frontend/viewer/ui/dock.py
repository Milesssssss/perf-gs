# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Edge-docking geometry for the viewer panel — including the resize fix.

THE BUG this replaces: the old ``layout()`` re-asserted BOTH position and size
every frame. ImGui's manual-resize path wins while the mouse is down, so the
panel followed the border drag, but on mouse-up the per-frame
``SetNextWindowSize`` took over again and snapped it straight back to the
hard-coded thickness (verified with a synthesized border drag: size tracked the
mouse to 440px, then reverted to 300px on release). Re-asserting the size every
frame also means fighting ImGui for ownership of the window rect all the time,
which is what made the panel feel "totally bugged" during a drag.

THE FIX, all of it in :meth:`Dock.resolve` (pure geometry, no ImGui):

  * ADOPT the user's drag: when the docked edge is still pinned and the panel
    still spans its edge, the live thickness IS the new thickness. The gesture
    therefore survives mouse-up instead of being reverted.
  * CLAMP it to ``[min_thickness, span * max_fraction]`` so the panel can never
    be dragged to unreadable or to swallowing the viewport.
  * Re-pin only what the user did NOT ask for (position, and the span axis), and
    let the caller write pos/size ONLY when they differ from what ImGui already
    has (:meth:`needs_write`) — so a settled panel is never written to at all and
    ImGui keeps ownership of an in-progress drag.

UNITS: thicknesses are LOGICAL (font-table units, i.e. pixels at a 100% display),
rects and the live rect from ImGui are DEVICE PIXELS, and ``scale`` converts
between them (see :mod:`.scale`). That keeps one number — the panel's thickness —
meaningful on any display: at 150% the panel is 1.5x wider in pixels and holds
exactly the same amount of text, because the font is 1.5x wider too.
"""
from __future__ import annotations

from dataclasses import dataclass

ANCHORS = ("left", "right", "top", "bottom")

# Content geometry of an ImGui window: 8px padding each side, plus room for the
# vertical scrollbar so wrapped text never sits under it. These are ImGui STYLE
# values, which sgl leaves in raw pixels even on a HiDPI display (unlike the
# font), so they are subtracted before converting to logical units.
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

    # A drag of the docked (outer) edge or the title bar moves the panel; only a
    # drag that leaves the panel pinned to its edge counts as a resize.
    PIN_TOL = 2.0
    # The span axis drifts by a pixel or two mid-drag / mid-window-resize; allow
    # that, but refuse to adopt a thickness while the panel is collapsed or
    # being corner-dragged (span far off), which would read a bogus size.
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
        """Panel thickness in LOGICAL units (width when docked left/right)."""
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
        """True when the panel is a side bar (thickness is a WIDTH)."""
        return self.anchor in ("left", "right")

    def content_width(self) -> float:
        """Usable text width inside the panel, in LOGICAL units — i.e. directly
        comparable with :mod:`.text` widths, whatever the display scale."""
        return max(32.0, self._rect.w - 2.0 * PADDING - SCROLLBAR) / self.scale

    def resolve(self, window_w: float, window_h: float,
                live: Rect | None = None) -> Rect:
        """Return the rect the panel should occupy this frame.

        ``live`` is the panel's CURRENT rect as reported by ImGui, in device
        pixels (i.e. including an in-progress or just-finished user resize); pass
        None to force the stored thickness.
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
        """Whether the resolved rect must be pushed to ImGui. False while the
        panel already sits there — the key to not fighting an active drag."""
        return live is None or not live.approx_eq(self._rect, self.WRITE_EPS)

    # ---- internals ----

    def _clamp(self, thickness: float, limit: float) -> float:
        """Both arguments and the result are in logical units."""
        largest = max(self.min_thickness, limit * self.max_fraction)
        return float(min(max(thickness, self.min_thickness), largest))

    def _dragged_thickness(self, live: Rect, w: float, h: float) -> float | None:
        """The thickness IN PIXELS the user just dragged the panel to, or None if
        this live rect is not a resize of the free edge (panel moved, collapsed,
        or the window itself resized)."""
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
