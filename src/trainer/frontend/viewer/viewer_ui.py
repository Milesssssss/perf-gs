"""Declarative UI + input abstraction for the in-process viewer.

Two small pieces that keep the viewer's render loop free of ImGui/event plumbing
and make it a one-liner to add controls or readouts.

``ViewerUI`` wraps the retained-mode ``slangpy.ui`` panel:
  * compose it declaratively — ``text`` (static hint), ``field`` (named live
    line), ``button``, ``slider_float`` / ``slider_int``, ``checkbox``, ``combo``,
    and ``group`` (nest a labelled section);
  * push live state INTO the panel each frame with ``set(name, value)``;
  * drive actions OUT of it via the widget callbacks;
  * it owns the frame lifecycle (``begin`` / ``end``) and reports whether ImGui
    consumed a keyboard/mouse event.

``InputMap`` maps keyboard keys to viewer actions:
  * ``hold(key, tag)`` adds ``tag`` to the live ``held`` set while the key is
    down (continuous actions like WASD movement);
  * ``press(key, fn)`` fires ``fn()`` once per key-down (discrete commands).

Extending the viewer's UI or controls is then adding one ``field`` / ``button``
/ ``hold`` / ``press`` line — no surgery on the event handlers or render loop.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Sequence

import slangpy as spy


class ViewerUI:
    """Thin wrapper over a ``slangpy.ui`` context + one panel window.

    Widgets are created once at build time (retained mode); ``field`` values are
    updated at runtime via :meth:`set`. All widget factories add to the current
    parent, which :meth:`group` swaps so sections can be nested. The panel is
    DOCKED to a window edge (``anchor``) and re-pinned every frame by
    :meth:`layout` — an anchored panel, not a floating window.
    """

    def __init__(self, device: spy.Device, *, title: str = "Viewer",
                 anchor: str = "left", thickness: float = 280.0) -> None:
        if anchor not in ("left", "right", "top", "bottom"):
            raise ValueError(f"anchor must be left/right/top/bottom, got {anchor!r}")
        self._ctx = spy.ui.Context(device)
        self._root = spy.ui.Window(self._ctx.screen, title,
                                   position=spy.float2(0.0, 0.0),
                                   size=spy.float2(thickness, thickness))
        self._parent: spy.ui.Widget = self._root
        self._fields: dict[str, spy.ui.Text] = {}
        self._anchor = anchor
        self._thickness = float(thickness)

    # ---- build-time composition (call these to lay out / EXTEND the panel) ----

    def text(self, text: str = "") -> None:
        """A static (non-updating) line, e.g. a help/hint line."""
        spy.ui.Text(self._parent, text)

    def field(self, name: str, initial: str = "") -> "ViewerUI":
        """A named text line whose value is pushed at runtime via ``set(name)``."""
        self._fields[name] = spy.ui.Text(self._parent, initial)
        return self

    def button(self, label: str, on_click: Callable[[], None]) -> spy.ui.Button:
        # Returned so callers can flip .label / .enabled at runtime.
        return spy.ui.Button(self._parent, label, callback=on_click)

    def checkbox(self, label: str, value: bool,
                 on_change: Callable[[bool], None]) -> spy.ui.CheckBox:
        return spy.ui.CheckBox(self._parent, label, value=bool(value),
                               callback=on_change)

    def slider_float(self, label: str, value: float, minimum: float, maximum: float,
                     on_change: Callable[[float], None],
                     fmt: str = "%.3f") -> spy.ui.SliderFloat:
        return spy.ui.SliderFloat(self._parent, label, value=float(value),
                                  min=float(minimum), max=float(maximum),
                                  callback=on_change, format=fmt)

    def slider_int(self, label: str, value: int, minimum: int, maximum: int,
                   on_change: Callable[[int], None]) -> spy.ui.SliderInt:
        return spy.ui.SliderInt(self._parent, label, value=int(value),
                                min=int(minimum), max=int(maximum),
                                callback=on_change)

    def combo(self, label: str, items: Sequence[str], value: int,
              on_change: Callable[[int], None]) -> spy.ui.ComboBox:
        return spy.ui.ComboBox(self._parent, label, value=int(value),
                               items=list(items), callback=on_change)

    @contextlib.contextmanager
    def group(self, title: str = ""):
        """Nest subsequent widgets under a labelled group. Usage::

            with ui.group("Render"):
                ui.checkbox("mip", True, on_mip)
        """
        prev = self._parent
        self._parent = spy.ui.Group(prev, title)
        try:
            yield self
        finally:
            self._parent = prev

    # ---- runtime: push state INTO the panel ----

    def set(self, name: str, text: str) -> None:
        """Update a :meth:`field`'s displayed text. Unknown names are ignored."""
        w = self._fields.get(name)
        if w is not None:
            w.text = str(text)

    def __setitem__(self, name: str, text: str) -> None:
        self.set(name, text)

    # ---- layout: dock the panel to a window edge (engine-style) ----

    def layout(self, window_width: int, window_height: int) -> None:
        """Pin the panel to its anchored edge, spanning that edge and tracking
        the current window size. Call every frame: re-asserting position/size
        keeps the panel docked (can't be dragged off) and following resizes."""
        w, h, t = float(window_width), float(window_height), self._thickness
        if self._anchor == "left":
            pos, size = (0.0, 0.0), (t, h)
        elif self._anchor == "right":
            pos, size = (max(0.0, w - t), 0.0), (t, h)
        elif self._anchor == "top":
            pos, size = (0.0, 0.0), (w, t)
        else:  # bottom
            pos, size = (0.0, max(0.0, h - t)), (w, t)
        self._root.position = spy.float2(*pos)
        self._root.size = spy.float2(*size)

    # ---- input + frame lifecycle ----

    def handle_keyboard(self, event) -> bool:
        """True if ImGui consumed the event (e.g. typing in a widget)."""
        return self._ctx.handle_keyboard_event(event)

    def handle_mouse(self, event) -> bool:
        """True if ImGui consumed the event (e.g. clicking/hovering the panel)."""
        return self._ctx.handle_mouse_event(event)

    def begin(self, width: int, height: int) -> None:
        self._ctx.begin_frame(width, height)

    def end(self, surface_texture, encoder) -> None:
        self._ctx.end_frame(surface_texture, encoder)


class InputMap:
    """Keyboard bindings: ``hold`` keys for continuous actions, ``press`` keys
    for discrete commands. Feed events via :meth:`handle`; read :attr:`held`
    from the render loop."""

    def __init__(self) -> None:
        self._hold: dict[object, str] = {}
        self._press: dict[object, Callable[[], None]] = {}
        self.held: set[str] = set()

    def hold(self, key, tag: str) -> "InputMap":
        self._hold[key] = tag
        return self

    def hold_many(self, pairs: Iterable[tuple[object, str]]) -> "InputMap":
        for key, tag in pairs:
            self._hold[key] = tag
        return self

    def press(self, key, on_press: Callable[[], None]) -> "InputMap":
        self._press[key] = on_press
        return self

    def handle(self, event) -> None:
        """Update ``held`` from key-down/up and fire ``press`` callbacks."""
        tag = self._hold.get(event.key)
        if event.is_key_press() or event.is_key_repeat():
            if tag is not None:
                self.held.add(tag)
        elif event.is_key_release():
            if tag is not None:
                self.held.discard(tag)
        if event.is_key_press():
            fn = self._press.get(event.key)
            if fn is not None:
                fn()
