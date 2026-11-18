# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The viewer's docked panel: declarative widgets bound to state getters, re-evaluated once per frame.

Captions sit above each control: sgl hard-codes ``PushItemWidth(300)``, which pushes inline labels off the panel.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import Any

import slangpy as spy

from trainer.frontend.viewer.ui import text as textmod
from trainer.frontend.viewer.ui.dock import Dock, Rect
from trainer.frontend.viewer.ui.scale import display_scale

Getter = Callable[[], Any]
# A static value or a zero-arg getter pulled each frame; every widget factory accepts both.
Dynamic = Any

# How a text value that overflows the panel is shortened.
WRAP = "wrap"
ELIDE_START = "start"     # keep the tail (paths: the leaf dir identifies it)
ELIDE_END = "end"
ELIDE_MIDDLE = "middle"


def _as_getter(value: Dynamic) -> Getter:
    return value if callable(value) else (lambda: value)


def _fit(value: str, mode: str, width: float, lines: int) -> list[str]:
    if mode == WRAP:
        return textmod.wrap(value, width, max_lines=lines)
    if mode == ELIDE_START:
        return [textmod.elide_start(value, width)]
    if mode == ELIDE_MIDDLE:
        return [textmod.elide_middle(value, width)]
    return [textmod.elide_end(value, width)]


class _TextBlock:
    """One text value over a fixed ``Text`` row pool, allocated at build time so widget order is build order."""

    def __init__(self, parent: spy.ui.Widget, getter: Getter, *,
                 lines: int, mode: str, gate: Getter | None = None) -> None:
        self._getter = getter
        self._mode = mode
        self._gate = gate
        self.rows = [spy.ui.Text(parent, "") for _ in range(max(1, lines))]
        self._cache: tuple[str, float, bool] | None = None

    def refresh(self, width: float) -> None:
        shown = True if self._gate is None else bool(self._gate())
        value = self._getter() if shown else ""
        value = "" if value is None else str(value)
        if self._cache == (value, width, shown):
            return
        self._cache = (value, width, shown)
        fitted = _fit(value, self._mode, width, len(self.rows)) if shown else []
        for i, row in enumerate(self.rows):
            row.visible = i < len(fitted)
            if row.visible:
                row.text = fitted[i]


class _LogBlock:
    """A scrolling tail of log lines, wrapped, newest last."""

    def __init__(self, parent: spy.ui.Widget, getter: Getter, *, lines: int) -> None:
        self._getter = getter
        self._rows = [spy.ui.Text(parent, "") for _ in range(max(1, lines))]
        self._cache: tuple[tuple[str, ...], float] | None = None

    def refresh(self, width: float) -> None:
        raw = tuple(str(line) for line in (self._getter() or ()))
        if self._cache == (raw, width):
            return
        self._cache = (raw, width)
        wrapped: list[str] = []
        for line in raw:
            wrapped.extend(textmod.wrap(line, width))
        tail = wrapped[-len(self._rows):]
        for i, row in enumerate(self._rows):
            if i < len(tail):
                row.text = tail[i]
                row.visible = True
            else:
                row.visible = False


class Panel:
    """A docked ``slangpy.ui`` window: declarative widgets bound to state."""

    def __init__(self, device: spy.Device, *, title: str = "Viewer",
                 anchor: str = "left", thickness: float = 380.0,
                 min_thickness: float = 340.0, scale: float | None = None) -> None:
        self._ctx = spy.ui.Context(device)
        # ``thickness``/``min_thickness`` are logical, so the panel holds the same text at any display scale.
        self._dock = Dock(anchor, thickness, min_thickness,
                          scale=display_scale() if scale is None else scale)
        rect = self._dock.rect
        self._root = spy.ui.Window(self._ctx.screen, title,
                                   position=spy.float2(rect.x, rect.y),
                                   size=spy.float2(rect.w, rect.h))
        self._parent: spy.ui.Widget = self._root
        self._bindings: list[Callable[[float], None]] = []
        self._failed: set[int] = set()
        self._widget_seq = 0

    @property
    def dock(self) -> Dock:
        return self._dock

    @property
    def window(self) -> spy.ui.Window:
        return self._root

    @property
    def visible(self) -> bool:
        return bool(self._root.visible)

    def toggle(self) -> None:
        """Show/hide the panel: the way back after a stray click on ImGui's window close box."""
        self._root.visible = not self._root.visible

    def _widget_id(self, visible_label: str = "") -> str:
        """``visible_label`` plus a unique, invisible ``##`` ImGui id suffix.

        Every id-bearing widget must use this: same-labelled (e.g. empty) widgets would otherwise share one id.
        """
        self._widget_seq += 1
        return f"{visible_label}##w{self._widget_seq}"

    @staticmethod
    def _relabel(getter: Getter, suffix: str) -> Getter:
        """Keep a dynamic label's ``##`` id suffix, which a per-frame ``label`` binding would strip."""
        return lambda: f"{getter()}{suffix}"

    def bind(self, widget: spy.ui.Widget, **getters: Dynamic) -> spy.ui.Widget:
        """Drive any writable widget property (``enabled``, ``visible``, ``label``, ...) from state, every frame."""
        pairs = [(name, _as_getter(g)) for name, g in getters.items() if g is not None]
        if pairs:
            def apply(_width: float, widget=widget, pairs=pairs) -> None:
                for name, getter in pairs:
                    setattr(widget, name, getter())
            self._bindings.append(apply)
        return widget

    def text(self, value: Dynamic = "", *, lines: int = 1, mode: str = WRAP,
             visible: Dynamic = None) -> None:
        """A text readout wrapped (or elided per ``mode``) to the panel's live width over ``lines`` rows."""
        block = _TextBlock(self._parent, _as_getter(value), lines=lines, mode=mode,
                           gate=None if visible is None else _as_getter(visible))
        self._bindings.append(block.refresh)

    def spacer(self) -> None:
        """One blank line of vertical space."""
        spy.ui.Text(self._parent, "")

    def log(self, lines_getter: Getter, *, lines: int = 8) -> None:
        """A tail of the last ``lines`` (wrapped) log lines from a sequence getter."""
        block = _LogBlock(self._parent, lines_getter, lines=lines)
        self._bindings.append(block.refresh)

    def progress(self, fraction: Dynamic, *, caption: Dynamic | None = None,
                 **binds: Dynamic) -> spy.ui.ProgressBar:
        """A determinate progress bar (0..1), optionally captioned above."""
        if caption is not None:
            self.text(caption)
        bar = spy.ui.ProgressBar(self._parent, 0.0)
        self.bind(bar, fraction=lambda g=_as_getter(fraction): float(
            min(1.0, max(0.0, float(g() or 0.0)))), **binds)
        return bar

    def button(self, label: Dynamic, on_click: Callable[[], None],
               **binds: Dynamic) -> spy.ui.Button:
        # A button's label is its id, so same-text (or not-yet-bound "") buttons would collide.
        suffix = self._widget_id()
        button = spy.ui.Button(
            self._parent, (label if isinstance(label, str) else "") + suffix,
            callback=on_click)
        if not isinstance(label, str):
            binds.setdefault("label", self._relabel(_as_getter(label), suffix))
        self.bind(button, **binds)
        return button

    def _caption_row(self, caption: str, binds: dict) -> None:
        """A control's caption on the line above it, sharing the control's ``visible`` gate so it never orphans."""
        if caption:
            self.text(caption, visible=binds.get("visible"))

    def input_text(self, value: Dynamic, on_change: Callable[[str], None],
                   **binds: Dynamic) -> spy.ui.InputText:
        """A single-line text field whose value is not re-bound each frame, which would fight the caret."""
        widget = spy.ui.InputText(self._parent, self._widget_id(),
                                  value=str(_as_getter(value)()),
                                  callback=on_change)
        self.bind(widget, **binds)
        return widget

    def checkbox(self, caption: str, value: Dynamic,
                 on_change: Callable[[bool], None], **binds: Dynamic) -> spy.ui.CheckBox:
        box = spy.ui.CheckBox(self._parent, self._widget_id(caption),
                              value=bool(_as_getter(value)()),
                              callback=on_change)
        if callable(value):
            binds.setdefault("value", value)
        self.bind(box, **binds)
        return box

    def combo(self, caption: str, items: Dynamic, value: Dynamic,
              on_change: Callable[[int], None], **binds: Dynamic) -> spy.ui.ComboBox:
        """A dropdown captioned on the line above; ``items`` may be a getter for a list that changes at runtime."""
        self._caption_row(caption, binds)
        get_items = _as_getter(items)
        box = spy.ui.ComboBox(self._parent, self._widget_id(),
                              value=int(_as_getter(value)()),
                              items=list(get_items()), callback=on_change)
        if callable(items):
            binds.setdefault("items", lambda: list(get_items()))
        if callable(value):
            binds.setdefault("value", lambda g=_as_getter(value): int(g()))
        self.bind(box, **binds)
        return box

    def slider_float(self, caption: str, value: Dynamic, minimum: float,
                     maximum: float, on_change: Callable[[float], None],
                     fmt: str = "%.3f", **binds: Dynamic) -> spy.ui.SliderFloat:
        self._caption_row(caption, binds)
        slider = spy.ui.SliderFloat(self._parent, self._widget_id(),
                                    value=float(_as_getter(value)()),
                                    min=float(minimum), max=float(maximum),
                                    callback=on_change, format=fmt)
        if callable(value):
            binds.setdefault("value", lambda g=_as_getter(value): float(g()))
        self.bind(slider, **binds)
        return slider

    def slider_int(self, caption: str, value: Dynamic, minimum: int, maximum: int,
                   on_change: Callable[[int], None], **binds: Dynamic) -> spy.ui.SliderInt:
        self._caption_row(caption, binds)
        slider = spy.ui.SliderInt(self._parent, self._widget_id(),
                                  value=int(_as_getter(value)()),
                                  min=int(minimum), max=int(maximum),
                                  callback=on_change)
        if callable(value):
            binds.setdefault("value", lambda g=_as_getter(value): int(g()))
        self.bind(slider, **binds)
        return slider

    def drag_int(self, caption: str, value: Dynamic, on_change: Callable[[int], None],
                 *, minimum: int = 0, maximum: int = 0, speed: float = 1.0,
                 fmt: str = "%d", **binds: Dynamic) -> spy.ui.DragInt:
        """A drag field for a wide-range int (iterations, splat counts); a slider can't reach exact values."""
        self._caption_row(caption, binds)
        widget = spy.ui.DragInt(
            self._parent, self._widget_id(), value=int(_as_getter(value)()),
            callback=on_change,
            speed=float(speed), min=int(minimum), max=int(maximum), format=fmt,
            flags=spy.ui.SliderFlags.always_clamp)
        if callable(value):
            binds.setdefault("value", lambda g=_as_getter(value): int(g()))
        self.bind(widget, **binds)
        return widget

    def drag_float(self, caption: str, value: Dynamic,
                   on_change: Callable[[float], None], *, minimum: float = 0.0,
                   maximum: float = 0.0, speed: float = 0.01, fmt: str = "%.3f",
                   **binds: Dynamic) -> spy.ui.DragFloat:
        """A drag field for a float (see :meth:`drag_int` for why not a slider)."""
        self._caption_row(caption, binds)
        widget = spy.ui.DragFloat(
            self._parent, self._widget_id(), value=float(_as_getter(value)()),
            callback=on_change,
            speed=float(speed), min=float(minimum), max=float(maximum), format=fmt,
            flags=spy.ui.SliderFlags.always_clamp)
        if callable(value):
            binds.setdefault("value", lambda g=_as_getter(value): float(g()))
        self.bind(widget, **binds)
        return widget

    @contextlib.contextmanager
    def section(self, title: str, **binds: Dynamic):
        """A collapsible section header; widgets built inside the ``with`` block nest under it."""
        group = spy.ui.Group(self._parent, title)
        self.bind(group, **binds)
        previous, self._parent = self._parent, group
        try:
            yield group
        finally:
            self._parent = previous

    def add(self, build: Callable[[Panel], None]) -> Panel:
        build(self)
        return self

    def layout(self, window_width: float, window_height: float) -> None:
        """Re-pin the panel to its edge, adopting a user resize of its free edge.

        Writes pos/size only when they differ, so an in-progress border drag is never fought.
        """
        live = Rect(self._root.position.x, self._root.position.y,
                    self._root.size.x, self._root.size.y)
        rect = self._dock.resolve(window_width, window_height, live)
        if self._dock.needs_write(live):
            self._root.position = spy.float2(rect.x, rect.y)
            self._root.size = spy.float2(rect.w, rect.h)

    def refresh(self) -> None:
        """Re-evaluate every binding. Call once per frame, before rendering."""
        width = self._dock.content_width()
        for index, binding in enumerate(self._bindings):
            try:
                binding(width)
            except Exception as error:  # noqa: BLE001 - one bad getter must not kill the render thread or the panel
                if index not in self._failed:
                    self._failed.add(index)
                    print(f"[Viewer] UI binding #{index} failed: {error}")

    def handle_keyboard(self, event) -> bool:
        """True if ImGui consumed the event (e.g. typing in a widget)."""
        return self._ctx.handle_keyboard_event(event)

    def handle_mouse(self, event) -> bool:
        """True if ImGui consumed the event (e.g. clicking/hovering the panel)."""
        return self._ctx.handle_mouse_event(event)

    def begin(self, width: int, height: int, window=None) -> None:
        """Passing the sgl ``window`` lets ImGui drive the mouse cursor (e.g. the border's resize cursor)."""
        self._ctx.begin_frame(width, height, window)

    def end(self, surface_texture, encoder) -> None:
        self._ctx.end_frame(surface_texture, encoder)


def bullets(entries: Sequence[tuple[str, str]], pad: int = 11) -> str:
    """Format ``(key, action)`` help rows into roughly aligned lines (the panel font is proportional)."""
    return "\n".join(f"{key:<{pad}s}{action}" for key, action in entries)
