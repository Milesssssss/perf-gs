"""The viewer's docked panel: declarative composition + state BINDINGS.

The old panel could only be updated by pushing strings into it from the render
loop (``ui.set("status", ...)``) and by flipping ``widget.enabled`` by hand, so
every readout cost a line in the panel build AND a line in the hot loop, and the
loop grew a copy of the UI's logic. Here a widget instead PULLS from a getter:

    panel.text(lambda: f"{stats().step}/{stats().total} iters")
    panel.button("Train", start, enabled=lambda: state.can_train)
    panel.progress(lambda: state.progress().fraction)

:meth:`refresh` re-evaluates every binding once per frame — the render loop calls
it and knows nothing else about the UI. Adding a control or a readout is one
declarative line next to the others, and it can both read and drive state.

Text bindings are wrapped/elided to the panel's live content width (see
:mod:`.text`), which is what stops long paths and log lines from being clipped
mid-word. That width is in LOGICAL units and the panel is sized in pixels via the
display scale (see :mod:`.scale`) — sgl builds the font atlas at the monitor's
content scale, so on a 150% display raw-pixel wrapping under-measures every line
by 1.5x and the text clips anyway. Widget labels are deliberately EMPTY: sgl
hard-codes ``ImGui::PushItemWidth(300)``, so a labelled combo/slider draws its
label past the 300px control and off the edge of a ~380px panel; captions
therefore go on their own line above the control.
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
# A value that is either static or pulled from state each frame (a zero-arg
# callable). Every widget factory below accepts both.
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
    """One logical text value rendered over a fixed pool of ``Text`` widgets.

    The pool is allocated at build time (so widget order matches the build order)
    and unused rows are hidden, which keeps a one-line value one line tall.
    """

    def __init__(self, parent: spy.ui.Widget, getter: Getter, *,
                 lines: int, mode: str, gate: Getter | None = None) -> None:
        self._getter = getter
        self._mode = mode
        self._gate = gate
        self.rows = [spy.ui.Text(parent, "") for _ in range(max(1, lines))]
        self._cache: tuple[str, float, bool] | None = None

    def refresh(self, width: float) -> None:
        shown = True if self._gate is None else bool(self._gate())
        value = self._getter()
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
        # ``thickness``/``min_thickness`` are logical: the panel holds the same
        # amount of text on a 100% and a 150% display (it is just wider in pixels
        # there, because so is the font). Pass ``scale`` explicitly to test.
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
        """Show/hide the panel. ImGui gives its window a close box, so without a
        way back a stray click on it would lose the whole UI for the session."""
        self._root.visible = not self._root.visible

    # ---- composition -----------------------------------------------------

    def _widget_id(self, visible_label: str = "") -> str:
        """``visible_label`` plus a unique, INVISIBLE ImGui id suffix.

        ImGui identifies a widget BY ITS LABEL (``window->GetID(label)``) and
        hides everything after a ``##``. Every control here puts its caption on
        the line above (sgl's hard-coded item width pushes a real label off the
        panel — see the module docstring), so they were all constructed with the
        same empty label and therefore shared ONE id across the whole window.
        ImGui tracks a single active id, so the first such widget drawn each
        frame reclaimed it: a text edit lost focus after one keystroke, and a
        drag could scrub a different field than the one under the cursor.

        Every id-bearing widget must therefore route its label through here. The
        suffix is positional, which is stable because the panel is BUILT once and
        then only re-bound (see :meth:`bind`) — ids do not shuffle between frames.
        """
        self._widget_seq += 1
        return f"{visible_label}##w{self._widget_seq}"

    @staticmethod
    def _relabel(getter: Getter, suffix: str) -> Getter:
        """Keep a dynamically-labelled widget's id STABLE.

        A ``label`` binding overwrites the label every frame, which would strip
        the ``##`` suffix that :meth:`_widget_id` added and change the widget's
        id mid-run — worse than sharing an id, because ImGui would drop any
        interaction in progress.
        """
        return lambda: f"{getter()}{suffix}"

    def bind(self, widget: spy.ui.Widget, **getters: Dynamic) -> spy.ui.Widget:
        """Drive widget PROPERTIES from state, re-evaluated every frame::

            panel.bind(button, enabled=lambda: state.can_train)

        Any writable property works (``enabled``, ``visible``, ``label``,
        ``value``, ``fraction``, ``items``, ...), so a new control needs no
        support code in the render loop.
        """
        pairs = [(name, _as_getter(g)) for name, g in getters.items() if g is not None]
        if pairs:
            def apply(_width: float, widget=widget, pairs=pairs) -> None:
                for name, getter in pairs:
                    setattr(widget, name, getter())
            self._bindings.append(apply)
        return widget

    def text(self, value: Dynamic = "", *, lines: int = 1, mode: str = WRAP,
             visible: Dynamic = None) -> None:
        """A text readout. ``value`` may be a string or a getter; it is wrapped
        (or elided, see ``mode``) to the panel's live width across ``lines``
        rows. ``visible`` gates the whole block (hidden rows take no space)."""
        block = _TextBlock(self._parent, _as_getter(value), lines=lines, mode=mode,
                           gate=None if visible is None else _as_getter(visible))
        self._bindings.append(block.refresh)

    def spacer(self) -> None:
        """One blank line of vertical space."""
        spy.ui.Text(self._parent, "")

    def log(self, lines_getter: Getter, *, lines: int = 8) -> None:
        """A tail of the last ``lines`` (wrapped) log lines from a sequence
        getter — the multi-line replacement for a single clipped status line."""
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
        # A button's label IS its id, so two buttons showing the same text (or
        # both momentarily "" while their getters have not run) would collide.
        suffix = self._widget_id()
        button = spy.ui.Button(
            self._parent, (label if isinstance(label, str) else "") + suffix,
            callback=on_click)
        if not isinstance(label, str):
            binds.setdefault("label", self._relabel(_as_getter(label), suffix))
        self.bind(button, **binds)
        return button

    def _caption_row(self, caption: str, binds: dict) -> None:
        """A control's caption, on the line above it (sgl's hard-coded item width
        pushes a real widget label off the panel — see the module docstring).

        It inherits the control's ``visible`` gate: a caption that outlives its own
        widget is an orphan label, which is exactly how a hidden section leaks a
        stray "feature matcher" into an empty panel.
        """
        if caption:
            self.text(caption, visible=binds.get("visible"))

    def input_text(self, value: Dynamic, on_change: Callable[[str], None],
                   **binds: Dynamic) -> spy.ui.InputText:
        """A single-line text field — the only text entry this toolkit has.

        Used for the export filename, so a result can be named without a file
        dialog (there is none in this windowing layer). The value is NOT re-bound
        from state while the user types: pushing a getter into the widget every
        frame would fight the caret.
        """
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
        """A dropdown with its caption on the line ABOVE (see module docstring).
        ``items`` may be a getter, so a list that changes at runtime (the camera
        list) needs no manual rebuild in the render loop."""
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
        """A drag field for an INT with a wide range (iterations, splat counts).

        Preferred over :meth:`slider_int` whenever the range spans orders of
        magnitude: a slider maps 100..100,000,000 onto ~300px, so a single pixel
        is ~300k splats and an exact value is unreachable. A drag has no track —
        drag to scrub coarsely, ctrl+click to type the exact number — and
        ``always_clamp`` keeps a typed value inside [minimum, maximum].
        """
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
        """A drag field for a FLOAT (see :meth:`drag_int` for why not a slider)."""
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
        """A collapsible section header; nest widgets inside it::

            with panel.section("DATASET"):
                panel.text(...)
        """
        group = spy.ui.Group(self._parent, title)
        self.bind(group, **binds)
        previous, self._parent = self._parent, group
        try:
            yield group
        finally:
            self._parent = previous

    def add(self, build: Callable[[Panel], None]) -> Panel:
        """Append a section built by a callable — how the viewer composes the
        panel out of independent, testable section builders."""
        build(self)
        return self

    # ---- per-frame ------------------------------------------------------

    def layout(self, window_width: float, window_height: float) -> None:
        """Re-pin the panel to its edge, ADOPTING a user resize of its free edge
        (see :mod:`.dock`). Writes pos/size only when they actually differ, so an
        in-progress border drag is never fought."""
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
            except Exception as error:  # noqa: BLE001 - one bad getter must not
                if index not in self._failed:  # kill the render thread or the
                    self._failed.add(index)    # rest of the panel
                    print(f"[Viewer] UI binding #{index} failed: {error}")

    def handle_keyboard(self, event) -> bool:
        """True if ImGui consumed the event (e.g. typing in a widget)."""
        return self._ctx.handle_keyboard_event(event)

    def handle_mouse(self, event) -> bool:
        """True if ImGui consumed the event (e.g. clicking/hovering the panel)."""
        return self._ctx.handle_mouse_event(event)

    def begin(self, width: int, height: int, window=None) -> None:
        """Passing the sgl ``window`` lets ImGui drive the mouse cursor, so the
        panel's resizable border actually shows a resize cursor."""
        self._ctx.begin_frame(width, height, window)

    def end(self, surface_texture, encoder) -> None:
        self._ctx.end_frame(surface_texture, encoder)


def bullets(entries: Sequence[tuple[str, str]], pad: int = 11) -> str:
    """Format ``(key, action)`` help rows into aligned lines. The panel font is
    proportional, so this is only a rough column — good enough for a help block
    and it wraps cleanly."""
    return "\n".join(f"{key:<{pad}s}{action}" for key, action in entries)
