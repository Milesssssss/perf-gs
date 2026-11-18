# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Device-free harness for the viewer panel.

A section builder declares widgets and binds getters to them; :class:`RecordingPanel`
captures those declarations instead of creating ImGui objects, so what a section
shows in a given state is testable without a GPU, a window or a font atlas.

The bindings are kept LIVE (they are the same callables the real panel would
re-evaluate every frame), so a test can move the session on and re-read them — which
is exactly what the UI does.
"""
from __future__ import annotations

import contextlib


class Declared:
    """One widget a section asked for, plus the getters bound to it."""

    def __init__(self, kind: str, **info) -> None:
        self.kind = kind
        self.binds = info.pop("binds", {})
        self.info = info

    # ---- what the user would see -------------------------------------

    @property
    def shown(self) -> bool:
        """Whether this widget is visible in the CURRENT state."""
        gate = self.binds.get("visible")
        return True if gate is None else bool(gate())

    @property
    def enabled(self) -> bool:
        gate = self.binds.get("enabled")
        return True if gate is None else bool(gate())

    @property
    def label(self) -> str:
        """The live label (buttons and sections may bind a dynamic one)."""
        binding = self.binds.get("label")
        if binding is not None:
            return str(binding())
        return str(self.info.get("label", self.info.get("title", "")))

    def rendered(self) -> str:
        """The live text of a text widget."""
        value = self.info.get("value", "")
        return str(value() if callable(value) else value)

    # ---- what the user would do --------------------------------------

    def click(self) -> None:
        self.info["on_click"]()

    def change(self, value) -> None:
        self.info["on_change"](value)


class RecordingPanel:
    """Stand-in for :class:`~trainer.frontend.viewer.ui.panel.Panel`."""

    def __init__(self) -> None:
        self.widgets: list[Declared] = []

    # ---- queries -----------------------------------------------------

    def of(self, kind: str) -> list[Declared]:
        return [w for w in self.widgets if w.kind == kind]

    def only(self, kind: str, **match) -> Declared:
        found = [w for w in self.widgets if w.kind == kind
                 and all(w.info.get(k) == v for k, v in match.items())]
        assert len(found) == 1, f"expected one {kind} {match}, got {len(found)}"
        return found[0]

    def find(self, label: str) -> Declared:
        """The button whose LIVE label is ``label`` (labels can be dynamic).

        Named ``find`` rather than ``button`` because :meth:`button` is part of the
        Panel surface a section calls.
        """
        found = [w for w in self.of("button") if w.label == label]
        assert len(found) == 1, (f"expected one button {label!r}, got "
                                 f"{[w.label for w in self.of('button')]}")
        return found[0]

    def visible_buttons(self) -> list[str]:
        return [w.label for w in self.of("button") if w.shown]

    def screen(self) -> str:
        """Everything the user can currently read, as one blob.

        Assertions read as "the panel says X in this state", which is the level the
        UX contract lives at — not which widget index holds it.
        """
        lines = []
        for widget in self.widgets:
            if not widget.shown:
                continue
            if widget.kind == "text":
                lines.append(widget.rendered())
            elif widget.kind in ("button", "checkbox", "section"):
                lines.append(widget.label or str(widget.info.get("caption", "")))
            elif widget.kind == "log":
                lines.extend(str(line) for line in (widget.info["getter"]() or ()))
        return "\n".join(lines)

    # ---- the Panel surface a section may use -------------------------

    @contextlib.contextmanager
    def section(self, title: str, **binds):
        section = Declared("section", title=title, label=title, binds=binds)
        self.widgets.append(section)
        first = len(self.widgets)
        yield
        for child in self.widgets[first:]:
            visible = child.binds.get("visible")
            child.binds["visible"] = lambda gate=visible: section.shown and (gate is None or gate())

    def text(self, value="", *, lines=1, mode="wrap", visible=None) -> None:
        self.widgets.append(Declared("text", value=value, lines=lines, mode=mode,
                                     binds={} if visible is None
                                     else {"visible": visible}))

    def spacer(self) -> None:
        self.widgets.append(Declared("spacer"))

    def log(self, lines_getter, *, lines=8) -> None:
        self.widgets.append(Declared("log", getter=lines_getter, lines=lines))

    def progress(self, fraction, *, caption=None, **binds) -> None:
        if caption is not None:
            self.text(caption)
        self.widgets.append(Declared("progress",
                                     binds={"fraction": fraction, **binds}))

    def button(self, label, on_click, **binds) -> None:
        if not isinstance(label, str):
            binds.setdefault("label", label)
        self.widgets.append(Declared(
            "button", label=label if isinstance(label, str) else label(),
            on_click=on_click, binds=binds))

    def _caption_row(self, caption, binds) -> None:
        """Mirror Panel._caption_row: a caption inherits its control's visibility."""
        if caption:
            self.text(caption, visible=binds.get("visible"))

    def combo(self, caption, items, value, on_change, **binds) -> None:
        self._caption_row(caption, binds)
        self.widgets.append(Declared("combo", caption=caption, items=items,
                                     value=value, on_change=on_change, binds=binds))

    def checkbox(self, caption, value, on_change, **binds) -> None:
        self.widgets.append(Declared("checkbox", caption=caption, label=caption,
                                     value=value, on_change=on_change, binds=binds))

    def input_text(self, value, on_change, **binds) -> None:
        self.widgets.append(Declared("input_text", value=value,
                                     on_change=on_change, binds=binds))

    def slider_float(self, caption, value, minimum, maximum, on_change,
                     fmt="%.3f", **binds) -> None:
        self.widgets.append(Declared("slider", value=value, on_change=on_change,
                                     binds=binds))

    slider_int = slider_float

    def drag_int(self, caption, value, on_change, *, minimum=0, maximum=0,
                 speed=1.0, fmt="%d", **binds) -> None:
        self._caption_row(caption, binds)
        self.widgets.append(Declared("drag_int", value=value, on_change=on_change,
                                     minimum=minimum, maximum=maximum, binds=binds))

    def drag_float(self, caption, value, on_change, *, minimum=0.0, maximum=0.0,
                   speed=0.01, fmt="%.3f", **binds) -> None:
        self._caption_row(caption, binds)
        self.widgets.append(Declared("drag_float", value=value, on_change=on_change,
                                     minimum=minimum, maximum=maximum, binds=binds))


class Recorder:
    """Records which action callbacks a panel fired, for wiring assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def action(self, name: str):
        def fire(*args):
            self.calls.append((name, args[0] if args else None))
        return fire

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]
