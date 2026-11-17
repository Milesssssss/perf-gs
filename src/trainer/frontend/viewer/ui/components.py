# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Reusable panel components: the UI's visual language, in one place.

``slangpy.ui`` has no colour, weight or separator: case, brackets, indentation and blank lines carry hierarchy.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from trainer.frontend.viewer.session import Activity, View
from trainer.frontend.viewer.ui.panel import ELIDE_START, Panel

# Label column: wide enough for "densification" without pushing values off a ~420px panel.
LABEL_W = 16

# ASCII/Latin-1 only: sgl's font atlas may not cover glyphs above U+00FF, which would draw as boxes.
GLYPHS = {
    "ok": "+",
    "info": "-",
    "warn": "!",
    "error": "x",
    "bullet": "·",     # Latin-1 middle dot
    "arrow": "->",
}

_ELLIPSIS = ("", ".", "..", "...")


def ellipsis(period: float = 1.2, clock: Callable[[], float] = time.monotonic) -> str:
    """A growing ellipsis for indeterminate work; proportional-font spinners would jitter."""
    return _ELLIPSIS[int(clock() / (period / len(_ELLIPSIS))) % len(_ELLIPSIS)]


def badge(text: str) -> str:
    """``[ RUNNING ]`` — the strongest emphasis this toolkit allows."""
    return f"[ {text} ]"


def row(label: str, value: str, *, indent: int = 0) -> str:
    """One ``label   value`` line in the aligned column; an over-wide label still keeps a two-space gap."""
    pad = " " * indent
    width = max(1, LABEL_W - indent)
    text = f"{pad}{label:<{width}s}"
    if not text.endswith("  "):
        text += "  "
    return f"{text}{value}"


def metric(panel: Panel, label: str, value: Callable[[], str], *, indent: int = 0,
           visible=None) -> None:
    """A label/value readout that re-reads ``value`` every frame."""
    panel.text(lambda: row(label, str(value()), indent=indent), visible=visible)


def note(panel: Panel, text, *, lines: int = 2, visible=None) -> None:
    """Secondary explanatory text, indented under whatever it belongs to."""
    getter = text if callable(text) else (lambda: text)
    panel.text(lambda: f"  {getter()}", lines=lines, visible=visible)


def status_line(panel: Panel, state: Callable[[], str], detail: Callable[[], str],
                *, visible=None) -> None:
    """``[ STATE ]  detail`` — the header of any section that has a lifecycle."""
    panel.text(lambda: f"{badge(state())}  {detail()}", lines=2, visible=visible)


def inline_error(panel: Panel, message: Callable[[], str], *,
                 title: str = "Failed") -> None:
    """An error shown where its cause is, not in the console: a headline plus the wrapped reason."""
    panel.text(lambda: f"{GLYPHS['error']} {title}", visible=lambda: bool(message()))
    note(panel, message, lines=3, visible=lambda: bool(message()))


def activity_block(panel: Panel, activity: Callable[[], Activity], *,
                   visible=None) -> None:
    """The one busy readout: caption, plus a bar only when determinate (a bar parked at 0% reads as stuck)."""
    shown = (lambda: activity().active) if visible is None else visible

    def caption() -> str:
        current = activity()
        return (f"  {current.caption()}" if current.determinate
                else f"  {current.caption()}{ellipsis()}")

    panel.text(caption, lines=2, visible=shown)
    panel.progress(lambda: activity().fraction or 0.0,
                   visible=lambda: activity().active and activity().determinate)


def bucket(values: Sequence[float], rows: int) -> list[float]:
    """Reduce a series to ``rows`` contiguous-bucket means, oldest first, so a spike survives compression."""
    series = list(values)
    if not series or rows <= 0:
        return []
    if len(series) <= rows:
        return series
    out: list[float] = []
    for index in range(rows):
        lo = (index * len(series)) // rows
        hi = max(lo + 1, ((index + 1) * len(series)) // rows)
        chunk = series[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def bars(values: Sequence[float], *, rows: int = 10, cells: int = 24,
         glyph: str = "#") -> list[str]:
    """A series as one-glyph horizontal bars (a proportional font can't plot a line), oldest first.

    Scaled to the window's range, so callers print :func:`bar_scale`: an empty bar means the minimum, not zero.
    """
    series = bucket(values, rows)
    if not series:
        return []
    lo, hi = min(series), max(series)
    span = hi - lo
    out = []
    for value in series:
        # A flat window gets one cell: visible, but honestly not a magnitude.
        filled = 1 if span <= 0 else int(round(cells * (value - lo) / span))
        out.append(glyph * max(0, min(cells, filled)))
    return out


def bar_scale(values: Sequence[float], *, fmt: str = "%.5f") -> str:
    """The window's range, so the bars above it mean something."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    return (f"scale {fmt % lo} .. {fmt % hi}" if hi > lo
            else f"flat at {fmt % lo}")


def chart(panel: Panel, series: Callable[[], Sequence[float]], *, rows: int = 10,
          cells: int = 24, visible=None) -> None:
    """Render :func:`bars` into a fixed row pool built once (widget order is build order); unused rows hide."""
    def line(index: int) -> str:
        drawn = bars(series(), rows=rows, cells=cells)
        return f"  {drawn[index]}" if index < len(drawn) else ""

    for index in range(rows):
        panel.text(lambda i=index: line(i),
                   visible=(lambda i=index: line(i) != "") if visible is None
                   else (lambda i=index: bool(visible()) and line(i) != ""))


def toasts(panel: Panel, view: View, *, slots: int = 3) -> None:
    """Transient acknowledgements, newest last, in a fixed row pool so the panel doesn't reflow."""
    for index in range(slots):
        panel.text(lambda i=index: notice_line(view.notices, i, slots),
                   visible=lambda i=index: notice_line(view.notices, i, slots) != "")


def notice_line(notices: Sequence, index: int, slots: int) -> str:
    """Row ``index`` of a ``slots``-row pool, bottom-filled so the newest notice is always on the last row."""
    recent = list(notices)[-slots:]
    offset = index - (slots - len(recent))
    return recent[offset].line() if 0 <= offset < len(recent) else ""


def confirm_block(panel: Panel, view: View, key: str, *,
                  on_confirm: Callable[[], None], on_alt: Callable[[], None],
                  on_cancel: Callable[[], None]) -> None:
    """An inline confirmation (the toolkit has no dialog), shown while one is pending for section ``key``."""
    def pending():
        confirm = view.confirm
        return confirm if confirm is not None and confirm.key == key else None

    def shown() -> bool:
        return pending() is not None

    panel.text(lambda: f"{GLYPHS['warn']} {pending().title if pending() else ''}",
               lines=2, visible=shown)
    note(panel, lambda: pending().body if pending() else "", lines=3, visible=shown)
    panel.button(lambda: pending().confirm_label if pending() else "Continue",
                 on_confirm, visible=shown)
    panel.button(lambda: pending().alt_label if pending() else "",
                 on_alt, visible=lambda: bool(pending() and pending().alt_label))
    panel.button("Cancel", on_cancel, visible=shown)


def section_summary(*parts: str) -> str:
    """``DENSIFICATION   structure_aware · 100-1900``: a collapsed section that still says what it holds."""
    kept = [p for p in parts if p]
    return f"  {GLYPHS['bullet']} ".join(kept) if kept else ""


def titled(title: str, summary: Callable[[], str]) -> Callable[[], str]:
    """Section label binding: the title, plus its live summary."""
    return lambda: f"{title}      {summary()}" if summary() else title


def actions(panel: Panel, primary: Sequence[tuple], secondary: Sequence[tuple] = ()) -> None:
    """A primary action row and, below a blank line, the secondary ones (with no colour, distance is hierarchy).

    Each entry is ``(label, callback, enabled_getter[, visible_getter])``.
    """
    for entry in primary:
        _action(panel, entry)
    if secondary:
        panel.spacer()
        for entry in secondary:
            _action(panel, entry)


def _action(panel: Panel, entry: tuple) -> None:
    label, callback, enabled, *rest = (*entry, None)[:4]
    panel.button(label, callback, enabled=enabled,
                 **({"visible": rest[0]} if rest and rest[0] is not None else {}))


def path_row(panel: Panel, label: str, path: Callable[[], str], *, visible=None) -> None:
    """A filesystem path, elided at the start so the identifying leaf survives."""
    panel.text(lambda: f"{label}", visible=visible)
    panel.text(lambda: f"  {path()}", mode=ELIDE_START, visible=visible)


def name_field(panel: Panel, label: str, value: Callable[[], str],
               on_change: Callable[[str], None], *, enabled=None,
               visible=None) -> None:
    """An editable short string (the export filename), captioned on the line above like every control."""
    panel.text(label, visible=visible)
    panel.input_text(value, on_change, enabled=enabled, visible=visible)
