# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Reusable panel components — the UI's visual language, in one place.

Every section is built from these, so a status badge, a metric row or an inline
error looks and behaves the same wherever it appears. Adding a readout should mean
picking a component, never hand-rolling another slightly different one.

WHAT THIS TOOLKIT CAN AND CANNOT DO. ``slangpy.ui`` exposes Text, Button, CheckBox,
ComboBox, Drag/Slider, ProgressBar, InputText, Group and Window — and on Text only
``text``/``visible``/``enabled``. There is no color, no font weight, no separator,
no horizontal layout, no icon set and no modal dialog. So hierarchy here is built
from what does exist:

  * CASE and bracketing carry emphasis: ``[ RUNNING ]`` reads as a badge without a
    colour to make it one.
  * Indentation (two spaces) marks detail under a heading.
  * A fixed label column aligns values (see :data:`LABEL_W`) — the font is
    proportional, so this is a close column rather than a table, which is enough
    for scanning and wraps cleanly.
  * Blank lines do the grouping a separator would, and are used sparingly: a rule
    between every row is noise, not hierarchy.
  * Glyphs stay inside ASCII and Latin-1. The ImGui font atlas is built by sgl and
    its coverage above U+00FF is not guaranteed, so a tempting checkmark could
    render as a box on someone's machine. ``+ - ! x`` and ``·`` (U+00B7, Latin-1)
    carry the same meaning and always draw. :data:`GLYPHS` is the single place to
    change this if a richer atlas is ever confirmed.

Indeterminate work animates through a growing ellipsis rather than a spinner:
proportional-font spinners jitter, and ``.`` ``..`` ``...`` reads as "still going"
at a glance.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from trainer.frontend.viewer.session import Activity, View
from trainer.frontend.viewer.ui.panel import ELIDE_MIDDLE, ELIDE_START, Panel

# The label column. Wide enough for "densification" without pushing values off a
# ~420px panel.
LABEL_W = 16

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
    """A growing ellipsis for indeterminate work (see the module docstring)."""
    return _ELLIPSIS[int(clock() / (period / len(_ELLIPSIS))) % len(_ELLIPSIS)]


def badge(text: str) -> str:
    """``[ RUNNING ]`` — the strongest emphasis this toolkit allows."""
    return f"[ {text} ]"


def row(label: str, value: str, *, indent: int = 0) -> str:
    """One ``label   value`` line, value in the aligned column.

    A label wider than the column keeps a two-space gap rather than butting
    straight into its value ("backward pixel stride1").
    """
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
    """An error where its cause is, not in the console.

    Two blocks: a headline that names what failed, and the reason wrapped over as
    many lines as it needs (a path or an exception message is usually long).
    """
    panel.text(lambda: f"{GLYPHS['error']} {title}", visible=lambda: bool(message()))
    note(panel, message, lines=3, visible=lambda: bool(message()))


def activity_block(panel: Panel, activity: Callable[[], Activity], *,
                   visible=None) -> None:
    """The one busy readout: caption plus a bar when a fraction is knowable.

    An indeterminate operation gets the caption and the ellipsis but NO bar — a bar
    parked at 0% reads as stuck, which is the opposite of the reassurance a busy
    state is for.
    """
    shown = (lambda: activity().active) if visible is None else visible

    def caption() -> str:
        current = activity()
        return (f"  {current.caption()}" if current.determinate
                else f"  {current.caption()}{ellipsis()}")

    panel.text(caption, lines=2, visible=shown)
    panel.progress(lambda: activity().fraction or 0.0,
                   visible=lambda: activity().active and activity().determinate)


def bucket(values: Sequence[float], rows: int) -> list[float]:
    """Reduce a series to ``rows`` bucket means, oldest first.

    Buckets are contiguous index ranges, so a long window compresses uniformly
    instead of dropping samples — a spike survives as a raised bucket rather than
    disappearing between two picked points.
    """
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
    """A series as horizontal bars, oldest first, autoscaled to its own range.

    WHY BARS AND NOT A LINE. The panel font is PROPORTIONAL (see :mod:`.text`:
    ``' '`` is 3.21px, ``'#'`` 8.54px), so a character grid does not line up —
    a plotted curve would shear left and right by whatever mix of glyphs each row
    happened to contain. A bar built from ONE repeated glyph has width exactly
    proportional to its length, so bars remain comparable to each other in any
    font. That is the only chart shape this toolkit can draw honestly.

    Autoscaled to ``[min, max]`` OF THE WINDOW, because a loss curve late in
    training moves in the third decimal and a zero-based axis would show a flat
    wall. The caller must therefore print the range (:func:`bar_scale`) — an empty
    bar means "window minimum", not zero.
    """
    series = bucket(values, rows)
    if not series:
        return []
    lo, hi = min(series), max(series)
    span = hi - lo
    out = []
    for value in series:
        # A flat window gets a single cell: something visible, and honestly not a
        # magnitude.
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
    """Render :func:`bars` into a fixed pool of rows that pull every frame.

    The pool is allocated once (widget order is build order) and unused rows hide
    themselves, so a chart that has not filled up yet does not reserve blank space.
    """
    def line(index: int) -> str:
        drawn = bars(series(), rows=rows, cells=cells)
        return f"  {drawn[index]}" if index < len(drawn) else ""

    for index in range(rows):
        panel.text(lambda i=index: line(i),
                   visible=(lambda i=index: line(i) != "") if visible is None
                   else (lambda i=index: bool(visible()) and line(i) != ""))


def toasts(panel: Panel, view: View, *, slots: int = 3) -> None:
    """Transient acknowledgements, newest last.

    A fixed pool of rows so the panel does not reflow as notices come and go, and
    hidden rows take no space (see :class:`~..panel._TextBlock`).
    """
    for index in range(slots):
        panel.text(lambda i=index: notice_line(view.notices, i, slots),
                   visible=lambda i=index: notice_line(view.notices, i, slots) != "")


def notice_line(notices: Sequence, index: int, slots: int) -> str:
    """Row ``index`` of a ``slots``-row pool, filled from the BOTTOM.

    Bottom-filled so the newest notice is always on the last row and posting one
    does not shuffle the others up the panel.
    """
    recent = list(notices)[-slots:]
    offset = index - (slots - len(recent))
    return recent[offset].line() if 0 <= offset < len(recent) else ""


def confirm_block(panel: Panel, view: View, key: str, *,
                  on_confirm: Callable[[], None], on_alt: Callable[[], None],
                  on_cancel: Callable[[], None]) -> None:
    """An inline confirmation for the section named by ``key``.

    Inline rather than modal: this toolkit has no dialog, and keeping the question
    next to the thing it would destroy is better anyway. Only shown when a pending
    confirmation belongs to this section.
    """
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


def drop_target(panel: Panel, view: View, *, on_browse=None) -> None:
    """The empty-state affordance.

    There is no drag-enter event in this windowing layer (only ``on_drop_files``),
    so the target cannot light up while a folder hovers. It compensates by being
    unmissable when idle and by reacting the instant a drop lands — see
    ``Session.dataset_inspecting``, which publishes INSPECTING before anything is
    read from disk.
    """
    panel.spacer()
    panel.text("      DROP A DATASET HERE")
    panel.text("   image folder or COLMAP data")
    panel.spacer()
    if on_browse is not None:
        panel.button("Browse...", on_browse, enabled=lambda: view.can_drop)
    note(panel, "A folder of images, or one with sparse/ + images/.", lines=2)


def section_summary(*parts: str) -> str:
    """``DENSIFICATION   structure_aware · 100-1900`` — a collapsed section that
    still says what it holds."""
    kept = [p for p in parts if p]
    return f"  {GLYPHS['bullet']} ".join(kept) if kept else ""


def titled(title: str, summary: Callable[[], str]) -> Callable[[], str]:
    """Section label binding: the title, plus its live summary."""
    return lambda: f"{title}      {summary()}" if summary() else title


def actions(panel: Panel, primary: Sequence[tuple], secondary: Sequence[tuple] = ()) -> None:
    """A primary action row and, below a blank line, the secondary ones.

    Each entry is ``(label, callback, enabled_getter[, visible_getter])``. The
    spacing IS the hierarchy: with no colour available, distance from the primary
    action is what stops a destructive button reading as an equal choice.
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
    """A filesystem path: elided at the START so the leaf (which identifies it)
    survives, and never wrapped mid-segment."""
    panel.text(lambda: f"{label}", visible=visible)
    panel.text(lambda: f"  {path()}", mode=ELIDE_START, visible=visible)


def name_field(panel: Panel, label: str, value: Callable[[], str],
               on_change: Callable[[str], None], *, enabled=None,
               visible=None) -> None:
    """An editable short string (the export filename).

    ``InputText`` is the only text entry this toolkit has; the caption goes on the
    line above for the same reason every other control's does (see the panel
    module docstring).
    """
    panel.text(label, visible=visible)
    panel.input_text(value, on_change, enabled=enabled, visible=visible)


def elide_middle_row(panel: Panel, label: str, value: Callable[[], str],
                     *, visible=None) -> None:
    """A long single-line value that must keep both ends (a run directory)."""
    panel.text(lambda: row(label, str(value())), mode=ELIDE_MIDDLE, visible=visible)
