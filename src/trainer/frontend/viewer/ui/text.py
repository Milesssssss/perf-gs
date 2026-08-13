"""Text metrics for the panel font: measure, wrap, elide.

``slangpy.ui.Text`` renders with ``ImGui::TextUnformatted`` and sgl exposes
neither ``PushTextWrapPos`` nor ``CalcTextSize``, so a long line is silently
CLIPPED at the panel edge (the "messages break down" symptom: half a path, half
a status line). Everything the panel shows is therefore wrapped/elided here,
before it reaches a widget.

:data:`ADVANCES` holds the per-character advance in pixels for the bundled panel
font at its default size, measured offscreen: each glyph was rendered n and 2n
times and the advance taken as ``(width(2n) - width(n)) / n``, which cancels the
left/right bearings exactly. The font is PROPORTIONAL (``l`` 3.3px, ``m`` 13.0px)
so a character count is not a usable proxy for width.
"""
from __future__ import annotations

# Measured advance in pixels per character (see module docstring).
ADVANCES: dict[str, float] = {
    ' ': 3.21, '!': 3.21, '"': 4.58, '#': 8.54, '$': 7.58, '%': 10.21,
    '&': 8.21, "'": 2.50, '(': 4.04, ')': 4.08, '*': 4.75, '+': 7.08,
    ',': 2.62, '-': 4.71, '.': 2.62, '/': 4.12,
    '0': 8.17, '1': 4.42, '2': 6.96, '3': 6.96, '4': 8.17, '5': 6.96,
    '6': 7.50, '7': 7.25, '8': 7.88, '9': 7.46,
    ':': 2.62, ';': 2.62, '<': 7.08, '=': 7.08, '>': 7.08, '?': 6.96,
    '@': 12.71,
    'A': 8.83, 'B': 9.29, 'C': 8.83, 'D': 10.17, 'E': 8.25, 'F': 7.79,
    'G': 9.50, 'H': 10.00, 'I': 3.71, 'J': 6.17, 'K': 8.75, 'L': 7.25,
    'M': 11.75, 'N': 10.00, 'O': 10.29, 'P': 8.83, 'Q': 10.33, 'R': 8.92,
    'S': 7.58, 'T': 7.04, 'U': 9.75, 'V': 8.62, 'W': 13.67, 'X': 8.08,
    'Y': 7.79, 'Z': 8.04,
    '[': 3.92, '\\': 4.08, ']': 3.92, '^': 7.12, '_': 6.17, '`': 7.42,
    'a': 7.25, 'b': 8.33, 'c': 6.92, 'd': 8.33, 'e': 7.46, 'f': 4.17,
    'g': 8.42, 'h': 8.33, 'i': 3.29, 'j': 3.38, 'k': 7.38, 'l': 3.29,
    'm': 13.04, 'n': 8.33, 'o': 7.71, 'p': 8.33, 'q': 8.33, 'r': 4.96,
    's': 6.00, 't': 5.00, 'u': 8.25, 'v': 6.67, 'w': 10.79, 'x': 6.54,
    'y': 6.67, 'z': 6.29,
    '{': 4.08, '|': 3.62, '}': 4.12, '~': 7.08,
}

# Unmeasured (non-ASCII) glyphs: assume slightly wider than the mean lowercase
# advance (7.06) so an unusual character makes us wrap EARLY, never late.
DEFAULT_ADVANCE = 8.0

ELLIPSIS = "..."

# Break points inside an unbreakable-looking token (paths, flags, numbers). A
# break is taken AFTER these, so "datasets/360_v2/bicycle" splits at the
# separators instead of mid-word.
_SOFT_BREAK_AFTER = "/\\-_,;:=&"


def width(text: str) -> float:
    """Rendered width of ``text`` in pixels, in the panel font."""
    total = 0.0
    for ch in text:
        total += ADVANCES.get(ch, DEFAULT_ADVANCE)
    return total


def fits(text: str, max_width: float) -> bool:
    return width(text) <= max_width


def truncate(text: str, max_width: float) -> str:
    """Longest prefix of ``text`` that fits in ``max_width`` (no ellipsis)."""
    if max_width <= 0.0:
        return ""
    total = 0.0
    for i, ch in enumerate(text):
        total += ADVANCES.get(ch, DEFAULT_ADVANCE)
        if total > max_width:
            return text[:i]
    return text


def elide_end(text: str, max_width: float) -> str:
    """Drop the TAIL, ending with ``...`` — for text whose start identifies it."""
    if fits(text, max_width):
        return text
    keep = truncate(text, max_width - width(ELLIPSIS))
    return keep.rstrip() + ELLIPSIS if keep else ELLIPSIS


def elide_start(text: str, max_width: float) -> str:
    """Drop the HEAD, starting with ``...`` — for paths, where the leaf matters
    (``...datasets/360_v2/bicycle``)."""
    if fits(text, max_width):
        return text
    budget = max_width - width(ELLIPSIS)
    if budget <= 0.0:
        return ELLIPSIS
    total = 0.0
    for i in range(len(text) - 1, -1, -1):
        total += ADVANCES.get(text[i], DEFAULT_ADVANCE)
        if total > budget:
            return ELLIPSIS + text[i + 1:]
    return text


def elide_middle(text: str, max_width: float) -> str:
    """Keep both ends, ``...`` in the middle — for long labels/identifiers."""
    if fits(text, max_width):
        return text
    budget = max_width - width(ELLIPSIS)
    if budget <= 0.0:
        return ELLIPSIS
    head, tail, used = 0, 0, 0.0
    # Grow from both ends alternately, biasing the tail (it usually carries the
    # distinguishing part, e.g. a filename).
    while head + tail < len(text):
        take_tail = tail <= head
        ch = text[len(text) - 1 - tail] if take_tail else text[head]
        used += ADVANCES.get(ch, DEFAULT_ADVANCE)
        if used > budget:
            break
        if take_tail:
            tail += 1
        else:
            head += 1
    return text[:head] + ELLIPSIS + (text[len(text) - tail:] if tail else "")


def wrap(text: str, max_width: float, max_lines: int | None = None) -> list[str]:
    """Word-wrap ``text`` to ``max_width`` pixels.

    Wraps on spaces first, then inside over-long tokens at path/flag separators,
    then character-by-character, so a 300px panel can never clip a line. When
    ``max_lines`` is given the result is truncated to that many lines and the
    last one is elided with ``...`` so nothing is dropped SILENTLY.
    """
    if max_width <= 0.0:
        return [""]
    lines: list[str] = []
    for paragraph in text.split("\n"):
        lines.extend(_wrap_one(paragraph, max_width))
    if not lines:
        lines = [""]
    if max_lines is not None and len(lines) > max_lines:
        if max_lines <= 0:
            return []
        kept = lines[:max_lines]
        rest = " ".join(lines[max_lines - 1:]).strip()
        kept[-1] = elide_end(rest, max_width)
        return kept
    return lines


def _wrap_one(text: str, max_width: float) -> list[str]:
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if fits(candidate, max_width):
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        # The word alone must be split (long path, long command line).
        if fits(word, max_width):
            current = word
            continue
        chunks = _split_token(word, max_width)
        lines.extend(chunks[:-1])          # full lines
        current = chunks[-1] if chunks else ""   # remainder continues this line
    if current:
        lines.append(current)
    return lines


def _split_token(token: str, max_width: float) -> list[str]:
    """Split one over-long token into <= max_width pieces, preferring a break
    after a path/flag separator and falling back to a hard character break."""
    pieces: list[str] = []
    start, last_break, total = 0, -1, 0.0
    for i, ch in enumerate(token):
        total += ADVANCES.get(ch, DEFAULT_ADVANCE)
        if total > max_width:
            cut = last_break + 1 if last_break > start else i
            cut = max(cut, start + 1)          # always make progress
            pieces.append(token[start:cut])
            start, last_break = cut, -1
            total = width(token[start:i + 1])
        if ch in _SOFT_BREAK_AFTER:
            last_break = i
    if start < len(token):
        pieces.append(token[start:])
    return pieces
