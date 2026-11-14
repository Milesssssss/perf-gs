# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Turn setup/reconstruction output into a determinate progress fraction, keyed on the explicit
``[stage] <name>`` lines our own code emits so rewording a log line cannot break the bar."""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# "[12/300]", "[180/360 MB]" — colmap's own counters, and our download's.
_COUNTER = re.compile(r"\[(\d+)\s*/\s*(\d+)")


@dataclass(frozen=True)
class Stage:
    marker: str
    label: str
    start: float
    end: float

    def at(self, within: float) -> float:
        return self.start + (self.end - self.start) * min(1.0, max(0.0, within))


@dataclass(frozen=True)
class ParsedProgress:
    label: str
    fraction: float


# ReconTools.setup(): colmap's CUDA zip (~360 MB) dwarfs glomap's (~70 MB), hence the wider span.
SETUP_STAGES: tuple[Stage, ...] = (
    Stage("colmap", "installing colmap", 0.0, 0.75),
    Stage("glomap", "installing glomap", 0.75, 0.98),
    Stage("done", "tools ready", 1.0, 1.0),
)

RECON_STAGES: tuple[Stage, ...] = (
    Stage("start", "starting", 0.0, 0.02),
    Stage("features", "extracting features", 0.02, 0.30),
    Stage("matching", "matching images", 0.30, 0.55),
    Stage("mapping", "mapping (SfM)", 0.55, 0.85),
    Stage("undistort", "undistorting", 0.85, 0.93),
    Stage("arrange", "writing dataset", 0.93, 0.99),
    Stage("done", "done", 1.0, 1.0),
)


class ProgressParser:
    """Feed it output lines; it tracks the current stage and interpolates, never rewinding."""

    def __init__(self, stages: Iterable[Stage]) -> None:
        self._stages = tuple(stages)
        self._index = -1
        self._fraction = 0.0
        self._label = ""

    @property
    def progress(self) -> ParsedProgress:
        return ParsedProgress(self._label, self._fraction)

    def feed(self, line: str) -> ParsedProgress | None:
        """Update from one line; the new progress, or None if it did not move."""
        text = str(line)
        changed = False
        marker = text.partition("[stage] ")[2].strip()
        if marker:
            index = next((i for i, s in enumerate(self._stages)
                          if s.marker == marker), None)
            if index is not None and index > self._index:
                self._index, self._label = index, self._stages[index].label
                changed = True
                changed |= self._advance(self._stages[index].start)
        elif self._index >= 0:
            counter = _COUNTER.search(text)
            if counter is not None and int(counter.group(2)) > 0:
                changed |= self._advance(self._stages[self._index].at(
                    int(counter.group(1)) / int(counter.group(2))))
        return self.progress if changed else None

    def _advance(self, fraction: float) -> bool:
        if fraction > self._fraction + 1e-6:
            self._fraction = min(1.0, fraction)
            return True
        return False


def setup_parser() -> ProgressParser:
    return ProgressParser(SETUP_STAGES)


def reconstruction_parser() -> ProgressParser:
    return ProgressParser(RECON_STAGES)
