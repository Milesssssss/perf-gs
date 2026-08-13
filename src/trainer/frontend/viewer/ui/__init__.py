"""Viewer UI toolkit: a docked, state-bound panel plus input mapping.

  * :mod:`.text`     — measured font metrics; wrap/elide so nothing is clipped
  * :mod:`.dock`     — edge docking + user-resizable thickness (the resize fix)
  * :mod:`.panel`    — declarative widgets whose properties BIND to state getters
  * :mod:`.sections` — the viewer's actual sections, as independent builders
  * :mod:`.input`    — hold/press key bindings that also document themselves

The render loop only samples state and calls ``panel.layout`` / ``panel.refresh``;
all UI composition and policy lives here.
"""
from __future__ import annotations

from trainer.frontend.viewer.ui.dock import Dock, Rect
from trainer.frontend.viewer.ui.input import InputMap
from trainer.frontend.viewer.ui.panel import (
    ELIDE_END,
    ELIDE_MIDDLE,
    ELIDE_START,
    WRAP,
    Panel,
)

__all__ = ["Dock", "Rect", "InputMap", "Panel",
           "WRAP", "ELIDE_START", "ELIDE_END", "ELIDE_MIDDLE"]
