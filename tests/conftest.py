# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Make the checkout's GPU capture and splat-importance scripts importable by their tests."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

for _sub in ("running", "debug"):
    _p = _SCRIPTS_DIR / _sub
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))
