# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Thin entry point for the v2 trainer.

All training logic lives in :mod:`trainer.frontend.app` (extracted from this

"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent  # scripts/
for _sub in ("running", "debug", "plots"):
    _p = str(_SCRIPTS / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

from trainer.frontend.app import main 
from trainer.frontend.cli import build_parser as build_parser


if __name__ == "__main__":
    main()
