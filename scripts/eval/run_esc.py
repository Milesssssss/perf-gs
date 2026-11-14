#!/usr/bin/env python
# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Entry point for the ESC harness: puts ``scripts/eval`` on ``sys.path`` and runs :func:`esc.cli.main`.

    python scripts/eval/run_esc.py analyze
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from esc.cli import main  # noqa: E402  (path set above)

if __name__ == "__main__":
    raise SystemExit(main())
