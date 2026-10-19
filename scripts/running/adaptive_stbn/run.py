# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Path-setting shim so the suite runs from anywhere.

    .venv/Scripts/python.exe scripts/running/adaptive_stbn/run.py --stages sampling

``scripts/running`` is not an installed package, so ``-m adaptive_stbn`` only works
with that directory already on sys.path. This adds it (and the repo's ``src``, for
the same reason the other runners do) and delegates.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_RUNNING = _HERE.parents[1]
_REPO = _HERE.parents[3]
for _path in (str(_RUNNING), str(_REPO / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from adaptive_stbn.__main__ import main                          # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
