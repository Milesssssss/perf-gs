"""Thin entry point for the v2 trainer.

All training logic lives in :mod:`trainer.training.app` (extracted from this
script). This entry is kept so ``python scripts/running/v2_train.py ...`` keeps
working.

The frontend utilities app.py imports by name are split across sibling
``scripts/`` folders: ``_v2_profiling`` / ``_v2_sampling`` live in
``scripts/running/`` (this folder, already on ``sys.path[0]`` when run directly)
while ``bin_viz``, ``popstat`` and ``densify_diag`` live in ``scripts/debug/``.
The shim below adds those sibling folders to ``sys.path`` BEFORE importing the
app so the flat imports resolve regardless of which folder each helper lives in.

``build_parser`` is re-exported so ``from v2_train import build_parser`` (and the
CLI-freeze tests) keep resolving.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent  # scripts/
for _sub in ("running", "debug", "plots"):
    _p = str(_SCRIPTS / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

from trainer.training.app import main  # noqa: E402
from trainer.training.cli import build_parser  # noqa: E402

__all__ = ["main", "build_parser"]


if __name__ == "__main__":
    main()
