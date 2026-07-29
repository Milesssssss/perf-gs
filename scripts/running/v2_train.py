from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent  # scripts/
for _sub in ("running", "debug", "plots"):
    _p = str(_SCRIPTS / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

from trainer.frontend.app import main  # noqa: E402
from trainer.frontend.cli import build_parser  # noqa: E402

__all__ = ["main", "build_parser"]


if __name__ == "__main__":
    main()
