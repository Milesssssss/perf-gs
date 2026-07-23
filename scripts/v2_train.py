"""Thin entry point for the v2 trainer.

All training logic lives in :mod:`trainer.training.app` (extracted from this
script). This entry is kept so ``python scripts/v2_train.py ...`` keeps working:
running it as a script puts ``scripts/`` on ``sys.path[0]``, which is where the
sibling frontend utilities app.py imports (``bin_viz``, ``_v2_profiling``,
``_v2_sampling``, ``popstat``, ``densify_diag``) currently live.

``build_parser`` is re-exported so ``from v2_train import build_parser`` (and the
CLI-freeze tests) keep resolving.
"""

from trainer.training.app import main
from trainer.training.cli import build_parser

__all__ = ["main", "build_parser"]


if __name__ == "__main__":
    main()
