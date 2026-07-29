"""Pytest configuration for the trainer test suite.

Adds the repo ``scripts/`` directory to ``sys.path`` so the characterization
tests can import the training-frontend helpers (``build_parser``,
``CameraSampler``, ...) that currently live under ``scripts/`` rather than the
installed ``trainer`` package. This is TEST-ONLY path setup — it changes no
shipped behavior. The frontend now lives in the ``trainer.frontend`` package;
these imports move to the package and this shim shrinks / disappears.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
# The frontend helpers imported by the tests + trainer are split across
# scripts/ subfolders (running/ = v2_train, _v2_sampling, _v2_profiling;
# debug/ = bin_viz, densify_diag, popstat). Append each (not insert) so they
# resolve by name without shadowing any installed/stdlib module.
for _sub in ("running", "debug", "plots"):
    _p = _SCRIPTS_DIR / _sub
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))
