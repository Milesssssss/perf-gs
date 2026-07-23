"""Pytest configuration for the trainer test suite.

Adds the repo ``scripts/`` directory to ``sys.path`` so the characterization
tests can import the training-frontend helpers (``build_parser``,
``CameraSampler``, ...) that currently live under ``scripts/`` rather than the
installed ``trainer`` package. This is TEST-ONLY path setup — it changes no
shipped behavior. As the frontend is extracted into the ``trainer.training``
package (later refactor slices), these imports move to the package and this
shim shrinks / disappears.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if _SCRIPTS_DIR.is_dir() and str(_SCRIPTS_DIR) not in sys.path:
    # Append (not insert) so scripts/ never shadows an installed/stdlib module;
    # the frontend helpers imported here (v2_train, _v2_sampling) are unique to
    # scripts/, so being last on the path is sufficient to resolve them.
    sys.path.append(str(_SCRIPTS_DIR))
