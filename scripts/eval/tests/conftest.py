"""Pytest path setup for the ESC harness self-tests.

Adds ``scripts/eval`` to ``sys.path`` so ``import esc`` resolves. Kept separate
from the trainer's own ``tests/`` suite so the two never interfere.
"""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).resolve().parent.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
