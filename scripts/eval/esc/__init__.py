# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Expected Safe Changeability (ESC): scores a codebase by how well it absorbs unseen change.

ESC = 100 * C^0.40 * E^0.15 * U^0.15 * L^0.15 * T^0.10 * S^0.05, scored only after hard behavioural gates pass.
"""

from __future__ import annotations

__version__ = "0.1.0"
