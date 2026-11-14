# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Paper-experiment suite for the STBN sparse-supervision study.

Driven by ``scripts/running/eval_stbn_ablation.py``.
"""
from __future__ import annotations

OUTPUT_SUBDIRS = ("raw", "tables", "figures", "qualitative",
                  "sampling_analysis", "gradient_analysis", "logs", "latex")


def output_layout(root):
    """Create the fixed ``outputs/paper_stbn/`` tree up front and return it as a dict of paths."""
    from pathlib import Path

    root = Path(root)
    layout = {"root": root}
    root.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_SUBDIRS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        layout[name] = path
    return layout
