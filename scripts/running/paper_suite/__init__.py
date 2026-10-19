# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Paper-experiment suite for the STBN sparse-supervision study.

Driven by ``scripts/running/eval_stbn_ablation.py``; see that script's docstring
for the experiment list and the output layout. Modules split by responsibility:

    config      the cell plan: scenes, regimes, seeds, fractions, exact CLI
    env         session provenance (git, GPU, driver, asset hash) + memory poller
    manifest    manifest.json, resume-by-config-hash, integrity accounting
    runner      one cell = one subprocess, with failure capture
    results     load run artifacts, index them for paired queries
    stats       paired differences, bootstrap CIs, sign tests, interpolation
    tables      CSV + booktabs LaTeX emission
    figures     shared matplotlib style, PDF + PNG output
    exp_a..g    one module per experiment
    report      report.md
"""
from __future__ import annotations

OUTPUT_SUBDIRS = ("raw", "tables", "figures", "qualitative",
                  "sampling_analysis", "gradient_analysis", "logs", "latex")


def output_layout(root):
    """Create and return the ``outputs/paper_stbn/`` tree as a dict of paths.

    The suite is meant to be a self-contained folder someone can zip and hand to
    a co-author, so the layout is fixed and created up front rather than
    appearing piecemeal as experiments happen to run.
    """
    from pathlib import Path

    root = Path(root)
    layout = {"root": root}
    root.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_SUBDIRS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        layout[name] = path
    return layout
