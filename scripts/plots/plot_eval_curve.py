# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Plot convergence curves from an eval_curve.jsonl produced by v2_train.py.

Reads a run's eval_curve.jsonl (one JSON line per --eval-every sweep) and writes
two PNGs into the same run dir:
  - psnr_vs_iter.png       PSNR (linear + sRGB) vs training iteration
  - psnr_vs_time.png       PSNR (linear + sRGB) vs pure training wall-clock

Requires matplotlib. Install once with `.venv/Scripts/python.exe -m pip install matplotlib`.

Usage:
    .venv/Scripts/python.exe scripts/plot_eval_curve.py <run_dir> [<run_dir> ...]

Passing multiple run dirs overlays them on one figure (labeled by dir basename)
for A/B comparison. The X-axis for the time plot uses each run's own elapsed
so throughput differences show up as horizontal stretch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_curve(run_dir: Path) -> tuple[list[dict], str]:
    p = run_dir / "eval_curve.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"no eval_curve.jsonl in {run_dir} — was --eval-every > 0?")
    entries = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return entries, run_dir.name


def plot(runs: list[tuple[list[dict], str]], out_path: Path, x_field: str, x_label: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    for entries, label in runs:
        xs = [e[x_field] for e in entries]
        # Current schema: psnr_srgb / psnr_linear (mean of per-image PSNR on the
        # test split, paper metric). Old runs used avg_psnr_srgb / avg_psnr_linear
        # (dB-of-mean-MSE over all cams) — fall back so old curves still plot.
        y_lin = [e.get("psnr_linear", e.get("avg_psnr_linear")) for e in entries]
        y_srgb = [e.get("psnr_srgb", e.get("avg_psnr_srgb")) for e in entries]
        # Solid line = sRGB (paper-reported metric); dashed = linear for context.
        ax.plot(xs, y_srgb, "-", label=f"{label} (sRGB)", linewidth=1.8)
        ax.plot(xs, y_lin, "--", label=f"{label} (linear)", linewidth=1.0, alpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel("avg PSNR (dB)")
    ax.set_title(f"PSNR convergence vs {x_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_dirs = [Path(a) for a in sys.argv[1:]]
    runs = [load_curve(d) for d in run_dirs]
    # Single-run outputs go into that run's dir; multi-run comparison goes into
    # the FIRST run's dir with a "compare_" prefix.
    if len(runs) == 1:
        out_dir = run_dirs[0]
        prefix = ""
    else:
        out_dir = run_dirs[0]
        prefix = "compare_"
    plot(runs, out_dir / f"{prefix}psnr_vs_iter.png", "iter", "iteration")
    plot(runs, out_dir / f"{prefix}psnr_vs_time.png", "elapsed_s", "training time (s)")


if __name__ == "__main__":
    main()
