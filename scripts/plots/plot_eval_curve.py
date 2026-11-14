# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Plot convergence curves from an eval_curve.jsonl produced by v2_train.py.

Writes psnr_vs_iter.png and psnr_vs_time.png into the run dir; several run dirs are overlaid into the first (compare_ prefix).
Usage: python scripts/plots/plot_eval_curve.py <run_dir> [<run_dir> ...]
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
        # Older runs only have avg_psnr_* (dB of mean MSE over all cams).
        y_lin = [e.get("psnr_linear", e.get("avg_psnr_linear")) for e in entries]
        y_srgb = [e.get("psnr_srgb", e.get("avg_psnr_srgb")) for e in entries]
        # sRGB is the paper-reported metric, so it gets the prominent line.
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
