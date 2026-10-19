# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Overlay + average PSNR-vs-iteration curves from one or more runs.

Consumes the ``psnr_curve.json`` files written by ``v2_train.py --psnr-curve``
(each has aligned ``iters[]`` / ``psnr_srgb[]`` arrays), overlays every run's
curve, and draws the mean curve across runs (averaged over the iterations the
runs share). Use it to compare configs or to get an averaged convergence curve
over several seeds.

Requires matplotlib (`.venv/Scripts/python.exe -m pip install matplotlib`).

Usage:
    # Point at run dirs and/or psnr_curve.json files (mix freely):
    .venv/Scripts/python.exe scripts/plot_psnr_curves.py outputs/train/RUN_A outputs/train/RUN_B
    .venv/Scripts/python.exe scripts/plot_psnr_curves.py a/psnr_curve.json b/psnr_curve.json --out avg.png

Options:
    --out PATH     output PNG path (default: psnr_curves.png in the CWD)
    --label-avg    also print the averaged (iter, mean_psnr) table to stdout
    --no-average   overlay the individual curves only (skip the mean curve)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _resolve(path: Path) -> Path:
    """Accept a run dir (looks for psnr_curve.json inside) or the json directly."""
    if path.is_dir():
        cand = path / "psnr_curve.json"
        if not cand.exists():
            raise FileNotFoundError(
                f"no psnr_curve.json in {path} — was the run trained with --psnr-curve?")
        return cand
    return path


def load_curve(path: Path) -> tuple[str, str, dict[int, float]]:
    """Return (label, scene, {iter: psnr_srgb}) for one run's psnr_curve.json."""
    data = json.loads(path.read_text())
    label = data.get("run_name") or path.parent.name
    scene = data.get("scene", "")
    scene_name = Path(scene).name or scene if scene else "?"
    iters = data["iters"]
    psnr = data["psnr_srgb"]
    return label, scene_name, {int(i): float(p) for i, p in zip(iters, psnr)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path,
                    help="run dirs and/or psnr_curve.json files to plot")
    ap.add_argument("--out", type=Path, default=Path("psnr_curves.png"))
    ap.add_argument("--no-average", action="store_true",
                    help="overlay individual curves only (skip the mean curve)")
    ap.add_argument("--label-avg", action="store_true",
                    help="also print the averaged (iter, mean_psnr) table")
    args = ap.parse_args()

    curves = [load_curve(_resolve(p)) for p in args.paths]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    for label, scene_name, series in curves:
        xs = sorted(series)
        ys = [series[x] for x in xs]
        ax.plot(xs, ys, "-", lw=1.2, alpha=0.6, label=f"{label} [{scene_name}]")

    # Mean curve over the iterations shared by ALL runs (avoids biasing the
    # average where only some runs have samples).
    if not args.no_average and len(curves) > 1:
        common = set(curves[0][2])
        for _, _, series in curves[1:]:
            common &= set(series)
        if common:
            xs = sorted(common)
            mean = [float(np.mean([s[x] for _, _, s in curves])) for x in xs]
            ax.plot(xs, mean, "-", color="black", lw=2.6,
                    label=f"mean of {len(curves)} runs")
            if args.label_avg:
                print("iter, mean_psnr_srgb")
                for x, m in zip(xs, mean):
                    print(f"{x}, {m:.4f}")
        else:
            print("[warn] runs share no common iterations; skipping mean curve.")

    # Title reflects the dataset(s): a single scene if all runs share one, else
    # a comma-joined set so a mixed-scene comparison is self-documenting.
    scenes = sorted({c[1] for c in curves})
    scene_str = scenes[0] if len(scenes) == 1 else ", ".join(scenes)
    ax.set_xlabel("iteration")
    ax.set_ylabel("test PSNR sRGB (dB)")
    ax.set_title(f"PSNR vs iteration — {scene_str}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out)
    plt.close(fig)
    print(f"wrote {args.out}  ({len(curves)} run(s))")


if __name__ == "__main__":
    main()
