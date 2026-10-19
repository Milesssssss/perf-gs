# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Plot the Mip-NeRF 360 SH0-vs-SH3 sweep: PSNR + training-time bar charts.

Reads outputs/eval_sweep/results.json (written by eval_mipnerf_sweep.py) and
writes, into that folder:
  - psnr_sweep.png     grouped bars: held-out sRGB PSNR per scene, SH0 vs SH3
  - timing_sweep.png   grouped bars: train wall time (s) per scene, SH0 vs SH3
  - eval_sweep_summary.csv

Design (per the dataviz method): two categorical series (SH0, SH3) get a fixed
CVD-safe hue pair (Okabe-Ito blue/orange), one y-axis per chart, a legend, direct
value labels, recessive grid. Scenes in canonical outdoor-then-indoor order with
an AVERAGE group appended.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
OUT = REPO / "outputs" / "eval_sweep"
SCENE_ORDER = ["bicycle", "flowers", "garden", "stump", "treehill",
               "bonsai", "counter", "kitchen", "room"]
# Okabe-Ito: canonical 2-category colorblind-safe pair (identity, fixed order).
SH_COLORS = {0: "#0072B2", 3: "#E69F00"}   # SH0 = blue, SH3 = orange
SH_LABEL = {0: "SH degree 0", 3: "SH degree 3"}


def _index(results, key):
    """{(scene, sh): value} for the requested field."""
    return {(r["scene"], r["sh"]): r.get(key) for r in results}


def _ordered_scenes(results):
    present = {r["scene"] for r in results}
    return [s for s in SCENE_ORDER if s in present]


def _grouped_bar(results, field, ylabel, title, out_path, fmt):
    scenes = _ordered_scenes(results)
    shs = sorted({r["sh"] for r in results})
    vals = _index(results, field)

    # Per-SH mean over scenes (the AVERAGE group), computed over scenes present
    # for that SH so a missing run doesn't bias the mean.
    avg = {}
    for sh in shs:
        xs = [vals[(s, sh)] for s in scenes if vals.get((s, sh)) is not None]
        avg[sh] = float(np.mean(xs)) if xs else None

    labels = scenes + ["AVERAGE"]
    x = np.arange(len(labels))
    n = len(shs)
    bar_w = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(labels)), 5.2), dpi=140)
    for i, sh in enumerate(shs):
        col = [vals.get((s, sh)) for s in scenes] + [avg[sh]]
        col = [np.nan if v is None else v for v in col]
        offs = (i - (n - 1) / 2) * bar_w
        bars = ax.bar(x + offs, col, bar_w, label=SH_LABEL.get(sh, f"SH{sh}"),
                      color=SH_COLORS.get(sh, "#888888"),
                      edgecolor="white", linewidth=0.6, zorder=3)
        for rect, v in zip(bars, col):
            if np.isnan(v):
                continue
            ax.annotate(fmt.format(v), (rect.get_x() + rect.get_width() / 2,
                        rect.get_height()), ha="center", va="bottom",
                        fontsize=7.5, color="#222222", xytext=(0, 1.5),
                        textcoords="offset points")

    # AVERAGE separator + emphasis.
    ax.axvline(len(scenes) - 0.5, color="#bbbbbb", lw=1.0, ls="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    # Figure-level title at the very top; the legend sits just under it (below),
    # both above the axes so neither collides with the tall bars.
    fig.suptitle(title, y=0.99, fontsize=12)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Headroom so the top value labels + legend never collide with the tallest
    # bars, then place the legend in that reserved band above the plot.
    finite = [v for v in
              ([vals.get((s, sh)) for s in scenes for sh in shs] + list(avg.values()))
              if v is not None]
    ax.set_ylim(0, max(finite) * 1.12)
    # Legend between the suptitle and the axes (upper area of the figure).
    ax.legend(frameon=False, ncol=n, loc="lower center",
              bbox_to_anchor=(0.5, 1.01))
    # Reserve a top band for suptitle + legend so tight_layout doesn't overlap.
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(OUT / "results.json"))
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    if not results:
        print("[plot] no results to plot")
        return 1

    _grouped_bar(results, "psnr_srgb", "held-out PSNR sRGB (dB)",
                 "Mip-NeRF 360: PSNR by scene (SH0 vs SH3, 3k iters, quarter-res)",
                 OUT / "psnr_sweep.png", "{:.2f}")
    _grouped_bar(results, "train_wall_s", "train wall time (s)",
                 "Mip-NeRF 360: training time by scene (SH0 vs SH3, 3k iters)",
                 OUT / "timing_sweep.png", "{:.0f}")

    # CSV summary.
    scenes = _ordered_scenes(results)
    shs = sorted({r["sh"] for r in results})
    psnr = _index(results, "psnr_srgb")
    wall = _index(results, "train_wall_s")
    ng = _index(results, "n_gaussians")
    with (OUT / "eval_sweep_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "sh", "psnr_srgb", "train_wall_s", "n_gaussians"])
        for sh in shs:
            for s in scenes:
                w.writerow([s, sh, psnr.get((s, sh)), wall.get((s, sh)), ng.get((s, sh))])
            xs = [psnr[(s, sh)] for s in scenes if psnr.get((s, sh)) is not None]
            ts = [wall[(s, sh)] for s in scenes if wall.get((s, sh)) is not None]
            w.writerow([f"AVERAGE_sh{sh}", sh,
                        f"{np.mean(xs):.3f}" if xs else "",
                        f"{np.mean(ts):.1f}" if ts else "", ""])
    print(f"[plot] wrote {OUT / 'eval_sweep_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
