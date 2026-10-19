# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""BACKWARD_PIXEL_STRIDE sweep: training time + test PSNR vs stride.

Reads outputs/train/timing_<scene>_s<stride>/{log.jsonl,test_psnr.json} for
stride in {1,2,4,8,16} across the 9 Mip-NeRF 360 scenes. Same flags except the
backward pixel-subsampling stride (a shader define). Writes:
  outputs/backward_stride_sweep.csv  (per-scene time + PSNR for each stride)
  outputs/backward_stride_sweep.png  (avg train time + avg PSNR vs stride)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENES = ["bicycle", "flowers", "garden", "stump", "treehill",
          "bonsai", "counter", "kitchen", "room"]
STRIDES = [1, 2, 4, 8, 16]
ROOT = Path("outputs/train")


def _wall_s(run: str) -> float | None:
    p = ROOT / run / "log.jsonl"
    if not p.exists():
        return None
    val = None
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("event") == "train_complete":
            val = float(d["train_wall_s"])
    return val


def _psnr(run: str) -> float | None:
    p = ROOT / run / "test_psnr.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["psnr_srgb"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def main() -> None:
    # time[stride][scene], psnr[stride][scene]
    time_by = {st: {} for st in STRIDES}
    psnr_by = {st: {} for st in STRIDES}
    for s in SCENES:
        for st in STRIDES:
            run = f"timing_{s}_s{st}"
            t, p = _wall_s(run), _psnr(run)
            if t is not None:
                time_by[st][s] = t
            if p is not None:
                psnr_by[st][s] = p

    def avg(d):
        vals = list(d.values())
        return float(np.mean(vals)) if vals else float("nan")

    time_avg = {st: avg(time_by[st]) for st in STRIDES}
    psnr_avg = {st: avg(psnr_by[st]) for st in STRIDES}

    # CSV: one row per scene, columns time_s<st> + psnr_s<st>.
    Path("outputs").mkdir(exist_ok=True)
    with (Path("outputs") / "backward_stride_sweep.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        header = ["scene"] + [f"time_s{st}" for st in STRIDES] + [f"psnr_s{st}" for st in STRIDES]
        w.writerow(header)
        for s in SCENES:
            row = [s]
            row += [f"{time_by[st].get(s, float('nan')):.1f}" for st in STRIDES]
            row += [f"{psnr_by[st].get(s, float('nan')):.3f}" for st in STRIDES]
            w.writerow(row)
        # AVERAGE row + speedup / psnr-drop vs stride 1.
        row = ["AVERAGE"]
        row += [f"{time_avg[st]:.1f}" for st in STRIDES]
        row += [f"{psnr_avg[st]:.3f}" for st in STRIDES]
        w.writerow(row)
        t1 = time_avg[STRIDES[0]]
        p1 = psnr_avg[STRIDES[0]]
        w.writerow(["speedup_vs_s1"] + [f"{t1 / time_avg[st]:.3f}x" for st in STRIDES]
                   + [f"{psnr_avg[st] - p1:+.3f}" for st in STRIDES])

    # Figure: two panels sharing the stride x-axis.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=140)
    x = np.arange(len(STRIDES))

    # Left: avg training time vs stride (+ per-scene faint lines).
    for s in SCENES:
        ys = [time_by[st].get(s) for st in STRIDES]
        if all(y is not None for y in ys):
            ax1.plot(x, ys, color="#bbbbbb", lw=0.8, alpha=0.6, zorder=1)
    ax1.plot(x, [time_avg[st] for st in STRIDES], "-o", color="#1f77b4", lw=2.2,
             ms=6, zorder=3, label="average")
    for xi, st in zip(x, STRIDES):
        ax1.annotate(f"{time_avg[st]:.0f}s\n{t1 / time_avg[st]:.2f}x",
                     (xi, time_avg[st]), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#1f77b4")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(st) for st in STRIDES])
    ax1.set_xlabel("BACKWARD_PIXEL_STRIDE")
    ax1.set_ylabel("training time, iter 0 \u2192 3000 (s)")
    ax1.set_title("Training time vs backward pixel stride (avg of 9 scenes)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Right: avg test PSNR vs stride (+ per-scene faint lines).
    for s in SCENES:
        ys = [psnr_by[st].get(s) for st in STRIDES]
        if all(y is not None for y in ys):
            ax2.plot(x, ys, color="#bbbbbb", lw=0.8, alpha=0.6, zorder=1)
    ax2.plot(x, [psnr_avg[st] for st in STRIDES], "-o", color="#d62728", lw=2.2,
             ms=6, zorder=3, label="average")
    for xi, st in zip(x, STRIDES):
        ax2.annotate(f"{psnr_avg[st]:.2f}\n({psnr_avg[st] - p1:+.2f})",
                     (xi, psnr_avg[st]), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#d62728")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(st) for st in STRIDES])
    ax2.set_xlabel("BACKWARD_PIXEL_STRIDE")
    ax2.set_ylabel("held-out test PSNR sRGB (dB)")
    ax2.set_title("Test PSNR vs backward pixel stride (avg of 9 scenes)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle("Backward pixel-subsampling sweep (SH0, 3000 iters, flat backward)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(Path("outputs") / "backward_stride_sweep.png")
    plt.close(fig)

    # --- Per-scene grid: each scene shows time (left axis) + PSNR (right) ---
    ncol = 3
    nrow = (len(SCENES) + ncol - 1) // ncol
    fig2, axes = plt.subplots(nrow, ncol, figsize=(14, 10), dpi=140)
    axes = np.atleast_2d(axes)
    for idx, s in enumerate(SCENES):
        ax = axes[idx // ncol][idx % ncol]
        ts = [time_by[st].get(s) for st in STRIDES]
        ps = [psnr_by[st].get(s) for st in STRIDES]
        # Left axis: training time (blue).
        ax.plot(x, ts, "-o", color="#1f77b4", lw=1.8, ms=5)
        ax.set_ylabel("time (s)", color="#1f77b4", fontsize=9)
        ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(st) for st in STRIDES], fontsize=8)
        ax.set_title(s, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        # Right axis: test PSNR (red).
        axr = ax.twinx()
        axr.plot(x, ps, "-s", color="#d62728", lw=1.8, ms=5)
        axr.set_ylabel("PSNR (dB)", color="#d62728", fontsize=9)
        axr.tick_params(axis="y", labelcolor="#d62728", labelsize=8)
        # Annotate PSNR drop from stride 1 at each point.
        if ps[0] is not None:
            for xi, pv in zip(x, ps):
                if pv is not None:
                    axr.annotate(f"{pv - ps[0]:+.2f}", (xi, pv),
                                 textcoords="offset points", xytext=(0, 4),
                                 ha="center", fontsize=6.5, color="#d62728")
        if idx % ncol == 0:
            ax.set_xlabel("BACKWARD_PIXEL_STRIDE", fontsize=8)
    # Hide any unused cells.
    for j in range(len(SCENES), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig2.suptitle("Per-scene: training time (blue) & test PSNR (red) vs backward "
                  "pixel stride  |  ΔPSNR vs stride 1 annotated", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    fig2.savefig(Path("outputs") / "backward_stride_sweep_perscene.png")
    plt.close(fig2)

    # Console table.
    print(f"{'stride':>7} {'avg_time':>9} {'speedup':>8} {'avg_psnr':>9} {'dPSNR':>7}")
    for st in STRIDES:
        print(f"{st:>7} {time_avg[st]:>9.1f} {t1 / time_avg[st]:>7.2f}x "
              f"{psnr_avg[st]:>9.3f} {psnr_avg[st] - p1:>+7.3f}")
    print("\nwrote outputs/backward_stride_sweep.png, "
          "outputs/backward_stride_sweep_perscene.png, "
          "outputs/backward_stride_sweep.csv")

if __name__ == "__main__":
    main()
