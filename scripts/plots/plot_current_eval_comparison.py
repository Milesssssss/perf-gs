# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Plot the current nine-scene timing and PSNR regression evaluation.

Reads ``outputs/current_eval_20260723_vs_baseline.csv``; writes the PNG beside it (wall time and PSNR delta panels).
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPORT = Path("outputs/current_eval_20260723_vs_baseline.csv")
OUTPUT = REPORT.with_suffix(".png")


def main() -> None:
    with REPORT.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["scene"] != "AVERAGE"]

    scenes = [row["scene"] for row in rows]
    baseline_time = np.array(
        [float(row["baseline_train_wall_s"]) for row in rows])
    current_time = np.array(
        [float(row["current_train_wall_s"]) for row in rows])
    time_delta_pct = np.array([float(row["delta_pct"]) for row in rows])
    psnr_delta = np.array([float(row["delta_psnr_db"]) for row in rows])

    baseline_time_avg = float(np.mean(baseline_time))
    current_time_avg = float(np.mean(current_time))
    time_delta_avg = 100.0 * (current_time_avg / baseline_time_avg - 1.0)
    psnr_delta_avg = float(np.mean(psnr_delta))

    y = np.arange(len(scenes))
    bar_height = 0.36
    fig, (timing_ax, psnr_ax) = plt.subplots(
        1, 2, figsize=(14, 7.5), dpi=160,
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    fig.patch.set_facecolor("#f7f8fa")

    baseline_bars = timing_ax.barh(
        y - bar_height / 2,
        baseline_time,
        bar_height,
        color="#68778d",
        label="Frozen baseline",
    )
    current_bars = timing_ax.barh(
        y + bar_height / 2,
        current_time,
        bar_height,
        color="#16836b",
        label="Current tree",
    )
    for baseline_bar, current_bar, baseline, current, delta in zip(
        baseline_bars,
        current_bars,
        baseline_time,
        current_time,
        time_delta_pct,
    ):
        timing_ax.text(
            baseline + 1.0,
            baseline_bar.get_y() + baseline_bar.get_height() / 2,
            f"{baseline:.1f}s",
            va="center",
            fontsize=8,
            color="#465267",
        )
        timing_ax.text(
            current + 1.0,
            current_bar.get_y() + current_bar.get_height() / 2,
            f"{current:.1f}s  ({delta:.1f}%)",
            va="center",
            fontsize=8,
            color="#075f4d",
            fontweight="bold",
        )

    timing_ax.set_yticks(y, scenes)
    timing_ax.invert_yaxis()
    timing_ax.set_xlim(0, max(baseline_time.max(), current_time.max()) * 1.23)
    timing_ax.set_xlabel("Training wall time (seconds, lower is better)")
    timing_ax.set_title(
        f"Runtime: {current_time_avg:.2f}s vs {baseline_time_avg:.2f}s average\n"
        f"Current tree is {-time_delta_avg:.2f}% faster",
        loc="left",
        fontweight="bold",
    )
    timing_ax.legend(frameon=False, loc="lower right")
    timing_ax.grid(axis="x", color="#d9dde4", linewidth=0.8)
    timing_ax.set_axisbelow(True)

    psnr_colors = np.where(psnr_delta >= 0.0, "#16836b", "#c44e52")
    psnr_ax.barh(y, psnr_delta, height=0.56, color=psnr_colors)
    psnr_ax.axvline(0.0, color="#273142", linewidth=1.1)
    psnr_ax.axvline(
        psnr_delta_avg,
        color="#d08a00",
        linewidth=1.3,
        linestyle="--",
        label=f"Average {psnr_delta_avg:+.4f} dB",
    )
    for yi, delta in zip(y, psnr_delta):
        alignment = "left" if delta >= 0.0 else "right"
        offset = 0.00045 if delta >= 0.0 else -0.00045
        psnr_ax.text(
            delta + offset,
            yi,
            f"{delta:+.4f}",
            va="center",
            ha=alignment,
            fontsize=8,
            color="#273142",
        )

    psnr_limit = max(abs(psnr_delta.min()), abs(psnr_delta.max())) * 1.32
    psnr_ax.set_xlim(-psnr_limit, psnr_limit)
    psnr_ax.set_yticks(y, scenes)
    psnr_ax.invert_yaxis()
    psnr_ax.set_xlabel("Current - baseline PSNR (dB)")
    psnr_ax.set_title(
        "Quality: effectively unchanged\n"
        f"Average delta {psnr_delta_avg:+.4f} dB",
        loc="left",
        fontweight="bold",
    )
    psnr_ax.legend(frameon=False, loc="lower right")
    psnr_ax.grid(axis="x", color="#d9dde4", linewidth=0.8)
    psnr_ax.set_axisbelow(True)

    for axis in (timing_ax, psnr_ax):
        axis.set_facecolor("#f7f8fa")
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)

    fig.suptitle(
        "Current Trainer vs Frozen Baseline",
        x=0.06,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color="#1d2735",
    )
    fig.text(
        0.06,
        0.925,
        "Nine Mip-NeRF 360 scenes | SH0 | 3000 iterations | RTX 3090",
        fontsize=10,
        color="#596579",
    )
    fig.tight_layout(rect=(0.035, 0.04, 0.99, 0.89), w_pad=3.0)
    fig.savefig(OUTPUT, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()