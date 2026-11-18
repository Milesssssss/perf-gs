# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Render the two camera-response figures from the checked-in 3,000-step benchmark CSVs.

Run `python scripts/plots/plot_camera_results.py` from the repository root.
"""
import csv
from pathlib import Path
from statistics import fmean

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
SCENES = ("bicycle", "flowers", "garden", "stump", "treehill", "bonsai", "counter", "kitchen", "room")
INK, MUTED, RULE = "#182531", "#52616c", "#d8dfe2"
BLUE, ORANGE, TEAL = "#2769a1", "#b85b0d", "#117d74"


def read_csv(name):
    with (ROOT / "docs" / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def frame(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=RULE, linewidth=.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", color=RULE, labelcolor=MUTED, labelsize=10, length=0, pad=5)


def heading(fig, title, subtitle):
    fig.text(.035, .963, title, color=INK, size=19, weight="bold", va="top")
    fig.text(.035, .885, subtitle, color=MUTED, size=11, va="top")


def handle(color, marker, label, filled=True):
    return Line2D([], [], linestyle="none", marker=marker, markersize=6.3, color=color,
                  markerfacecolor=color if filled else "white", markeredgewidth=1.4, label=label)


def save(fig, name):
    fig.savefig(ROOT / "assets" / name, dpi=160, facecolor="white")


def plot_mip360():
    rows = read_csv("mipnerf360_3k_results.csv")
    data = {(r["recipe"], int(r["seed"]), r["scene"]): r for r in rows}
    fig = Figure(figsize=(10.6, 5.65))
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=.146, right=.975, top=.733, bottom=.125, wspace=.23)
    heading(fig, "Mip-NeRF 360 at 3,000 steps", "Fixed camera falloff + stereo minus the previous tuned recipe, paired by scene and seed.")
    fig.legend(handles=[handle(BLUE, "o", "Seed 42"), handle(ORANGE, "D", "Seed 1", False)],
               loc="upper right", bbox_to_anchor=(.983, .974), frameon=False, ncol=2,
               handletextpad=.2, columnspacing=.75, labelcolor=INK, fontsize=10)
    metrics = [("psnr_db", 1, "PSNR (dB)", "Higher is better"),
               ("ssim", 1000, "SSIM (×10⁻³)", "Higher is better"),
               ("vgg_lpips", 1000, "VGG LPIPS (×10⁻³)", "Lower is better")]
    for ax, (key, scale, title, direction) in zip(axes, metrics):
        frame(ax)
        ax.text(0, 1.14, title, transform=ax.transAxes, weight="bold", size=12, color=INK)
        ax.text(0, 1.065, direction, transform=ax.transAxes, size=10, color=MUTED)
        ax.axvline(0, color="#707e86", lw=1.1)
        ax.axhline(8.58, color=RULE, lw=1.2)
        deltas = {}
        for seed in (42, 1):
            values = [scale * (float(data["optics-fast", seed, scene][key]) -
                               float(data["previous-tuned", seed, scene][key])) for scene in SCENES]
            deltas[seed] = values + [fmean(values)]
        y = [*range(len(SCENES)), 9.35]
        for i, a, b in zip(y, deltas[42], deltas[1]):
            ax.plot([a, b], [i-.085, i+.085], color="#a9b2b8", lw=1.1, zorder=2)
        ax.scatter(deltas[42], [i-.085 for i in y], color=BLUE, marker="o", s=39, zorder=3)
        ax.scatter(deltas[1], [i+.085 for i in y], facecolors="white", edgecolors=ORANGE,
                   marker="D", s=32, linewidths=1.4, zorder=4)
        ax.set_ylim(9.85, -.6)
        ax.margins(x=.09)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    axes[0].set_yticks(y, [s.title() for s in SCENES] + ["Nine-scene mean"])
    axes[0].get_yticklabels()[-1].set_weight("bold")
    fig.text(.035, .035, "Zero marks the previous recipe. Flowers has higher LPIPS on both seeds.", color=MUTED, size=10)
    save(fig, "mipnerf360_3k_results.png")


def plot_calibration():
    rows = read_csv("camera_falloff_results.csv")
    scenes = ("bicycle", "room", "truck", "drjohnson")
    arms = ("flat", "fixed", "calibrated")
    colors, shapes = (MUTED, BLUE, TEAL), ("s", "D", "o")
    scores = {(r["scene"], r["arm"]): r for r in rows if r["seed"] == "42"}
    fig = Figure(figsize=(10.6, 4.75))
    ax, table = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.14, 1]}, sharey=True)
    fig.subplots_adjust(left=.119, right=.977, top=.708, bottom=.187, wspace=.1)
    heading(fig, "Calibrating camera falloff", "Held-out PSNR across four scenes at 3,000 steps, seed 42.")
    frame(ax)
    ax.text(0, 1.1, "PSNR (dB) · higher is better", transform=ax.transAxes, size=12, weight="bold", color=INK)
    ax.set_yticks(range(4), ["Bicycle", "Room", "Truck", "DrJohnson"])
    ax.set_ylim(3.51, -.51)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    table.set_axis_off()
    table.set_xlim(0, 4)
    for j, arm in enumerate(arms):
        table.plot(j+.19, 1.115, marker=shapes[j], markersize=5.5, color=colors[j],
                   transform=table.get_xaxis_transform(), clip_on=False)
        table.text(j+.31, 1.09, arm.title(), transform=table.get_xaxis_transform(),
                   fontsize=10.3, color=INK, weight="bold")
    table.text(3.56, 1.09, "Fitted b", transform=table.get_xaxis_transform(), ha="center",
               fontsize=10.3, color=INK, weight="bold")
    for i, scene in enumerate(scenes):
        existing = [float(scores[scene, arm]["psnr"]) for arm in arms if (scene, arm) in scores]
        ax.plot([min(existing), max(existing)], [i, i], color=RULE, lw=2.0)
        for j, arm in enumerate(arms):
            record = scores.get((scene, arm))
            label = "—"
            if record is not None:
                value = float(record["psnr"])
                ax.scatter(value, i + (j-1)*.13, s=45, marker=shapes[j], c=colors[j], zorder=3)
                label = f"{value:.3f}"
            table.text(j+.51, i, label, ha="center", va="center",
                       size=11, color=colors[j] if record else MUTED)
        strength = float(scores[scene, "calibrated"]["strength"])
        table.text(3.56, i, f"{strength:.3f}", ha="center", va="center", size=11, color=INK)
        if i < len(scenes)-1:
            table.axhline(i+.50, color=RULE, lw=.7)
    fig.text(.035, .082, "Fitted b scales the lens falloff. Zero gives a flat response and one gives the fixed law.",
             color=MUTED, size=10)
    fig.text(.035, .033, "A dash marks an unmeasured arm.", color=MUTED, size=10)
    save(fig, "camera_falloff_results.png")


if __name__ == "__main__":
    plot_mip360()
    plot_calibration()
