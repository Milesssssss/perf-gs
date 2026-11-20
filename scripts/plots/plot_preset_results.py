# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Render the Mip-NeRF 360 preset figure and print its benchmark tables."""
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


def frame(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=RULE, linewidth=.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", color=RULE, labelcolor=MUTED, labelsize=10, length=0, pad=5)


def plot_presets():
    with (ROOT / "docs/mipnerf360_3k_presets.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    presets = [("ultrafast", BLUE, "o"), ("balanced", ORANGE, "D"), ("quality", TEAL, "s")]
    means = {}
    metrics = ["train_loop_s", "psnr_db", "ssim", "vgg_lpips"]
    legend = []
    timings = []
    print("| Preset | Train loop | Launch to saved PLY | PSNR ↑ | SSIM ↑ | VGG LPIPS ↓ |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for preset, color, marker in presets:
        selected = [r for r in rows if r["preset"] == preset]
        seeds = sorted({r["seed"] for r in selected})
        if not seeds:
            continue
        for seed in seeds:
            sample = [r for r in selected if r["seed"] == seed]
            if (len(sample) != len(SCENES) or {r["scene"] for r in sample} != set(SCENES)
                    or sum(int(r["heldout_views"]) for r in sample) != 246):
                raise ValueError(f"Incomplete {preset} sample at seed {seed}")
        legend.append(Line2D([], [], linestyle="none", marker=marker, markersize=6.3, color=color,
                             markerfacecolor=color, markeredgewidth=1.4, label=preset.title()))
        for scene in SCENES:
            scene_rows = [r for r in selected if r["scene"] == scene]
            means[preset, scene] = [fmean(float(r[key]) for r in scene_rows) for key in metrics]
        avg = {key: fmean(float(r[key]) for r in selected) for key in metrics}
        clocks = []
        for key in ("setup_plus_train_loop_s", "launch_to_model_s", "launch_to_closed_ply_s",
                    "launch_to_process_exit_s", "launch_to_synced_ply_s"):
            values = [float(r[key]) for r in selected if r[key]]
            clocks.append(f"{fmean(values):.2f} s" if len(values) == len(selected) else "—")
        print(f"| {preset.title()} | {avg['train_loop_s']:.2f} s | {clocks[-1]} | "
              f"{avg['psnr_db']:.2f} dB | {avg['ssim']:.4f} | {avg['vgg_lpips']:.4f} |")
        splats = fmean(int(r["final_gaussians"]) for r in selected) / 1e6
        timings.append([preset.title(), *clocks, f"{splats:.2f} M"])

    if len(legend) < len(presets):
        return
    print("\n| Preset | Setup + train loop | Launch to model | Launch to closed PLY | Launch to exit | Launch to synced PLY | Gaussians |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cells in timings:
        print("| " + " | ".join(cells) + " |")
    fig = Figure(figsize=(11, 4.8))
    axes = fig.subplots(1, 4, sharey=True)
    fig.subplots_adjust(left=.105, right=.975, top=.68, bottom=.085, wspace=.24)
    seed_text = ", ".join(str(n) for n in sorted({int(r["seed"]) for r in rows}))
    fig.text(.035, .963, "Training presets on Mip-NeRF 360", color=INK, size=19, weight="bold", va="top")
    fig.text(.035, .885, f"3,000 steps on an RTX 4080 at seed {seed_text}. Image metrics are Graphdeco RGB8 scene means.",
             color=MUTED, size=11, va="top")
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.025, .827), labelcolor=INK,
               frameon=False, ncol=3, fontsize=10, handletextpad=.15, columnspacing=1.5)
    titles = ["Train loop (s) ↓", "PSNR (dB) ↑", "SSIM ↑", "VGG LPIPS ↓"]
    for metric_index, (ax, title) in enumerate(zip(axes, titles)):
        frame(ax)
        ax.set_title(title, loc="left", fontsize=12, color=INK, weight="bold", pad=8)
        for i, scene in enumerate(SCENES):
            values = [means[preset, scene][metric_index] for preset, _, _ in presets]
            heights = [i-.14, i, i+.14]
            ax.plot(values, heights, color=RULE, lw=1.3)
            for value, height, (_, color, marker) in zip(values, heights, presets):
                ax.scatter(value, height, color=color, marker=marker, s=32, edgecolor="white",
                           linewidth=.5, zorder=3)
        ax.set_ylim(len(SCENES)-.45, -.55)
        ax.margins(x=.09)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].set_yticks(range(len(SCENES)), [s.title() for s in SCENES])
    fig.savefig(ROOT / "assets/mipnerf360_3k_presets.png", dpi=160, facecolor="white")


if __name__ == "__main__":
    plot_presets()
