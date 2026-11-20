# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Render the Mip-NeRF 360 preset figure and print its benchmark tables."""
import argparse
import csv
import math
import re
from pathlib import Path
from statistics import fmean

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "docs/mipnerf360_3k_presets.csv"
HELDOUT_VIEWS = {"bicycle": 25, "flowers": 22, "garden": 24, "stump": 16, "treehill": 18,
                 "bonsai": 37, "counter": 30, "kitchen": 35, "room": 39}
INK, MUTED, RULE = "#182531", "#52616c", "#d8dfe2"
BLUE, ORANGE, TEAL = "#2769a1", "#b85b0d", "#117d74"


def frame(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=RULE, linewidth=.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", color=RULE, labelcolor=MUTED, labelsize=10, length=0, pad=5)


def plot_presets(presets_path=DEFAULT_CSV, output=None):
    with presets_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    presets = [("ultrafast", BLUE, "o"), ("balanced", ORANGE, "D"), ("quality", TEAL, "s")]
    adapters = {(r.get("gpu") or "").strip() for r in rows}
    if len(adapters) != 1 or "" in adapters:
        raise ValueError("All measurements must name the same GPU")
    gpu = adapters.pop().removeprefix("NVIDIA GeForce ")
    seeds = sorted({int(r["seed"]) for r in rows})
    expected = {(preset, seed, scene) for preset, _, _ in presets for seed in seeds for scene in HELDOUT_VIEWS}
    observed = {(r["preset"], int(r["seed"]), r["scene"]) for r in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("Each seed needs one run per preset for each of the nine scenes")
    if any(int(r["steps"]) != 3000 or int(r["heldout_views"]) != HELDOUT_VIEWS[r["scene"]] for r in rows):
        raise ValueError("Each run needs 3,000 steps and the standard held-out views for its scene")
    if output is None:
        adapter_id = re.sub("[^a-z0-9]", "", gpu.lower())
        filename = ("mipnerf360_3k_presets.png" if presets_path.resolve() == DEFAULT_CSV
                    else f"mipnerf360-{adapter_id}-presets.png")
        output = ROOT / "assets" / filename
    means = {}
    metrics = ["train_loop_s", "psnr_db", "ssim", "vgg_lpips"]
    if any(not math.isfinite(float(r[key])) for r in rows for key in metrics):
        raise ValueError("Every run needs finite timing and image metrics")
    legend = []
    timings = []
    summaries = []
    for preset, color, marker in presets:
        selected = [r for r in rows if r["preset"] == preset]
        legend.append(Line2D([], [], linestyle="none", marker=marker, markersize=6.3, color=color,
                             markerfacecolor=color, markeredgewidth=1.4, label=preset.title()))
        for scene in HELDOUT_VIEWS:
            scene_rows = [r for r in selected if r["scene"] == scene]
            means[preset, scene] = [fmean(float(r[key]) for r in scene_rows) for key in metrics]
        avg = {key: fmean(float(r[key]) for r in selected) for key in metrics}
        clocks = []
        for key in ("setup_plus_train_loop_s", "launch_to_model_s", "launch_to_closed_ply_s",
                    "launch_to_process_exit_s", "launch_to_synced_ply_s"):
            values = [float(r[key]) for r in selected if r.get(key)]
            clocks.append(f"{fmean(values):.2f} s" if len(values) == len(selected) else "—")
        summaries.append(f"| {preset.title()} | {avg['train_loop_s']:.2f} s | {clocks[-1]} | "
                         f"{avg['psnr_db']:.2f} dB | {avg['ssim']:.4f} | {avg['vgg_lpips']:.4f} |")
        splats = fmean(int(r["final_gaussians"]) for r in selected) / 1e6
        timings.append([preset.title(), *clocks, f"{splats:.2f} M"])

    print("| Preset | Train loop | Launch to saved PLY | PSNR ↑ | SSIM ↑ | VGG LPIPS ↓ |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    print("\n".join(summaries))
    print("\n| Preset | Setup + train loop | Launch to model | Launch to closed PLY | Launch to exit | Launch to synced PLY | Gaussians |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cells in timings:
        print("| " + " | ".join(cells) + " |")
    fig = Figure(figsize=(11, 4.8))
    axes = fig.subplots(1, 4, sharey=True)
    fig.subplots_adjust(left=.105, right=.975, top=.68, bottom=.085, wspace=.24)
    seed_text = ", ".join(map(str, seeds))
    card = f"an {gpu}" if gpu.startswith("RTX ") else f"the {gpu}"
    fig.text(.035, .963, "Training presets on Mip-NeRF 360", color=INK, size=19, weight="bold", va="top")
    fig.text(.035, .885, f"3,000 steps on {card} at seed {seed_text}. Image metrics are Graphdeco RGB8 scene means.",
             color=MUTED, size=11, va="top")
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.025, .827), labelcolor=INK,
               frameon=False, ncol=3, fontsize=10, handletextpad=.15, columnspacing=1.5)
    titles = ["Train loop (s) ↓", "PSNR (dB) ↑", "SSIM ↑", "VGG LPIPS ↓"]
    for metric_index, (ax, title) in enumerate(zip(axes, titles)):
        frame(ax)
        ax.set_title(title, loc="left", fontsize=12, color=INK, weight="bold", pad=8)
        for i, scene in enumerate(HELDOUT_VIEWS):
            values = [means[preset, scene][metric_index] for preset, _, _ in presets]
            heights = [i-.14, i, i+.14]
            ax.plot(values, heights, color=RULE, lw=1.3)
            for value, height, (_, color, marker) in zip(values, heights, presets):
                ax.scatter(value, height, color=color, marker=marker, s=32, edgecolor="white",
                           linewidth=.5, zorder=3)
        ax.set_ylim(len(HELDOUT_VIEWS)-.45, -.55)
        ax.margins(x=.09)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].set_yticks(range(len(HELDOUT_VIEWS)), [s.title() for s in HELDOUT_VIEWS])
    fig.savefig(output, dpi=160, facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", type=Path, default=DEFAULT_CSV, help="Per-scene preset CSV for one GPU")
    parser.add_argument("--output", type=Path, help="PNG destination (defaults to the preset figure in assets/)")
    args = parser.parse_args()
    plot_presets(args.presets, args.output)
