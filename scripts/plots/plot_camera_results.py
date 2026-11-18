# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Render camera-response comparisons from the checked-in 3,000-step benchmark CSVs.

Run `python scripts/plots/plot_camera_results.py` from the repository root.
"""
import csv
from math import isfinite
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


def plot_comparison(data, recipe, baseline, title, subtitle, note, filename):
    fig = Figure(figsize=(10.6, 5.65))
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=.146, right=.975, top=.733, bottom=.125, wspace=.23)
    heading(fig, title, subtitle)
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
            values = [scale * (float(data[recipe, seed, scene][key]) -
                               float(data[baseline, seed, scene][key])) for scene in SCENES]
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
    fig.text(.035, .035, note, color=MUTED, size=10)
    save(fig, filename)


def plot_mip360():
    rows = read_csv("mipnerf360_3k_results.csv")
    data = {}
    for row in rows:
        key = row["recipe"], int(row["seed"]), row["scene"]
        if key in data:
            raise ValueError(f"Duplicate benchmark row: {key}")
        data[key] = row
    plot_comparison(data, "optics-fast", "previous-tuned", "Mip-NeRF 360 at 3,000 steps",
                    "Fixed camera falloff + stereo minus the previous tuned recipe, paired by scene and seed.",
                    "Zero marks the previous recipe. Flowers has higher LPIPS on both seeds.",
                    "mipnerf360_3k_results.png")
    for scene in SCENES:
        for seed in (42, 1):
            adaptive = data.get(("optics-adaptive", seed, scene))
            if adaptive is None or any(not adaptive[k] or not isfinite(float(adaptive[k]))
                                       for k in ("psnr_db", "ssim", "vgg_lpips", "train_loop_s")):
                print("Adaptive plot awaits both complete nine-scene sweeps.")
                return
            fixed = data["optics-fast", seed, scene]
            if adaptive["steps"] != "3000" or adaptive["heldout_views"] != fixed["heldout_views"]:
                raise ValueError(f"Adaptive and fixed protocols differ for {scene}, seed {seed}")
    plot_comparison(data, "optics-adaptive", "optics-fast", "Adaptive versus fixed camera falloff",
                    "Adaptive minus fixed falloff on Mip-NeRF 360, with stereo and 3,000 steps for both recipes.",
                    "Zero marks the fixed falloff result. The nine-scene mean gives each scene equal weight.",
                    "mipnerf360_3k_adaptive_vs_fixed.png")


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


def plot_paired_timing():
    try:
        rows = read_csv("mipnerf360_3k_timing.csv")
    except FileNotFoundError:
        return
    times = {}
    for row in rows:
        if row["arm"] not in ("stock", "adaptive") or row["seed"] != "42":
            continue
        key = row["arm"], row["scene"]
        if key in times:
            raise ValueError(f"Duplicate timing run: {key}")
        loop, total = float(row["train_s"]), float(row["setup_plus_train_s"])
        if not (isfinite(loop) and isfinite(total) and 0 < loop <= total):
            raise ValueError(f"Invalid timing for {key}")
        times[key] = loop, total
    expected = {(arm, scene) for arm in ("stock", "adaptive") for scene in SCENES}
    if times.keys() != expected:
        raise ValueError("The paired timing figure requires one complete stock and adaptive sweep.")

    for arm in ("stock", "adaptive"):
        pairs = [times[arm, scene] for scene in SCENES]
        times[arm, "mean"] = tuple(fmean(column) for column in zip(*pairs))
    fig = Figure(figsize=(10.6, 5.85))
    ax, table = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.45, 1]}, sharey=True)
    fig.subplots_adjust(left=.148, right=.968, top=.733, bottom=.135, wspace=.14)
    heading(fig, "Paired training time on Mip-NeRF 360",
            "Stock and adaptive runs alternated on one RTX 4080. Each scene used 3,000 steps and seed 42.")
    frame(ax)
    ax.text(0, 1.15, "Change from stock (%)", transform=ax.transAxes, color=INK, size=12, weight="bold")
    ax.legend(handles=[handle(TEAL, "o", "Setup + train"), handle(BLUE, "D", "Train loop only", False)],
              loc="lower left", bbox_to_anchor=(-.02, 1.005), frameon=False, ncol=2,
              handletextpad=.15, columnspacing=.8, borderaxespad=0, labelcolor=INK, fontsize=10)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(8, color=ORANGE, lw=1.1, ls="--")
    ax.text(8, .975, "8% threshold", transform=ax.get_xaxis_transform(), color=ORANGE,
            ha="center", va="top", size=9, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    table.set_axis_off()
    table.set_xlim(0, 1)
    table.text(0, 1.15, "Setup + training", transform=table.transAxes, color=INK, size=12, weight="bold")
    columns = (.23, .60, .99)
    for x, label in zip(columns, ("Stock (s)", "Adaptive (s)", "Change")):
        table.text(x, 1.045, label, transform=table.transAxes, ha="right", color=MUTED, size=10)

    y = [*range(len(SCENES)), 9.35]
    for scene, height in zip((*SCENES, "mean"), y):
        stock_loop, stock_total = times["stock", scene]
        adaptive_loop, adaptive_total = times["adaptive", scene]
        loop_delta = 100 * (adaptive_loop / stock_loop - 1)
        total_delta = 100 * (adaptive_total / stock_total - 1)
        ax.plot([loop_delta, total_delta], [height+.07, height-.07], color=RULE, lw=1.5)
        ax.scatter(loop_delta, height+.07, marker="D", facecolors="white", edgecolors=BLUE,
                   s=31, linewidths=1.4, zorder=3)
        ax.scatter(total_delta, height-.07, color=TEAL, s=46, zorder=4)
        for x, label in zip(columns, (f"{stock_total:.2f}", f"{adaptive_total:.2f}", f"{total_delta:+.2f}%")):
            table.text(x, height, label, ha="right", va="center", color=INK, size=11,
                       weight="bold" if scene == "mean" else "normal")
    for panel in (ax, table):
        panel.axhline(8.58, color=RULE, lw=1.2)
    ax.set_ylim(9.85, -.95)
    ax.set_yticks(y, [s.title() for s in SCENES] + ["Nine-scene mean"])
    ax.get_yticklabels()[-1].set_weight("bold")
    ax.margins(x=.1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    fig.text(.035, .036, "Positive values mean longer runs. The mean compares the two nine-scene average times.",
             color=MUTED, size=10)
    save(fig, "mipnerf360_3k_paired_timing.png")


def plot_presets():
    rows = read_csv("mipnerf360_3k_presets.csv")
    presets = [("ultrafast", BLUE, "o"), ("balanced", ORANGE, "D"), ("quality", TEAL, "s")]
    means = {}
    metrics = ["train_loop_s", "psnr_db", "vgg_lpips"]
    legend = []
    print("| Preset | Seeds | Train loop | Setup + training | Launch to model | PSNR ↑ | SSIM ↑ | VGG LPIPS ↓ |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
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
        count = len(seeds)
        legend.append(handle(color, marker, preset.title()))
        for scene in SCENES:
            scene_rows = [r for r in selected if r["scene"] == scene]
            means[preset, scene] = [fmean(float(r[key]) for r in scene_rows) for key in metrics]
        avg = {key: fmean(float(r[key]) for r in selected) for key in [*metrics, "ssim"]}
        setup_times = [float(r["setup_plus_train_loop_s"]) for r in selected if r["setup_plus_train_loop_s"]]
        setup = f"{fmean(setup_times):.2f} s" if len(setup_times) == len(selected) else "—"
        model_times = [float(r["launch_to_model_s"]) for r in selected if r["launch_to_model_s"]]
        ready = f"{fmean(model_times):.2f} s" if len(model_times) == len(selected) else "—"
        print(f"| {preset.title()} | {count} | {avg['train_loop_s']:.2f} s | {setup} | {ready} | "
              f"{avg['psnr_db']:.2f} dB | {avg['ssim']:.4f} | {avg['vgg_lpips']:.4f} |")

    if len(legend) < len(presets):
        return
    fig = Figure(figsize=(9.8, 4.8))
    axes = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=.13, right=.975, top=.68, bottom=.085, wspace=.2)
    seed_text = ", ".join(str(n) for n in sorted({int(r["seed"]) for r in rows}))
    heading(fig, "Training presets on Mip-NeRF 360",
            f"3,000 steps on an RTX 4080 at seed {seed_text}. Quality is the Graphdeco RGB8 mean for each scene.")
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.025, .827), labelcolor=INK,
               frameon=False, ncol=3, fontsize=10, handletextpad=.15, columnspacing=1.5)
    titles = ["Train loop (s) ↓", "PSNR (dB) ↑", "VGG LPIPS ↓"]
    for metric_index, (ax, title) in enumerate(zip(axes, titles)):
        frame(ax)
        ax.set_title(title, loc="left", fontsize=12, color=INK, weight="bold", pad=8)
        for i, scene in enumerate(SCENES):
            values = [means[preset, scene][metric_index] for preset, _, _ in presets]
            ax.plot([min(values), max(values)], [i, i], color=RULE, lw=1.6)
            for value, (_, color, marker) in zip(values, presets):
                ax.scatter(value, i, color=color, marker=marker, s=32, edgecolor="white",
                           linewidth=.5, zorder=3)
        ax.set_ylim(len(SCENES)-.45, -.55)
        ax.margins(x=.09)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axes[0].set_yticks(range(len(SCENES)), [s.title() for s in SCENES])
    save(fig, "mipnerf360_3k_presets.png")


if __name__ == "__main__":
    plot_mip360()
    plot_calibration()
    plot_paired_timing()
    plot_presets()
