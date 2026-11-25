# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Plot training-loop time and held-out PSNR measured on a common GPU."""
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, ScalarFormatter
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[2]
HELDOUT_VIEWS = {"bicycle": 25, "flowers": 22, "garden": 24, "stump": 16, "treehill": 18,
                 "bonsai": 37, "counter": 30, "kitchen": 35, "room": 39}


MUTED = "#a3a19c"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
LABEL_OFFSETS = ((7, 4, "left"), (7, -12, "left"), (-7, 4, "right"), (-7, -12, "right"),
                 (0, 9, "center"), (0, -18, "center"),
                 (12, 18, "left"), (-12, 18, "right"), (12, -26, "left"), (-12, -26, "right"),
                 (0, 26, "center"), (0, -34, "center"))
NEAR_OFFSETS = 6
MARKER_HALF_PX = 7


def _frontier(points: dict[str, tuple[float, float]]) -> list[str]:
    """Pareto-optimal configs (not dominated on time and quality), sorted by time."""
    front = [name for name, (t, q) in points.items()
             if not any(t2 <= t and q2 >= q and (t2 < t or q2 > q)
                        for other, (t2, q2) in points.items() if other != name)]
    return sorted(front, key=lambda name: points[name][0])


def _overlap(box: Bbox, others: list[Bbox]) -> float:
    return sum(cut.width * cut.height for other in others
               if (cut := Bbox.intersection(box, other)) is not None)


def _place_labels(ax, points: dict[str, tuple[float, float]], labels: dict[str, str],
                  front: list[str]) -> None:
    """Greedy label placement on the final layout, frontier first: first clear in-panel offset, else least overlap."""
    renderer = ax.figure.canvas.get_renderer()
    panel = ax.get_window_extent(renderer)
    taken = [Bbox.from_bounds(x - MARKER_HALF_PX, y - MARKER_HALF_PX,
                              2 * MARKER_HALF_PX, 2 * MARKER_HALF_PX)
             for x, y in ax.transData.transform(list(points.values()))]
    order = sorted(points, key=lambda n: (n not in front, -points[n][1]))
    for name in order:
        color = INK if name in front else INK_MUTED
        scored = []
        for index, (dx, dy, ha) in enumerate(LABEL_OFFSETS):
            leader = ({"arrowstyle": "-", "color": MUTED, "linewidth": 0.6,
                       "shrinkA": 0, "shrinkB": 3} if index >= NEAR_OFFSETS else None)
            note = ax.annotate(labels[name], points[name], xytext=(dx, dy),
                               textcoords="offset points", ha=ha, color=color,
                               fontsize="small", arrowprops=leader)
            note.update_positions(renderer)
            box = Text.get_window_extent(note, renderer)  # the text, not the leader
            inside = (panel.x0 <= box.x0 and box.x1 <= panel.x1
                      and panel.y0 <= box.y0 and box.y1 <= panel.y1)
            penalty = _overlap(box, taken) + (0.0 if inside else 1e9)
            if penalty == 0.0:
                break
            scored.append((penalty, index, note, box))
            note.set_visible(False)
        else:
            _, _, note, box = min(scored, key=lambda s: (s[0], s[1]))
            note.set_visible(True)
        for _, _, other, _ in scored:
            if other is not note:
                other.remove()
        taken.append(box)


def scene_means(name, runs):
    views = {r["scene"]: int(r["heldout_views"]) for r in runs}
    if len(runs) != len(HELDOUT_VIEWS) or views != HELDOUT_VIEWS:
        raise ValueError(f"{name} requires one run per scene with the standard 246 held-out views")
    times = [float(r["train_loop_s"]) for r in runs]
    scores = [float(r["psnr_db"]) for r in runs]
    if not all(math.isfinite(v) and v > 0 for v in times + scores):
        raise ValueError(f"{name} has a missing or invalid time or PSNR")
    return mean(times), mean(scores)


def measurements(presets_path, baseline_paths, additional_presets=()):
    presets = []
    for path in (presets_path, *additional_presets):
        with path.open(newline="") as stream:
            presets.extend(csv.DictReader(stream))
    groups = defaultdict(list)
    for row in presets:
        groups[f"PERF-GS {row['preset'].capitalize()}"].append(row)
    fixed = {"PERF-GS Ultrafast", "PERF-GS Balanced", "PERF-GS Quality"}
    if not fixed <= set(groups) <= fixed | {"PERF-GS Quality+"}:
        raise ValueError("Provide Ultrafast, Balanced and Quality, optionally including Quality+")
    for path in baseline_paths:
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"No baseline measurements in {path}")
        for row in rows:
            name = row.get("method", "LiteGS (aggressive)").strip()
            if not name or name.startswith("PERF-GS"):
                raise ValueError(f"Invalid external method name in {path}")
            groups[name].append(row)
    adapters = {(row.get("gpu") or "").strip() for runs in groups.values() for row in runs}
    if len(adapters) != 1 or "" in adapters:
        raise ValueError("Every row in the plotted CSVs must name the same GPU")
    gpu = adapters.pop().removeprefix("NVIDIA GeForce ")
    points = {name: scene_means(name, runs) for name, runs in groups.items()}
    dimensions = {(r["scene"], r["train_width"], r["train_height"]) for r in presets}
    if (len(dimensions) != len(HELDOUT_VIEWS) or
            any((r["scene"], r["train_width"], r["train_height"]) not in dimensions
                for runs in groups.values() for r in runs)):
        raise ValueError("All trainers must use the same stored image dimensions per scene")
    return gpu, points


def draw(points, gpu):
    fig = Figure(figsize=(9, 4.8), dpi=160, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.10, 0.17, 0.86, 0.62])
    fig.text(0.06, 0.90, f"Mip-NeRF 360 on the {gpu}", weight="bold", size=19, color="#182532")
    fig.text(0.06, 0.84, "Mean across nine scenes", color="#52616d", size=11)
    ax.set_xlabel("Training-loop time per scene (seconds, log scale)")
    ax.set_ylabel("Held-out PSNR (dB)")
    times = [t for t, _ in points.values()]
    low, high = min(times) / 1.4, max(times) * 1.4
    ax.set_xscale("log", base=2)
    ax.set_xlim(low, high)
    ticks = [15 * 2**i for i in range(math.ceil(math.log2(low / 15)), math.floor(math.log2(high / 15)) + 1)]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.margins(y=0.25)
    ax.grid(color="#e0e6e9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=5, colors="#52616d")
    for spine in ax.spines.values():
        spine.set_visible(False)

    front = _frontier(points)
    ax.step(*zip(*(points[n] for n in front)), where="post", color="#99aaa9", linestyle="--", lw=1.3)
    labels = {}
    for name, (seconds, psnr) in points.items():
        external = not name.startswith("PERF-GS ")
        color = "#475e8b" if external else "#00766b"
        ax.scatter(seconds, psnr, color=color, s=55, marker="D" if external else "o", zorder=3)
        labels[name] = f"{name}\n{seconds:.2f} s · {psnr:.2f} dB"
    fig.canvas.draw()
    _place_labels(ax, points, labels, front)
    for label in ax.texts:
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.6})
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", type=Path, nargs="+", default=[
        ROOT / "docs/NVIDIA_GeForce_RTX_4080_presets.csv", ROOT / "docs/NVIDIA_GeForce_RTX_4080_qualityplus.csv"])
    parser.add_argument("--baselines", "--litegs", nargs="+", type=Path,
                        default=[ROOT / "docs/mipnerf360_litegs_4080.csv", ROOT / "docs/mipnerf360_peers_4080.csv"])
    args = parser.parse_args()
    gpu, points = measurements(args.presets[0], args.baselines, args.presets[1:])
    adapter_id = re.sub("[^a-z0-9]", "", gpu.lower())
    if not adapter_id:
        raise ValueError(f"Invalid adapter name {gpu!r}")
    output = ROOT / f"assets/mipnerf360-{adapter_id}-frontier.png"
    with rc_context({"font.size": 12}):
        draw(points, gpu).savefig(output)
    print(output)
