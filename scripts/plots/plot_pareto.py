# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Chart the speed/quality frontier written by scripts/running/eval_pareto.py.

Reads ``results.json`` (one record per run), averages seeds per (config, scene)
and writes next to it:

    pareto.png           held-out PSNR against train time, one panel per scene,
                         plus the mean change against the baseline over scenes
    pareto_summary.csv   every point as a table

FORM. Two quantities that trade against each other -> a scatter with time on x
(left is faster) and PSNR on y (up is better). The frontier, the configurations
no other one beats on both axes, is a connected line in the accent hue; the
dominated points are muted. Every point carries its label, so identity is never
color-alone, and the CSV holds every number.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox

ACCENT = "#2a78d6"
MUTED = "#a3a19c"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e4df"
BASELINE = "baseline"
# Label offsets (points) tried in order, nearest first; the far ring gets a leader line.
LABEL_OFFSETS = ((7, 4, "left"), (7, -12, "left"), (-7, 4, "right"), (-7, -12, "right"),
                 (0, 9, "center"), (0, -18, "center"),
                 (12, 18, "left"), (-12, 18, "right"), (12, -26, "left"), (-12, -26, "right"),
                 (0, 26, "center"), (0, -34, "center"))
NEAR_OFFSETS = 6
MARKER_HALF_PX = 7


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _by_scene(results: list[dict]) -> dict[str, dict[str, dict]]:
    """{scene: {config: seed-averaged metrics}}, configs in first-run order."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in results:
        groups[(record["scene"], record["config"])].append(record)
    table: dict[str, dict[str, dict]] = defaultdict(dict)
    for (scene, config), runs in groups.items():
        table[scene][config] = {
            "label": runs[0]["label"],
            "flags": runs[0].get("flags", ""),
            "seeds": len(runs),
            "psnr": _mean([r["psnr_srgb"] for r in runs]),
            "time": _mean([r["train_wall_s"] for r in runs]),
            "splats": _mean([r["n_gaussians"] for r in runs]),
            "ssim": _mean([r.get("ssim_srgb") for r in runs]),
            "lpips": _mean([r.get("lpips") for r in runs]),
        }
    return table


def _frontier(points: dict[str, tuple[float, float]]) -> list[str]:
    """Configs no other config matches on time AND quality while beating one."""
    front = [name for name, (t, q) in points.items()
             if not any(t2 <= t and q2 >= q and (t2 < t or q2 > q)
                        for other, (t2, q2) in points.items() if other != name)]
    return sorted(front, key=lambda name: points[name][0])


def _panel(ax, points: dict[str, tuple[float, float]],
           title: str, xlabel: str, ylabel: str) -> list[str]:
    front = _frontier(points)
    ax.plot([points[n][0] for n in front], [points[n][1] for n in front],
            color=ACCENT, linewidth=1.8, zorder=2)
    for name, (t, q) in points.items():
        on_front = name in front
        ax.scatter(t, q, s=60 if on_front else 40, zorder=3,
                   color=ACCENT if on_front else MUTED,
                   edgecolor=INK if name == BASELINE else "none", linewidth=1.4)
    ax.margins(x=0.18, y=0.2)
    ax.set_title(title, loc="left", color=INK)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return front


def _overlap(box: Bbox, others: list[Bbox]) -> float:
    return sum(cut.width * cut.height for other in others
               if (cut := Bbox.intersection(box, other)) is not None)


def _place_labels(ax, points: dict[str, tuple[float, float]], labels: dict[str, str],
                  front: list[str]) -> None:
    """Greedy placement on the final layout: frontier labels first, each at the
    first offset that stays inside the panel and clears markers and labels (else
    the offset with the least overlap)."""
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


def _relative(table: dict[str, dict[str, dict]]) -> dict[str, tuple[float, float]]:
    """Mean over scenes of (train time / baseline, PSNR - baseline), for configs
    present in every scene that also has a baseline."""
    scenes = [s for s, configs in table.items() if BASELINE in configs]
    shared = set.intersection(*(set(table[s]) for s in scenes)) if scenes else set()
    order = [c for c in table[scenes[0]] if c in shared] if scenes else []
    return {config: (
        100.0 * float(np.mean([table[s][config]["time"] / table[s][BASELINE]["time"]
                               for s in scenes])),
        float(np.mean([table[s][config]["psnr"] - table[s][BASELINE]["psnr"]
                       for s in scenes])))
        for config in order}


def _write_csv(path: Path, table: dict[str, dict[str, dict]],
               relative: dict[str, tuple[float, float]],
               fronts: dict[str, list[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scene", "config", "label", "seeds", "psnr", "train_s",
                         "splats_m", "ssim", "lpips", "on_frontier", "flags"])
        for scene, configs in table.items():
            for config, m in configs.items():
                writer.writerow([scene, config, m["label"], m["seeds"],
                                 f"{m['psnr']:.3f}", f"{m['time']:.2f}",
                                 f"{m['splats'] / 1e6:.2f}",
                                 "" if m["ssim"] is None else f"{m['ssim']:.4f}",
                                 "" if m["lpips"] is None else f"{m['lpips']:.4f}",
                                 config in fronts.get(scene, []), m["flags"]])
        for config, (time_pct, d_psnr) in relative.items():
            writer.writerow(["mean", config, "", "", f"{d_psnr:+.3f}",
                             f"{time_pct:.1f}%", "", "", "",
                             config in fronts.get("mean", []), ""])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, type=Path)
    args = ap.parse_args()

    results = json.loads(args.results.read_text())
    table = _by_scene(results)
    relative = _relative(table)
    labels = {config: m["label"] for configs in table.values()
              for config, m in configs.items()}

    # Scenes side by side on top, the cross-scene mean across the bottom row.
    with rc_context({"font.size": 12}):
        fig = Figure(figsize=(12, 10.5 if relative else 5.5), dpi=130)
        FigureCanvasAgg(fig)
        grid = fig.add_gridspec(2 if relative else 1, len(table))
        drawn: list[tuple[object, dict[str, tuple[float, float]], list[str]]] = []
        fronts: dict[str, list[str]] = {}
        for column, (scene, configs) in enumerate(table.items()):
            seeds = max(m["seeds"] for m in configs.values())
            title = scene if seeds == 1 else f"{scene} (mean of {seeds} seeds)"
            points = {c: (m["time"], m["psnr"]) for c, m in configs.items()}
            ax = fig.add_subplot(grid[0, column])
            fronts[scene] = _panel(ax, points, title, "train time (s)", "held-out PSNR (dB)")
            drawn.append((ax, points, fronts[scene]))
        if relative:
            ax = fig.add_subplot(grid[1, :])
            fronts["mean"] = _panel(ax, relative, "mean over scenes",
                                    "train time (% of baseline)", "PSNR change (dB)")
            ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, zorder=1)
            drawn.append((ax, relative, fronts["mean"]))
        fig.tight_layout()
        fig.canvas.draw()
        for ax, points, front in drawn:
            _place_labels(ax, points, labels, front)

        out_dir = args.results.parent
        fig.savefig(out_dir / "pareto.png")
    _write_csv(out_dir / "pareto_summary.csv", table, relative, fronts)
    print(f"[plot] {out_dir / 'pareto.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
