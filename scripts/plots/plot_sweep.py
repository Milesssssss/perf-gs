# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Chart a one-axis sweep: held-out PSNR + train time per scene, plus the table.

Run by eval_sweep.py after every cell, so a partial results.json must plot; outputs go beside it.
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
SCENE_ORDER = ["bicycle", "flowers", "garden", "stump", "treehill",
               "bonsai", "counter", "kitchen", "room"]

# Ordinal blue ramp, palette steps 650..250 (250 is the lightest usable on white); darkest = largest value.
RAMP = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def _sorted_values(results: list[dict]) -> list[str]:
    """Swept values, numerically descending when numeric, so the baseline (largest) comes first."""
    values = sorted({r["value"] for r in results})
    try:
        return sorted(values, key=float, reverse=True)
    except ValueError:
        return values


def _ramp_for(values: list[str]) -> dict[str, str]:
    """One RAMP color per ordered value by interpolated index (monotone; colors repeat past len(RAMP))."""
    if len(values) <= len(RAMP):
        picks = np.linspace(0, len(RAMP) - 1, num=max(len(values), 1))
        return {v: RAMP[int(round(p))] for v, p in zip(values, picks)}
    idx = np.linspace(0, len(RAMP) - 1, num=len(values))
    return {v: RAMP[int(round(i))] for v, i in zip(values, idx)}


def _ordered_scenes(results: list[dict]) -> list[str]:
    present = {r["scene"] for r in results}
    known = [s for s in SCENE_ORDER if s in present]
    return known + sorted(present - set(known))


def _index(results: list[dict], field: str) -> dict[tuple[str, str], float]:
    return {(r["scene"], r["value"]): r.get(field) for r in results}


def _grouped_bar(results, field, ylabel, title, out_path, fmt, axis_label,
                 *, relative_to_baseline: bool = False):
    scenes = _ordered_scenes(results)
    values = _sorted_values(results)
    cell = _index(results, field)
    colors = _ramp_for(values)
    baseline_value = values[0] if values else None

    tick_labels = list(scenes)
    if relative_to_baseline:
        # Sub-dB effects vanish on zero-based bars: plot change vs baseline, its absolute dB on the tick.
        absolute = dict(cell)
        cell = {}
        for scene in scenes:
            base = absolute.get((scene, baseline_value))
            for value in values:
                here = absolute.get((scene, value))
                if base is not None and here is not None:
                    cell[(scene, value)] = here - base
        tick_labels = [
            f"{s}\n{absolute[(s, baseline_value)]:.1f} dB"
            if absolute.get((s, baseline_value)) is not None else s
            for s in scenes
        ]
        values = values[1:]
        if not values:
            print(f"[plot] {out_path.name}: only the baseline "
                  f"({axis_label}={baseline_value}) has results; skipping")
            return

    # Per-value mean over only the scenes present for that value (the sweep may still be running).
    average = {}
    for value in values:
        xs = [cell[(s, value)] for s in scenes if cell.get((s, value)) is not None]
        average[value] = float(np.mean(xs)) if xs else None

    labels = tick_labels + ["AVERAGE"]
    x = np.arange(len(labels))
    bar_w = 0.82 / max(len(values), 1)

    fig, ax = plt.subplots(figsize=(max(10.0, 1.25 * len(labels)), 5.6), dpi=140)
    for i, value in enumerate(values):
        column = [cell.get((s, value)) for s in scenes] + [average[value]]
        column = [np.nan if v is None else v for v in column]
        offset = (i - (len(values) - 1) / 2) * bar_w
        bars = ax.bar(x + offset, column, bar_w * 0.92,   # gap between fills
                      label=f"{axis_label} = {value}", color=colors[value],
                      edgecolor="white", linewidth=0.6, zorder=3)
        for rect, v in zip(bars, column):
            if np.isnan(v):
                continue
            below = v < 0
            ax.annotate(fmt.format(v),
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        ha="center", va="top" if below else "bottom", rotation=90,
                        fontsize=6.2, color=INK_MUTED,
                        xytext=(0, -2 if below else 2), textcoords="offset points")

    ax.axvline(len(scenes) - 0.5, color="#cccccc", lw=1.0, ls="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0 if relative_to_baseline else 30,
                       ha="center" if relative_to_baseline else "right",
                       fontsize=7.5 if relative_to_baseline else 9, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    fig.suptitle(title, y=0.99, fontsize=12, color=INK)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    finite = [v for v in list(cell.values()) + list(average.values()) if v is not None]
    if finite and relative_to_baseline:
        # Symmetric range so a gain and a loss of the same size look the same size.
        ax.axhline(0.0, color=INK, lw=1.0, zorder=4)
        span = max(abs(min(finite)), abs(max(finite))) or 1.0
        ax.set_ylim(-span * 1.28, span * 1.28)
        ax.spines["bottom"].set_visible(False)
    elif finite:
        # Bars start at zero (a truncated axis misstates ratios); 20% headroom for the rotated labels.
        ax.set_ylim(0, max(finite) * 1.20)
    ax.legend(frameon=False, ncol=max(1, min(len(values), 5)), loc="lower center",
              bbox_to_anchor=(0.5, 1.005), fontsize=8, labelcolor=INK_MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def _write_csv(results, out_path, axis_label):
    scenes = _ordered_scenes(results)
    values = _sorted_values(results)
    psnr = _index(results, "psnr_srgb")
    wall = _index(results, "train_wall_s")
    count = _index(results, "n_gaussians")

    with out_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scene", axis_label, "psnr_srgb", "train_wall_s",
                         "n_gaussians"])
        for value in values:
            for scene in scenes:
                writer.writerow([scene, value, psnr.get((scene, value)),
                                 wall.get((scene, value)),
                                 count.get((scene, value))])
        # Averages only over cells that exist, so a partial sweep still gets an honest mean.
        for value in values:
            ps = [psnr[(s, value)] for s in scenes if psnr.get((s, value)) is not None]
            ws = [wall[(s, value)] for s in scenes if wall.get((s, value)) is not None]
            writer.writerow([f"AVERAGE (n={len(ps)})", value,
                             f"{np.mean(ps):.3f}" if ps else "",
                             f"{np.mean(ws):.1f}" if ws else "", ""])
    print(f"[plot] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--axis-label", default="value",
                    help="how the swept flag is named in legends/headers")
    ap.add_argument("--iters", type=int, default=0)
    args = ap.parse_args()

    path = Path(args.results)
    results = [r for r in json.loads(path.read_text())
               if r.get("psnr_srgb") is not None]
    if not results:
        print("[plot] nothing to plot yet")
        return 1
    out = path.parent
    scenes = _ordered_scenes(results)
    values = _sorted_values(results)
    baseline = values[0] if values else "?"
    run_note = f"{args.iters} iters, " if args.iters else ""
    suffix = (f"({run_note}quarter-res, "
              f"{len(scenes)} scene{'s' if len(scenes) != 1 else ''}, "
              f"held-out protocol)")

    _grouped_bar(results, "psnr_srgb",
                 f"held-out PSNR change vs {args.axis_label}={baseline} (dB)",
                 f"Mip-NeRF 360 PSNR cost of {args.axis_label} {suffix}\n"
                 f"above 0 = better than {args.axis_label}={baseline}; "
                 f"each scene's baseline dB is on its label",
                 out / "psnr_sweep.png", "{:+.2f}", args.axis_label,
                 relative_to_baseline=True)
    _grouped_bar(results, "train_wall_s", "train wall time (s)",
                 f"Mip-NeRF 360 training time vs {args.axis_label} {suffix}",
                 out / "timing_sweep.png", "{:.0f}", args.axis_label)
    _write_csv(results, out / "sweep_summary.csv", args.axis_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
