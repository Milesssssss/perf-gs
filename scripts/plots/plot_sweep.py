"""Chart a one-axis sweep: held-out PSNR + train time per scene, plus the table.

Reads the ``results.json`` written by ``scripts/running/eval_sweep.py`` and, next
to it, writes ``psnr_sweep.png``, ``timing_sweep.png`` and
``sweep_summary.csv``. Run by the sweep itself after every cell, so it must
tolerate a partial result set (missing cells are simply absent bars).

Design notes, following the dataviz method:

FORM. Two different jobs, so two different forms:

  * TIMING is "compare magnitude, low -> high" -> grouped bars from a zero
    baseline. Seconds genuinely start at zero, so the bar lengths are honest.
  * PSNR is "delta to a target" -> grouped bars of the CHANGE against the
    baseline configuration, centered on zero. Absolute PSNR from a zero baseline
    is the wrong form for it: every scene sits near 20-27 dB and the effect being
    measured is under 1 dB, so a zero-based bar renders the entire result as
    identical bars (verified — it does). Truncating the axis instead would
    misstate the ratios, which is the actual anti-pattern; changing the measure
    to a delta is the fix. Each scene's absolute baseline dB rides on its tick
    label, and every absolute number is in the CSV.

One y-axis per chart; PSNR and seconds are never put on one pair of axes.

COLOR. The swept value is an ORDERED quantity, not an identity, so the series
share a single hue and step along it (dark = the largest value) rather than taking
categorical hues. Steps come from the reference palette's blue ramp; for an
ordinal ramp on a light surface it documents step 250 (``#86b6ef``) as the
lightest usable step, so the ramp runs 250..650 and never lighter. NOTE: the
palette validator is a Node script and Node is not installed here, so these are
the documented-validated steps used as documented — not a fresh validator run.

ACCESSIBILITY. Identity is never color-alone: every chart carries a legend, every
bar carries its value, and ``sweep_summary.csv`` is the table view. Text stays in
ink colors, never the series color. Grid is recessive and behind the bars.
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

# Reference-palette blue ramp, ordinal-safe steps only (>= step 250 on light).
# Applied darkest-to-largest-value: more pixels supervised = more ink.
RAMP = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def _sorted_values(results: list[dict]) -> list[str]:
    """Swept values, numerically descending when they are numbers.

    Descending puts the densest / largest configuration first, which is the
    baseline most sweeps are read against.
    """
    values = sorted({r["value"] for r in results})
    try:
        return sorted(values, key=float, reverse=True)
    except ValueError:
        return values


def _ramp_for(values: list[str]) -> dict[str, str]:
    """One hue, stepped along the ordered values. More than 5 values reuses the
    ramp's endpoints by interpolation of INDEX, keeping order monotone."""
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
        # Re-express every cell as its change from the baseline configuration, and
        # drop the baseline series itself (it is the zero line). Its absolute value
        # is what the reader needs for context, so it goes on the tick label.
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
            # Only the baseline has run so far (a sweep in flight). There is
            # nothing to compare it against yet, so skip rather than emit an
            # empty axis — and say so, because this used to fail silently.
            print(f"[plot] {out_path.name}: only the baseline "
                  f"({axis_label}={baseline_value}) has results; skipping")
            return

    # Per-value mean over the scenes present FOR THAT VALUE, so a missing cell
    # (sweep still running, or one scene absent) cannot bias the average.
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
            # Labels sit outside the bar on whichever side it grows.
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
        # Centred on zero: the baseline IS the reference, so emphasize that line
        # and keep the range symmetric enough that a gain and a loss of the same
        # size look the same size.
        ax.axhline(0.0, color=INK, lw=1.0, zorder=4)
        span = max(abs(min(finite)), abs(max(finite))) or 1.0
        ax.set_ylim(-span * 1.28, span * 1.28)
        ax.spines["bottom"].set_visible(False)
    elif finite:
        # Bars start at zero (a truncated bar axis misstates ratios); the rotated
        # value labels need real headroom above the tallest bar.
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
        # Averages last, and only over cells that exist, so a partial sweep's
        # summary row is still an honest mean of what ran.
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
