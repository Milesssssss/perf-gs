"""Grouped bar chart comparing per-scene training time: Slang trainer vs SADGS.

Both series are PURE training-loop wall time (iter 0 -> 3000), excluding data
load and structure-tensor precompute, so they are apples-to-apples:
  * Slang : outputs/train/timing_<scene>/log.jsonl -> train_complete.train_wall_s
  * SADGS : outputs/sadgs_timing/<scene>/timing.json -> loop_wall_s

Writes outputs/timing_comparison.{png,csv}.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# outdoor then indoor (canonical Mip-NeRF 360 order), matching plot_slang_timings.
SCENES = ["bicycle", "flowers", "garden", "stump", "treehill",
          "bonsai", "counter", "kitchen", "room"]

SLANG_ROOT = Path("outputs/train")
SADGS_ROOT = Path("outputs/sadgs_timing")


def slang_wall_s(scene: str) -> float | None:
    p = SLANG_ROOT / f"timing_{scene}" / "log.jsonl"
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


def sadgs_wall_s(scene: str) -> float | None:
    p = SADGS_ROOT / scene / "timing.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["loop_wall_s"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def main() -> None:
    rows = []
    for s in SCENES:
        sl, sd = slang_wall_s(s), sadgs_wall_s(s)
        if sl is None and sd is None:
            continue
        rows.append((s, sl, sd))

    scenes = [s for s, _, _ in rows]
    slang = [sl for _, sl, _ in rows]
    sadgs = [sd for _, _, sd in rows]

    slang_avg = float(np.mean([v for v in slang if v is not None])) if any(v is not None for v in slang) else float("nan")
    sadgs_avg = float(np.mean([v for v in sadgs if v is not None])) if any(v is not None for v in sadgs) else float("nan")

    # CSV
    Path("outputs").mkdir(exist_ok=True)
    with (Path("outputs") / "timing_comparison.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "slang_wall_s", "sadgs_wall_s", "speedup_sadgs_over_slang"])
        for s, sl, sd in rows:
            sp = f"{sd / sl:.2f}x" if (sl and sd) else ""
            w.writerow([s, f"{sl:.1f}" if sl else "", f"{sd:.1f}" if sd else "", sp])
        w.writerow(["AVERAGE", f"{slang_avg:.1f}", f"{sadgs_avg:.1f}",
                    f"{sadgs_avg / slang_avg:.2f}x" if slang_avg else ""])

    # Grouped bar chart
    x = np.arange(len(scenes))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    b1 = ax.bar(x - width / 2, [v if v is not None else 0 for v in slang], width,
                label="Slang trainer (ours)", color="#1f77b4")
    b2 = ax.bar(x + width / 2, [v if v is not None else 0 for v in sadgs], width,
                label="SADGS (reference)", color="#ff7f0e")
    for bars, vals in ((b1, slang), (b2, sadgs)):
        for rect, v in zip(bars, vals):
            if v is not None:
                ax.text(rect.get_x() + rect.get_width() / 2, v + 1, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=8)

    ax.axhline(slang_avg, color="#1f77b4", ls="--", lw=1.4,
               label=f"Slang avg = {slang_avg:.1f}s")
    ax.axhline(sadgs_avg, color="#ff7f0e", ls="--", lw=1.4,
               label=f"SADGS avg = {sadgs_avg:.1f}s")

    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=20)
    ax.set_ylabel("training time, iter 0 \u2192 3000 (seconds)")
    _ymax = max([v for v in (slang + sadgs) if v is not None])
    ax.set_ylim(0, _ymax * 1.15)
    speed = f"{sadgs_avg / slang_avg:.2f}x" if slang_avg else "?"
    ax.set_title("Training time per scene: Slang trainer vs SADGS "
                 f"(SH0, 3000 iters; Slang {slang_avg:.0f}s vs SADGS {sadgs_avg:.0f}s avg, "
                 f"{speed} faster)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(Path("outputs") / "timing_comparison.png")
    plt.close(fig)

    print(f"{'scene':<10} {'slang_s':>9} {'sadgs_s':>9} {'x':>7}")
    for s, sl, sd in rows:
        sp = f"{sd / sl:.2f}x" if (sl and sd) else ""
        print(f"{s:<10} {sl if sl else float('nan'):>9.1f} "
              f"{sd if sd else float('nan'):>9.1f} {sp:>7}")
    print("-" * 38)
    print(f"{'AVERAGE':<10} {slang_avg:>9.1f} {sadgs_avg:>9.1f} "
          f"{sadgs_avg / slang_avg:.2f}x")
    print("\nwrote outputs/timing_comparison.png, outputs/timing_comparison.csv")


if __name__ == "__main__":
    main()
