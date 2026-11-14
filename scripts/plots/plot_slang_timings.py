# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Bar chart of Slang trainer per-scene training time (0->3k iters).

Reads train_complete.train_wall_s from outputs/train/timing_<scene>/log.jsonl; writes outputs/slang_timings.{png,csv}.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("outputs/train")
# outdoor then indoor (canonical Mip-NeRF 360 order).
SCENES = ["bicycle", "flowers", "garden", "stump", "treehill",
          "bonsai", "counter", "kitchen", "room"]


def train_wall_s(scene: str) -> float | None:
    p = ROOT / f"timing_{scene}" / "log.jsonl"
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


def main() -> None:
    rows = [(s, train_wall_s(s)) for s in SCENES]
    rows = [(s, t) for s, t in rows if t is not None]
    scenes = [s for s, _ in rows]
    times = [t for _, t in rows]
    avg = float(np.mean(times))

    with (Path("outputs") / "slang_timings.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "train_wall_s", "it_per_s_3000"])
        for s, t in rows:
            w.writerow([s, f"{t:.1f}", f"{3000 / t:.1f}"])
        w.writerow(["AVERAGE", f"{avg:.1f}", f"{3000 / avg:.1f}"])

    x = np.arange(len(scenes))
    colors = ["#1f77b4"] * len(scenes)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    bars = ax.bar(x, times, color=colors)
    for xi, t in zip(x, times):
        ax.text(xi, t + 1, f"{t:.0f}s", ha="center", va="bottom", fontsize=9)
    ax.axhline(avg, color="#d62728", ls="--", lw=1.8,
               label=f"average = {avg:.1f}s")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=20)
    ax.set_ylabel("training time, iter 0 \u2192 3000 (seconds)")
    ax.set_title("Slang trainer training time per scene "
                 f"(SH0, 3000 iters; avg {avg:.1f}s over {len(scenes)} scenes)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(Path("outputs") / "slang_timings.png")
    plt.close(fig)

    print(f"{'scene':<10} {'train_s':>9} {'it/s':>7}")
    for s, t in rows:
        print(f"{s:<10} {t:>9.1f} {3000 / t:>7.1f}")
    print("-" * 28)
    print(f"{'AVERAGE':<10} {avg:>9.1f} {3000 / avg:>7.1f}")
    print("\nwrote outputs/slang_timings.png, outputs/slang_timings.csv")


if __name__ == "__main__":
    main()
