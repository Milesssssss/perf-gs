# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Compare forwardRasterize before vs after the blend_data change.

Same-GPU A/B of outputs/train/timing_<scene>_{threadfwd,blendfwd} (only the forward gather differs); slang_timings.csv is another machine, reference only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCENES = ["bicycle", "flowers", "garden", "stump", "treehill",
          "bonsai", "counter", "kitchen", "room"]
ROOT = Path("outputs/train")


def _wall_s(run_name: str) -> float | None:
    p = ROOT / run_name / "log.jsonl"
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


def _psnr(run_name: str) -> float | None:
    p = ROOT / run_name / "test_psnr.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["psnr_srgb"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _reference() -> dict[str, float]:
    p = Path("outputs/slang_timings.csv")
    out: dict[str, float] = {}
    if not p.exists():
        return out
    with p.open() as fh:
        for row in csv.DictReader(fh):
            s = row["scene"]
            if s and s != "AVERAGE":
                try:
                    out[s] = float(row["train_wall_s"])
                except (KeyError, ValueError):
                    pass
    return out


def main() -> None:
    ref = _reference()
    rows = []
    for s in SCENES:
        fat = _wall_s(f"timing_{s}_threadfwd")
        blend = _wall_s(f"timing_{s}_blendfwd")
        if fat is None and blend is None:
            continue
        rows.append((s, fat, blend))

    scenes = [r[0] for r in rows]
    fat_t = [r[1] for r in rows]
    blend_t = [r[2] for r in rows]

    def _avg(xs):
        vals = [v for v in xs if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    fat_avg, blend_avg = _avg(fat_t), _avg(blend_t)

    Path("outputs").mkdir(exist_ok=True)
    with (Path("outputs") / "timing_blendfwd_vs_threadfwd.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "ref_otherhw_s", "fat_threadfwd_s", "blend_fwd_s",
                    "blend_vs_fat_speedup", "blend_delta_pct",
                    "fat_psnr", "blend_psnr", "psnr_delta"])
        for s, fat, bl in rows:
            sp = f"{fat / bl:.3f}" if (fat and bl) else ""
            pct = f"{(bl - fat) / fat * 100:+.1f}%" if (fat and bl) else ""
            fps, bps = _psnr(f"timing_{s}_threadfwd"), _psnr(f"timing_{s}_blendfwd")
            dps = (f"{bps - fps:+.3f}" if (fps is not None and bps is not None) else "")
            w.writerow([s, f"{ref.get(s):.1f}" if ref.get(s) else "",
                        f"{fat:.1f}" if fat else "", f"{bl:.1f}" if bl else "",
                        sp, pct,
                        f"{fps:.3f}" if fps is not None else "",
                        f"{bps:.3f}" if bps is not None else "", dps])
        sp = f"{fat_avg / blend_avg:.3f}" if (fat_avg and blend_avg) else ""
        pct = f"{(blend_avg - fat_avg) / fat_avg * 100:+.1f}%" if (fat_avg and blend_avg) else ""
        w.writerow(["AVERAGE", "", f"{fat_avg:.1f}", f"{blend_avg:.1f}", sp, pct, "", "", ""])

    x = np.arange(len(scenes))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    b1 = ax.bar(x - width / 2, [v if v is not None else 0 for v in fat_t], width,
                label="forwardRasterize: fat gather (before)", color="#7f7f7f")
    b2 = ax.bar(x + width / 2, [v if v is not None else 0 for v in blend_t], width,
                label="forwardRasterize: BlendData gather (after)", color="#2ca02c")
    for bars, vals in ((b1, fat_t), (b2, blend_t)):
        for rect, v in zip(bars, vals):
            if v is not None:
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(fat_avg, color="#7f7f7f", ls="--", lw=1.4, label=f"fat avg = {fat_avg:.1f}s")
    ax.axhline(blend_avg, color="#2ca02c", ls="--", lw=1.4, label=f"blend avg = {blend_avg:.1f}s")

    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=20)
    ax.set_ylabel("training time, iter 0 \u2192 3000 (seconds)")
    _vals = [v for v in (fat_t + blend_t) if v is not None]
    if _vals:
        ax.set_ylim(0, max(_vals) * 1.15)
    delta = f"{(blend_avg - fat_avg) / fat_avg * 100:+.1f}%" if fat_avg else "?"
    ax.set_title("forwardRasterize gather A/B (same GPU): fat structs vs BlendData "
                 f"(SH0, 3000 iters; {fat_avg:.0f}s vs {blend_avg:.0f}s avg, {delta})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(Path("outputs") / "timing_blendfwd_vs_threadfwd.png")
    plt.close(fig)

    print(f"{'scene':<10} {'fat':>8} {'blend':>7} {'x':>7} {'delta':>8} {'ref_hw':>8}")
    for s, fat, bl in rows:
        sp = f"{fat / bl:.2f}x" if (fat and bl) else ""
        pct = f"{(bl - fat) / fat * 100:+.1f}%" if (fat and bl) else ""
        r = ref.get(s)
        print(f"{s:<10} {(fat or float('nan')):>8.1f} {(bl or float('nan')):>7.1f} "
              f"{sp:>7} {pct:>8} {(r if r is not None else float('nan')):>8.1f}")
    print("-" * 56)
    dpct = f"{(blend_avg - fat_avg) / fat_avg * 100:+.1f}%" if fat_avg else ""
    print(f"{'AVERAGE':<10} {fat_avg:>8.1f} {blend_avg:>7.1f} "
          f"{fat_avg / blend_avg:>6.2f}x {dpct:>8}")
    print("\n(ref_hw = slang_timings.csv, DIFFERENT machine -- reference only)")
    print("wrote outputs/timing_blendfwd_vs_threadfwd.png, "
          "outputs/timing_blendfwd_vs_threadfwd.csv")


if __name__ == "__main__":
    main()
