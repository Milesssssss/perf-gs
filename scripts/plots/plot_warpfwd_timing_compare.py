# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Compare baseline thread-per-pixel forward vs warp-per-pixel forward training time.

Same-hardware A/B: for each scene we ran the pure 0->3000 training loop twice
with identical flags + identical (flat) backward, differing ONLY in the forward
rasterizer (--warp-per-pixel-forward). Both are read from
outputs/train/timing_<scene>_{threadfwd,warpfwd}/log.jsonl
(train_complete.train_wall_s) plus test_psnr.json for the quality check.

The warp-per-pixel forward trades the thread-per-pixel kernel's broadcast load
(256 threads/WG read the SAME splat) for a 32-way gather (one warp/pixel, 32
lanes read 32 different splats). It gathers the compact 36 B BlendData record to
keep that gather L1TEX-cheap. This plot shows whether the wave-scan latency win
beats the extra memory traffic end-to-end.

Writes outputs/timing_warpfwd_vs_baseline.{csv,png}.
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


def main() -> None:
    rows = []
    for s in SCENES:
        base = _wall_s(f"timing_{s}_threadfwd")
        warp = _wall_s(f"timing_{s}_warpfwd")
        if base is None and warp is None:
            continue
        rows.append((s, base, warp))

    scenes = [r[0] for r in rows]
    baseline = [r[1] for r in rows]
    warp = [r[2] for r in rows]

    def _avg(xs):
        vals = [v for v in xs if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    base_avg, warp_avg = _avg(baseline), _avg(warp)

    Path("outputs").mkdir(exist_ok=True)
    with (Path("outputs") / "timing_warpfwd_vs_baseline.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "threadfwd_s", "warpfwd_s",
                    "warp_vs_thread_speedup", "warp_delta_pct",
                    "threadfwd_psnr", "warpfwd_psnr", "psnr_delta"])
        for s, base, wp in rows:
            sp = f"{base / wp:.3f}" if (base and wp) else ""
            pct = f"{(wp - base) / base * 100:+.1f}%" if (base and wp) else ""
            base_ps, wp_ps = _psnr(f"timing_{s}_threadfwd"), _psnr(f"timing_{s}_warpfwd")
            dps = (f"{wp_ps - base_ps:+.3f}"
                   if (base_ps is not None and wp_ps is not None) else "")
            w.writerow([s, f"{base:.1f}" if base else "",
                        f"{wp:.1f}" if wp else "", sp, pct,
                        f"{base_ps:.3f}" if base_ps is not None else "",
                        f"{wp_ps:.3f}" if wp_ps is not None else "", dps])
        sp = f"{base_avg / warp_avg:.3f}" if (base_avg and warp_avg) else ""
        pct = f"{(warp_avg - base_avg) / base_avg * 100:+.1f}%" if (base_avg and warp_avg) else ""
        base_ps_all = [_psnr(f"timing_{s}_threadfwd") for s, *_ in rows]
        wp_ps_all = [_psnr(f"timing_{s}_warpfwd") for s, *_ in rows]
        base_ps_avg = (float(np.mean([v for v in base_ps_all if v is not None]))
                       if any(v is not None for v in base_ps_all) else None)
        wp_ps_avg = (float(np.mean([v for v in wp_ps_all if v is not None]))
                     if any(v is not None for v in wp_ps_all) else None)
        w.writerow(["AVERAGE", f"{base_avg:.1f}", f"{warp_avg:.1f}", sp, pct,
                    f"{base_ps_avg:.3f}" if base_ps_avg is not None else "",
                    f"{wp_ps_avg:.3f}" if wp_ps_avg is not None else "",
                    f"{wp_ps_avg - base_ps_avg:+.3f}" if (base_ps_avg is not None and wp_ps_avg is not None) else ""])

    # Grouped bar chart: baseline vs warp (current hardware).
    x = np.arange(len(scenes))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    b1 = ax.bar(x - width / 2, [v if v is not None else 0 for v in baseline], width,
                label="thread-per-pixel forward (baseline)", color="#1f77b4")
    b2 = ax.bar(x + width / 2, [v if v is not None else 0 for v in warp], width,
                label="warp-per-pixel forward", color="#d62728")
    for bars, vals in ((b1, baseline), (b2, warp)):
        for rect, v in zip(bars, vals):
            if v is not None:
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(base_avg, color="#1f77b4", ls="--", lw=1.4, label=f"baseline avg = {base_avg:.1f}s")
    ax.axhline(warp_avg, color="#d62728", ls="--", lw=1.4, label=f"warp avg = {warp_avg:.1f}s")

    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=20)
    ax.set_ylabel("training time, iter 0 \u2192 3000 (seconds)")
    _vals = [v for v in (baseline + warp) if v is not None]
    if _vals:
        ax.set_ylim(0, max(_vals) * 1.15)
    delta = f"{(warp_avg - base_avg) / base_avg * 100:+.1f}%" if base_avg else "?"
    ax.set_title("Forward A/B (same GPU): thread-per-pixel vs warp-per-pixel "
                 f"(SH0, 3000 iters; {base_avg:.0f}s vs {warp_avg:.0f}s avg, {delta})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(Path("outputs") / "timing_warpfwd_vs_baseline.png")
    plt.close(fig)

    print(f"{'scene':<10} {'thread':>8} {'warp':>7} {'x':>7} {'delta':>8} "
          f"{'th_psnr':>8} {'wp_psnr':>8}")
    for s, base, wp in rows:
        sp = f"{base / wp:.2f}x" if (base and wp) else ""
        pct = f"{(wp - base) / base * 100:+.1f}%" if (base and wp) else ""
        base_ps, wp_ps = _psnr(f"timing_{s}_threadfwd"), _psnr(f"timing_{s}_warpfwd")
        print(f"{s:<10} {(base or float('nan')):>8.1f} {(wp or float('nan')):>7.1f} "
              f"{sp:>7} {pct:>8} "
              f"{(base_ps if base_ps is not None else float('nan')):>8.2f} "
              f"{(wp_ps if wp_ps is not None else float('nan')):>8.2f}")
    print("-" * 60)
    dpct = f"{(warp_avg - base_avg) / base_avg * 100:+.1f}%" if base_avg else ""
    print(f"{'AVERAGE':<10} {'':>7} {base_avg:>8.1f} {warp_avg:>7.1f} "
          f"{base_avg / warp_avg:>6.2f}x {dpct:>8}")
    print("\nwrote outputs/timing_warpfwd_vs_baseline.png, "
          "outputs/timing_warpfwd_vs_baseline.csv")


if __name__ == "__main__":
    main()
