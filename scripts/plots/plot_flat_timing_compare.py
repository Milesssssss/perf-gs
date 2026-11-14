# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Compare per-tile vs flat (warp-per-bucket) backward training time.

Same-GPU A/B of outputs/train/timing_<scene>_{pertile,flat} (only --flat-bucket-backward differs); slang_timings.csv is another machine, reference only.
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


def _historical() -> dict[str, float]:
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
    hist = _historical()
    rows = []
    for s in SCENES:
        pt = _wall_s(f"timing_{s}_pertile")
        fl = _wall_s(f"timing_{s}_flat")
        if pt is None and fl is None:
            continue
        rows.append((s, hist.get(s), pt, fl))

    scenes = [r[0] for r in rows]
    pertile = [r[2] for r in rows]
    flat = [r[3] for r in rows]

    def _avg(xs):
        vals = [v for v in xs if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    pt_avg, fl_avg = _avg(pertile), _avg(flat)

    Path("outputs").mkdir(exist_ok=True)
    with (Path("outputs") / "timing_flat_vs_baseline.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "historical_prevhw_s", "pertile_s", "flat_s",
                    "flat_vs_pertile_speedup", "flat_delta_pct",
                    "pertile_psnr", "flat_psnr", "psnr_delta"])
        for s, h, pt, fl in rows:
            sp = f"{pt / fl:.3f}" if (pt and fl) else ""
            pct = f"{(fl - pt) / pt * 100:+.1f}%" if (pt and fl) else ""
            pt_ps, fl_ps = _psnr(f"timing_{s}_pertile"), _psnr(f"timing_{s}_flat")
            dps = (f"{fl_ps - pt_ps:+.3f}" if (pt_ps is not None and fl_ps is not None)
                   else "")
            w.writerow([s, f"{h:.1f}" if h else "", f"{pt:.1f}" if pt else "",
                        f"{fl:.1f}" if fl else "", sp, pct,
                        f"{pt_ps:.3f}" if pt_ps is not None else "",
                        f"{fl_ps:.3f}" if fl_ps is not None else "", dps])
        sp = f"{pt_avg / fl_avg:.3f}" if (pt_avg and fl_avg) else ""
        pct = f"{(fl_avg - pt_avg) / pt_avg * 100:+.1f}%" if (pt_avg and fl_avg) else ""
        pt_ps_all = [_psnr(f"timing_{s}_pertile") for s, *_ in rows]
        fl_ps_all = [_psnr(f"timing_{s}_flat") for s, *_ in rows]
        pt_ps_avg = float(np.mean([v for v in pt_ps_all if v is not None])) if any(v is not None for v in pt_ps_all) else None
        fl_ps_avg = float(np.mean([v for v in fl_ps_all if v is not None])) if any(v is not None for v in fl_ps_all) else None
        w.writerow(["AVERAGE", "", f"{pt_avg:.1f}", f"{fl_avg:.1f}", sp, pct,
                    f"{pt_ps_avg:.3f}" if pt_ps_avg is not None else "",
                    f"{fl_ps_avg:.3f}" if fl_ps_avg is not None else "",
                    f"{fl_ps_avg - pt_ps_avg:+.3f}" if (pt_ps_avg is not None and fl_ps_avg is not None) else ""])

    x = np.arange(len(scenes))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    b1 = ax.bar(x - width / 2, [v if v is not None else 0 for v in pertile], width,
                label="per-tile backward (baseline)", color="#1f77b4")
    b2 = ax.bar(x + width / 2, [v if v is not None else 0 for v in flat], width,
                label="flat warp-per-bucket backward", color="#2ca02c")
    for bars, vals in ((b1, pertile), (b2, flat)):
        for rect, v in zip(bars, vals):
            if v is not None:
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(pt_avg, color="#1f77b4", ls="--", lw=1.4, label=f"per-tile avg = {pt_avg:.1f}s")
    ax.axhline(fl_avg, color="#2ca02c", ls="--", lw=1.4, label=f"flat avg = {fl_avg:.1f}s")

    ax.set_xticks(x)
    ax.set_xticklabels(scenes, rotation=20)
    ax.set_ylabel("training time, iter 0 \u2192 3000 (seconds)")
    _vals = [v for v in (pertile + flat) if v is not None]
    if _vals:
        ax.set_ylim(0, max(_vals) * 1.15)
    delta = f"{(fl_avg - pt_avg) / pt_avg * 100:+.1f}%" if pt_avg else "?"
    ax.set_title("Backward A/B (same GPU): per-tile vs flat warp-per-bucket "
                 f"(SH0, 3000 iters; {pt_avg:.0f}s vs {fl_avg:.0f}s avg, {delta})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(Path("outputs") / "timing_flat_vs_baseline.png")
    plt.close(fig)

    print(f"{'scene':<10} {'pertile':>8} {'flat':>7} {'x':>7} {'delta':>8} "
          f"{'pt_psnr':>8} {'fl_psnr':>8}")
    for s, h, pt, fl in rows:
        sp = f"{pt / fl:.2f}x" if (pt and fl) else ""
        pct = f"{(fl - pt) / pt * 100:+.1f}%" if (pt and fl) else ""
        pt_ps, fl_ps = _psnr(f"timing_{s}_pertile"), _psnr(f"timing_{s}_flat")
        print(f"{s:<10} {(pt or float('nan')):>8.1f} {(fl or float('nan')):>7.1f} "
              f"{sp:>7} {pct:>8} "
              f"{(pt_ps if pt_ps is not None else float('nan')):>8.2f} "
              f"{(fl_ps if fl_ps is not None else float('nan')):>8.2f}")
    print("-" * 60)
    dpct = f"{(fl_avg - pt_avg) / pt_avg * 100:+.1f}%" if pt_avg else ""
    print(f"{'AVERAGE':<10} {'':>7} {pt_avg:>8.1f} {fl_avg:>7.1f} "
          f"{pt_avg / fl_avg:>6.2f}x {dpct:>8}")
    print("\nwrote outputs/timing_flat_vs_baseline.png, "
          "outputs/timing_flat_vs_baseline.csv")


if __name__ == "__main__":
    main()
