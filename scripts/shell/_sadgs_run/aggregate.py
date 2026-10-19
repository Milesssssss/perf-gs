# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Aggregate SADGS reference PSNR across scenes and compare to the Slang trainer.

Reads:
  - SADGS: outputs/SADGS_ref_sh0_<scene>/results.json  (canonical metrics.py)
  - Slang: outputs/train/mipnerf360_run_1/<scene>/psnr_curve.json (final point)

Writes into outputs/:
  - sadgs_ref_summary.csv    per-scene SADGS PSNR/SSIM/LPIPS + averages
  - sadgs_vs_slang.csv       side-by-side final PSNR (SADGS vs Slang) + averages
  - sadgs_vs_slang.png       grouped bar chart of the two
and prints the table.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("outputs")
SLANG_ROOT = ROOT / "train" / "mipnerf360_run_1"
# outdoor then indoor (canonical Mip-NeRF 360 order).
SCENES = ["bicycle", "flowers", "garden", "stump", "treehill",
          "bonsai", "counter", "kitchen", "room"]
EXCLUDE = {"flowers", "treehill"}  # released-7 average


def sadgs_metrics(scene: str) -> dict | None:
    p = ROOT / f"SADGS_ref_sh0_{scene}" / "results.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())["ours_3000"]
    return {"psnr": float(d["PSNR"]), "ssim": float(d["SSIM"]), "lpips": float(d["LPIPS"])}


def slang_psnr(scene: str) -> float | None:
    p = SLANG_ROOT / scene / "psnr_curve.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return float(d["psnr_srgb"][-1])


def main() -> None:
    rows = []
    for s in SCENES:
        ref = sadgs_metrics(s)
        sl = slang_psnr(s)
        rows.append({"scene": s,
                     "sadgs_psnr": ref["psnr"] if ref else None,
                     "sadgs_ssim": ref["ssim"] if ref else None,
                     "sadgs_lpips": ref["lpips"] if ref else None,
                     "slang_psnr": sl})

    def avg(key, scenes):
        vals = [r[key] for r in rows if r["scene"] in scenes and r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    all9 = set(SCENES)
    seven = all9 - EXCLUDE

    # --- SADGS-only summary CSV ---
    with (ROOT / "sadgs_ref_summary.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "psnr", "ssim", "lpips"])
        for r in rows:
            w.writerow([r["scene"],
                        f"{r['sadgs_psnr']:.3f}" if r["sadgs_psnr"] is not None else "",
                        f"{r['sadgs_ssim']:.4f}" if r["sadgs_ssim"] is not None else "",
                        f"{r['sadgs_lpips']:.4f}" if r["sadgs_lpips"] is not None else ""])
        w.writerow([f"AVG_9", f"{avg('sadgs_psnr', all9):.3f}",
                    f"{avg('sadgs_ssim', all9):.4f}", f"{avg('sadgs_lpips', all9):.4f}"])
        w.writerow([f"AVG_7_excl_flowers_treehill", f"{avg('sadgs_psnr', seven):.3f}",
                    f"{avg('sadgs_ssim', seven):.4f}", f"{avg('sadgs_lpips', seven):.4f}"])

    # --- SADGS vs Slang CSV ---
    with (ROOT / "sadgs_vs_slang.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "sadgs_psnr", "slang_psnr", "delta_slang_minus_sadgs"])
        for r in rows:
            d = (r["slang_psnr"] - r["sadgs_psnr"]
                 if (r["slang_psnr"] is not None and r["sadgs_psnr"] is not None) else None)
            w.writerow([r["scene"],
                        f"{r['sadgs_psnr']:.3f}" if r["sadgs_psnr"] is not None else "",
                        f"{r['slang_psnr']:.3f}" if r["slang_psnr"] is not None else "",
                        f"{d:+.3f}" if d is not None else ""])
        for label, sc in [("AVG_9", all9), ("AVG_7_excl_flowers_treehill", seven)]:
            w.writerow([label, f"{avg('sadgs_psnr', sc):.3f}", f"{avg('slang_psnr', sc):.3f}",
                        f"{avg('slang_psnr', sc) - avg('sadgs_psnr', sc):+.3f}"])

    # --- bar chart ---
    x = np.arange(len(SCENES))
    ref_v = [r["sadgs_psnr"] for r in rows]
    sl_v = [r["slang_psnr"] for r in rows]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    ax.bar(x - 0.2, ref_v, 0.4, label="SADGS reference", color="#d62728")
    ax.bar(x + 0.2, sl_v, 0.4, label="Slang trainer", color="#1f77b4")
    for xi, (rv, sv) in enumerate(zip(ref_v, sl_v)):
        if rv is not None: ax.text(xi - 0.2, rv + 0.1, f"{rv:.1f}", ha="center", va="bottom", fontsize=7)
        if sv is not None: ax.text(xi + 0.2, sv + 0.1, f"{sv:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(SCENES, rotation=20)
    ax.set_ylabel("test PSNR (dB)")
    ax.set_title("SADGS reference vs Slang trainer — SH0, 3000 iters, same holdout (llffhold=8)")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT / "sadgs_vs_slang.png"); plt.close(fig)

    # --- print ---
    print(f"{'scene':<10} {'SADGS':>8} {'Slang':>8} {'delta':>8}")
    for r in rows:
        d = (r["slang_psnr"] - r["sadgs_psnr"]) if (r["slang_psnr"] and r["sadgs_psnr"]) else None
        print(f"{r['scene']:<10} {r['sadgs_psnr']:>8.2f} "
              f"{(r['slang_psnr'] if r['slang_psnr'] else float('nan')):>8.2f} "
              f"{(d if d is not None else float('nan')):>+8.2f}")
    print("-" * 38)
    print(f"{'AVG(9)':<10} {avg('sadgs_psnr', all9):>8.2f} {avg('slang_psnr', all9):>8.2f} "
          f"{avg('slang_psnr', all9) - avg('sadgs_psnr', all9):>+8.2f}")
    print(f"{'AVG(7)':<10} {avg('sadgs_psnr', seven):>8.2f} {avg('slang_psnr', seven):>8.2f} "
          f"{avg('slang_psnr', seven) - avg('sadgs_psnr', seven):>+8.2f}   (excl flowers, treehill)")
    print(f"\nwrote outputs/sadgs_ref_summary.csv, outputs/sadgs_vs_slang.csv, outputs/sadgs_vs_slang.png")


if __name__ == "__main__":
    main()
