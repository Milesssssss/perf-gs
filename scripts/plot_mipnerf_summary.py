"""Aggregate per-scene psnr_curve.json files into a paper-style summary.

Given a folder whose subdirectories each hold a ``psnr_curve.json`` (as written
by ``v2_train.py --psnr-curve``), this produces, in that same folder:
  - psnr_avg_curve.png  paper-style PSNR-vs-iteration plot: faint per-scene
                        curves + a bold mean-over-scenes curve.
  - final_summary.csv   per-scene final (last-iter) PSNR + #Gaussians, plus the
                        average row.
  - final_summary.png   the same table rendered as an image.

The average (both the bold curve and the table AVERAGE row) is taken over the
INCLUDED scenes only; scenes passed via --exclude are still plotted/listed for
context but left out of the average (e.g. the Mip-NeRF 360 released-7 protocol
drops flowers + treehill).

Usage:
    .venv/Scripts/python.exe scripts/plot_mipnerf_summary.py [FOLDER] [--exclude flowers,treehill]
    (FOLDER defaults to outputs/train/mipnerf360_runs)
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

# Canonical Mip-NeRF 360 ordering: outdoor scenes then indoor scenes.
SCENE_ORDER = ["bicycle", "flowers", "garden", "stump", "treehill",
               "bonsai", "counter", "kitchen", "room"]


def load_scene(folder: Path) -> dict | None:
    p = folder / "psnr_curve.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["_name"] = folder.name
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default="outputs/train/mipnerf360_runs",
                    type=Path, help="folder of per-scene run subdirs")
    ap.add_argument("--exclude", default="",
                    help="comma-separated scene names to leave OUT of the average "
                         "(still plotted/listed). e.g. flowers,treehill")
    args = ap.parse_args()

    root = args.folder
    if not root.is_dir():
        raise SystemExit(f"folder not found: {root}")
    excluded = {s.strip() for s in args.exclude.split(",") if s.strip()}

    scenes = {}
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            d = load_scene(sub)
            if d is not None:
                scenes[sub.name] = d
    if not scenes:
        raise SystemExit(f"no scene psnr_curve.json files under {root}")

    # Order: canonical first, then any extras alphabetically.
    names = [s for s in SCENE_ORDER if s in scenes]
    names += [s for s in sorted(scenes) if s not in names]
    included = [n for n in names if n not in excluded]
    if not included:
        raise SystemExit("every scene was excluded — nothing to average")

    # ---- Averaged PSNR-vs-iteration curve (mean over INCLUDED shared iters). ----
    common = set(scenes[included[0]]["iters"])
    for n in included[1:]:
        common &= set(scenes[n]["iters"])
    iters = sorted(common)
    aligned = {}
    for n in names:
        m = {int(i): float(p) for i, p in zip(scenes[n]["iters"], scenes[n]["psnr_srgb"])}
        aligned[n] = np.array([m.get(i, np.nan) for i in iters], dtype=float)
    mean_curve = np.mean(np.stack([aligned[n] for n in included]), axis=0)

    excl_note = (f" (excl. {', '.join(n for n in names if n in excluded)})"
                 if excluded else "")

    fig, ax = plt.subplots(figsize=(9, 6), dpi=140)
    for n in names:
        style = ":" if n in excluded else "-"
        lab = f"{n} (excl.)" if n in excluded else n
        ax.plot(iters, aligned[n], style, lw=1.0, alpha=0.35, label=lab)
    ax.plot(iters, mean_curve, "-", color="black", lw=2.8,
            label=f"Average ({len(included)} scenes){excl_note}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("test PSNR (dB)")
    ax.set_title(f"Mip-NeRF 360 — PSNR vs iteration (avg over {len(included)} scenes)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    curve_path = root / "psnr_avg_curve.png"
    fig.savefig(curve_path)
    plt.close(fig)

    # ---- Final-metrics table (last-iter PSNR + #Gaussians). ----
    rows = []
    for n in names:
        rows.append((n, int(scenes[n]["iters"][-1]),
                     float(scenes[n]["psnr_srgb"][-1]),
                     int(scenes[n]["n_gaussians"][-1])))
    inc_rows = [r for r in rows if r[0] not in excluded]
    avg_psnr = float(np.mean([r[2] for r in inc_rows]))
    avg_ng = float(np.mean([r[3] for r in inc_rows]))

    # CSV.
    csv_path = root / "final_summary.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scene", "final_iter", "final_psnr_srgb", "final_n_gaussians",
                    "in_average"])
        for n, it, ps, ng in rows:
            w.writerow([n, it, f"{ps:.3f}", ng, int(n not in excluded)])
        w.writerow([f"AVERAGE({len(inc_rows)})", "", f"{avg_psnr:.3f}",
                    f"{avg_ng:.0f}", ""])

    # Rendered table image.
    table_rows = [[(n + " (excl.)" if n in excluded else n),
                   f"{ps:.2f}", f"{ng:,}"] for n, _it, ps, ng in rows]
    avg_label = f"AVERAGE ({len(inc_rows)} scenes)"
    table_rows.append([avg_label, f"{avg_psnr:.2f}", f"{avg_ng:,.0f}"])
    col_labels = ["Scene", "Final PSNR (dB)", "# Gaussians"]

    fig, ax = plt.subplots(figsize=(7.6, 0.5 + 0.42 * (len(table_rows) + 1)), dpi=140)
    ax.axis("off")
    ttl = (f"Mip-NeRF 360 — final PSNR & Gaussian count "
           f"(iter {rows[0][1]}); avg over {len(inc_rows)} scenes{excl_note}")
    ax.set_title(ttl, pad=12, fontsize=10)
    tbl = ax.table(cellText=table_rows, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.4)
    ncols = len(col_labels)
    for c in range(ncols):
        tbl[(0, c)].set_text_props(weight="bold")
        tbl[(0, c)].set_facecolor("#dddddd")
        tbl[(len(table_rows), c)].set_text_props(weight="bold")
        tbl[(len(table_rows), c)].set_facecolor("#eef3fb")
    # Grey + italicize excluded scene rows (display rows 1..len(rows)).
    for ri, (n, *_rest) in enumerate(rows, start=1):
        if n in excluded:
            for c in range(ncols):
                tbl[(ri, c)].set_text_props(color="#999999", style="italic")
    table_path = root / "final_summary.png"
    fig.tight_layout()
    fig.savefig(table_path, bbox_inches="tight")
    plt.close(fig)

    print(f"scenes ({len(names)}): {', '.join(names)}")
    if excluded:
        print(f"excluded from average: {', '.join(sorted(excluded))}")
    print(f"AVERAGE ({len(inc_rows)} scenes) final PSNR = {avg_psnr:.2f} dB   "
          f"avg #Gaussians = {avg_ng:,.0f}")
    print(f"wrote:\n  {curve_path}\n  {csv_path}\n  {table_path}")


if __name__ == "__main__":
    main()
