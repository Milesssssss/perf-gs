"""Recompute held-out LPIPS from SAVED renders, without retraining anything.

Why this exists: LPIPS was first computed with the AlexNet backbone, while 3DGS,
Mip-NeRF 360 and essentially every NeRF/splatting paper report the VGG variant.
The two are not interchangeable — measured on this trainer's own held-out bicycle
renders, AlexNet reads 0.1662 where VGG reads 0.2194, a factor of 1.32 — so the
AlexNet numbers looked better than published 30k-iteration results at 3000
iterations. Retraining 289 cells to change a post-hoc CPU metric would be absurd,
and it is unnecessary: cells run with ``--save-test-renders`` have the exact
8-bit renders that were scored, and the ground truth is on disk beside them.

This is deliberately a separate, idempotent pass rather than a change to the
training path: the renders are the evidence, and any perceptual metric can be
recomputed from them later without touching the GPU.

    .venv/Scripts/python.exe scripts/running/backfill_lpips.py --nets vgg alex

Writes ``lpips_backfill.json`` into each run directory and a combined
``lpips_backfill.csv`` under the suite output, leaving the original
``run_report.json`` untouched (the recomputation is additive evidence, not a
rewrite of what the run reported at the time).

CAVEAT it records: the original in-loop value was computed from the FLOAT render,
this one from the 8-bit PNG. On bicycle the two agree to 1e-4 for AlexNet, so the
round trip is not a meaningful source of difference — but the fact is recorded
rather than assumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

COLUMNS = ("run_name", "scene", "regime", "seed", "pixel_fraction",
           "select_mode", "net", "n_views", "lpips_mean", "lpips_median",
           "lpips_min", "lpips_max", "stored_lpips", "psnr_srgb")


def _load_png(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:                                      # noqa: BLE001
        return None


def _ground_truth_dir(run_dir: Path, manifest: dict) -> Path | None:
    """The cams/ directory holding this scene's ground truth.

    Only one cell per scene dumps it (``--write-ground-truths``), so this looks
    the owner up rather than assuming it is the current run.
    """
    own = run_dir / "cams"
    if own.is_dir() and any(own.glob("cam_*/gt.png")):
        return own
    record = (manifest.get("cells") or {}).get(run_dir.name)
    scene = (record or {}).get("scene")
    if not scene:
        return None
    for key, cell in (manifest.get("cells") or {}).items():
        if cell.get("scene") != scene:
            continue
        candidate = REPO / "outputs" / "train" / key / "cams"
        if candidate.is_dir() and any(candidate.glob("cam_*/gt.png")):
            return candidate
    return None


def backfill_run(run_dir: Path, nets: tuple[str, ...],
                 manifest: dict) -> dict | None:
    from trainer.common import perceptual

    renders = run_dir / "test_renders"
    report_path = run_dir / "run_report.json"
    if not renders.is_dir() or not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text())
    except ValueError:
        return None

    gt_dir = _ground_truth_dir(run_dir, manifest)
    if gt_dir is None:
        return None

    per_view = {int(v["cam_idx"]): v
                for v in (report.get("quality") or {}).get("per_view") or []}
    results: dict[str, list[float]] = {net: [] for net in nets}
    views: list[dict] = []
    mismatched = 0

    for path in sorted(renders.glob("cam_*.png")):
        cam_idx = int(path.stem.split("_")[1])
        rendered = _load_png(path)
        truth = _load_png(gt_dir / f"cam_{cam_idx:02d}" / "gt.png")
        if rendered is None or truth is None:
            continue
        if rendered.shape != truth.shape:
            # A resolution mismatch would silently compare different content;
            # count it rather than resize into a number nobody can trace.
            mismatched += 1
            continue
        row = {"cam_idx": cam_idx,
               "stored_lpips": (per_view.get(cam_idx) or {}).get("lpips")}
        for net in nets:
            value = perceptual.lpips(rendered, truth, net=net)
            if value is not None:
                results[net].append(value)
                row[f"lpips_{net}"] = value
        views.append(row)

    if not any(results.values()):
        return None
    stored = (report.get("quality") or {}).get("lpips")
    record = {
        "run_name": run_dir.name,
        "ground_truth_dir": str(gt_dir.relative_to(REPO)),
        "n_views": len(views),
        "shape_mismatched_views": mismatched,
        "stored_lpips": stored,
        "source": "8-bit saved PNG renders (original was the float render)",
        "per_net": {
            net: {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "n": len(values),
            } for net, values in results.items() if values},
        "per_view": views,
    }
    (run_dir / "lpips_backfill.json").write_text(json.dumps(record, indent=2))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute held-out LPIPS from saved renders.")
    parser.add_argument("--suite-root", default=str(REPO / "outputs" / "paper_stbn"))
    parser.add_argument("--nets", nargs="+", default=["vgg", "alex"],
                        help="LPIPS backbones; vgg is the published convention")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    manifest_path = suite_root / "manifest.json"
    manifest = (json.loads(manifest_path.read_text())
                if manifest_path.is_file() else {})
    cells = manifest.get("cells") or {}

    rows: list[dict] = []
    candidates = [key for key, cell in cells.items()
                  if cell.get("has_saved_renders")]
    print(f"[lpips-backfill] {len(candidates)} cells have saved renders; "
          f"nets={args.nets}", flush=True)

    for index, key in enumerate(sorted(candidates), start=1):
        cell = cells[key]
        run_dir = REPO / "outputs" / "train" / key
        record = backfill_run(run_dir, tuple(args.nets), manifest)
        if record is None:
            print(f"[lpips-backfill] skip {key} (renders or GT missing)",
                  flush=True)
            continue
        for net, block in record["per_net"].items():
            rows.append({
                "run_name": key,
                "scene": cell.get("scene"),
                "regime": cell.get("regime"),
                "seed": cell.get("seed"),
                "pixel_fraction": cell.get("pixel_fraction"),
                "select_mode": cell.get("select_mode"),
                "net": net,
                "n_views": block["n"],
                "lpips_mean": block["mean"],
                "lpips_median": block["median"],
                "lpips_min": block["min"],
                "lpips_max": block["max"],
                "stored_lpips": record["stored_lpips"],
                "psnr_srgb": cell.get("psnr_srgb"),
            })
        if index % 10 == 0 or index == len(candidates):
            print(f"[lpips-backfill] {index}/{len(candidates)}", flush=True)

    out_path = Path(args.out) if args.out else (
        suite_root / "tables" / "lpips_backfill.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(["" if row.get(c) is None else row.get(c)
                             for c in COLUMNS])
    print(f"[lpips-backfill] wrote {out_path} ({len(rows)} rows)", flush=True)

    # Consistency check on the one thing that could invalidate the whole pass.
    alex = [r for r in rows if r["net"] == "alex"
            and r["stored_lpips"] is not None]
    if alex:
        deltas = [abs(r["lpips_mean"] - r["stored_lpips"]) for r in alex]
        print(f"[lpips-backfill] PNG-vs-float AlexNet agreement over "
              f"{len(alex)} cells: max |delta| = {max(deltas):.6f}, "
              f"mean = {statistics.mean(deltas):.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
