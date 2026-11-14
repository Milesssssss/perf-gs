# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Experiment G: contact sheets, shared-scale error maps and ranked patches for every held-out view.

Crops are ranked by measured local-PSNR deltas, including where STBN loses, never chosen by eye.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as cfg
from . import figures as figs
from . import tables
from .results import ResultSet

# Fixed panel order for every contact sheet.
PANEL_ORDER: tuple[str, ...] = ("gt", "full", "iid_white", "indep_blue", "stbn")
PANEL_LABELS = {"gt": "Ground truth", "full": "Dense (100%)",
                "iid_white": "IID white", "indep_blue": "Indep. blue",
                "stbn": "STBN"}

# Big enough for a meaningful local PSNR, small enough to crop; 50% overlap catches boundary features.
PATCH_SIZE = 64
PATCH_STRIDE = 32
# Flat patches are skipped: a huge PSNR delta on a featureless wall is a numerical artifact.
MIN_PATCH_STD = 0.02
TOP_CANDIDATES = 40

CANDIDATE_COLUMNS = (
    "ranking", "rank", "scene", "dataset_family", "cam_idx", "image_name",
    "x0", "y0", "x1", "y1", "patch_std",
    "psnr_gt_dense", "psnr_gt_iid", "psnr_gt_indep", "psnr_gt_stbn",
    "delta_stbn_minus_iid", "delta_stbn_minus_indep", "delta_indep_minus_iid",
    "crop_png",
)

VIEW_COLUMNS = (
    "scene", "dataset_family", "cam_idx", "image_name",
    "psnr_full", "psnr_iid_white", "psnr_indep_blue", "psnr_stbn",
    "delta_stbn_minus_iid", "delta_stbn_minus_indep",
    "contact_sheet", "error_sheet",
)


def _load_png(path: Path) -> np.ndarray | None:
    try:
        from PIL import Image
        with Image.open(path) as handle:
            return np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:                                      # noqa: BLE001
        return None


def _gt_dir(results: ResultSet, scene: str) -> Path | None:
    """The camera directory of whichever cell dumped this scene's ground truth."""
    for result in results.select(scene=scene, regime="full", fraction=1.0,
                                 seed=cfg.PRIMARY_SEED):
        candidate = result.cell.run_dir / "cams"
        if candidate.is_dir():
            return candidate
    return None


def _patch_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * np.log10(mse))


def _scan_patches(images: dict[str, np.ndarray], scene: str, cam_idx: int,
                  image_name: str) -> list[dict]:
    """Score every patch of one view under all four regimes."""
    gt = images["gt"]
    height, width = gt.shape[:2]
    rows: list[dict] = []
    for y0 in range(0, max(1, height - PATCH_SIZE + 1), PATCH_STRIDE):
        for x0 in range(0, max(1, width - PATCH_SIZE + 1), PATCH_STRIDE):
            y1, x1 = y0 + PATCH_SIZE, x0 + PATCH_SIZE
            reference = gt[y0:y1, x0:x1]
            spread = float(reference.std())
            if spread < MIN_PATCH_STD:
                continue
            scores = {}
            for regime in ("full", "iid_white", "indep_blue", "stbn"):
                image = images.get(regime)
                if image is None:
                    scores[regime] = None
                    continue
                scores[regime] = _patch_psnr(image[y0:y1, x0:x1], reference)
            if any(scores.get(r) is None for r in
                   ("iid_white", "indep_blue", "stbn")):
                continue
            rows.append({
                "scene": scene,
                "dataset_family": cfg.scene_family(scene),
                "cam_idx": cam_idx,
                "image_name": image_name,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "patch_std": spread,
                "psnr_gt_dense": scores.get("full"),
                "psnr_gt_iid": scores["iid_white"],
                "psnr_gt_indep": scores["indep_blue"],
                "psnr_gt_stbn": scores["stbn"],
                "delta_stbn_minus_iid": scores["stbn"] - scores["iid_white"],
                "delta_stbn_minus_indep": scores["stbn"] - scores["indep_blue"],
                "delta_indep_minus_iid": (scores["indep_blue"]
                                          - scores["iid_white"]),
            })
    return rows


def _contact_sheet(images: dict[str, np.ndarray], out_path: Path, *,
                   title: str, per_view_psnr: dict) -> bool:
    plt = figs.pyplot()
    if plt is None:
        return False
    panels = [p for p in PANEL_ORDER if images.get(p) is not None]
    if len(panels) < 2:
        return False
    height, width = images[panels[0]].shape[:2]
    aspect = height / max(width, 1)
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(2.0 * len(panels),
                                      2.0 * aspect + 0.4), squeeze=False)
    for column, panel in enumerate(panels):
        ax = axes[0][column]
        ax.imshow(np.clip(images[panel], 0.0, 1.0))
        ax.set_xticks([]); ax.set_yticks([])
        label = PANEL_LABELS.get(panel, panel)
        value = per_view_psnr.get(panel)
        if panel != "gt" and value is not None:
            label += f"\n{value:.2f} dB"
        ax.set_title(label, fontsize=6.5)
    fig.suptitle(title, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figs.save(fig, out_path.parent, out_path.stem, also_png=True)
    return True


def _error_sheet(images: dict[str, np.ndarray], out_path: Path,
                 title: str) -> bool:
    """Absolute RGB error against GT on one shared scale (per-panel autoscaling would flatter the worst regime)."""
    plt = figs.pyplot()
    if plt is None:
        return False
    gt = images.get("gt")
    if gt is None:
        return False
    panels = [p for p in PANEL_ORDER[1:] if images.get(p) is not None]
    if not panels:
        return False
    errors = {p: np.abs(images[p] - gt).mean(axis=-1) for p in panels}
    vmax = float(max(e.max() for e in errors.values())) or 1e-6

    height, width = gt.shape[:2]
    aspect = height / max(width, 1)
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(2.0 * len(panels), 2.0 * aspect + 0.5),
                             squeeze=False)
    image = None
    for column, panel in enumerate(panels):
        ax = axes[0][column]
        image = ax.imshow(errors[panel], cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{PANEL_LABELS.get(panel, panel)}\n"
                     f"MAE {errors[panel].mean():.4f}", fontsize=6.5)
    if image is not None:
        bar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.025)
        bar.set_label(f"|error| (shared scale, max {vmax:.3f})", fontsize=6)
    fig.suptitle(title, fontsize=7)
    figs.save(fig, out_path.parent, out_path.stem, also_png=True)
    return True


def _save_crop(images: dict[str, np.ndarray], row: dict, out_path: Path) -> bool:
    """A GT|dense|IID|indep|STBN strip of one candidate patch."""
    plt = figs.pyplot()
    if plt is None:
        return False
    y0, y1, x0, x1 = row["y0"], row["y1"], row["x0"], row["x1"]
    panels = [p for p in PANEL_ORDER if images.get(p) is not None]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(1.35 * len(panels), 1.6), squeeze=False)
    keys = {"gt": None, "full": "psnr_gt_dense", "iid_white": "psnr_gt_iid",
            "indep_blue": "psnr_gt_indep", "stbn": "psnr_gt_stbn"}
    for column, panel in enumerate(panels):
        ax = axes[0][column]
        ax.imshow(np.clip(images[panel][y0:y1, x0:x1], 0.0, 1.0),
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        key = keys.get(panel)
        value = row.get(key) if key else None
        ax.set_title(PANEL_LABELS.get(panel, panel)
                     + (f"\n{value:.2f} dB" if value is not None else ""),
                     fontsize=6)
    fig.suptitle(f"{row['scene']} cam {row['cam_idx']} "
                 f"[{x0}:{x1}, {y0}:{y1}]  "
                 f"STBN$-$IID {row['delta_stbn_minus_iid']:+.2f} dB, "
                 f"STBN$-$indep {row['delta_stbn_minus_indep']:+.2f} dB",
                 fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    figs.save(fig, out_path.parent, out_path.stem, also_png=True)
    return True


def run(results: ResultSet, out: dict, *, scenes=cfg.SCENES,
        fraction: float = cfg.PRIMARY_FRACTION, seed: int = cfg.PRIMARY_SEED,
        max_views_per_scene: int | None = None) -> dict:
    """Build every contact sheet, error map and ranked candidate crop."""
    qualitative: Path = out["qualitative"]
    qualitative.mkdir(parents=True, exist_ok=True)

    view_rows: list[dict] = []
    all_patches: list[dict] = []
    skipped: dict[str, str] = {}

    for scene in scenes:
        cells: dict[str, object] = {}
        for regime in cfg.REGIMES:
            use_fraction = 1.0 if regime == "full" else fraction
            found = results.select(scene=scene, regime=regime,
                                   fraction=use_fraction, seed=seed)
            if found:
                cells[regime] = found[0]
        missing = [r for r in cfg.REGIMES if r not in cells]
        if missing:
            skipped[scene] = f"missing regimes: {', '.join(missing)}"
            continue
        gt_root = _gt_dir(results, scene)
        if gt_root is None:
            skipped[scene] = "no ground-truth dump found for this scene"
            continue

        per_view = {r: {v.get("cam_idx"): v
                        for v in cells[r].per_view} for r in cfg.REGIMES}
        cam_indices = sorted(per_view["stbn"])
        if max_views_per_scene:
            cam_indices = cam_indices[:max_views_per_scene]

        scene_dir = qualitative / scene
        sheets_dir = scene_dir / "contact_sheets"
        errors_dir = scene_dir / "error_maps"

        for cam_idx in cam_indices:
            gt = _load_png(gt_root / f"cam_{cam_idx:02d}" / "gt.png")
            if gt is None:
                continue
            images: dict[str, np.ndarray] = {"gt": gt}
            for regime in cfg.REGIMES:
                images[regime] = _load_png(cells[regime].render_path(cam_idx))
            if any(images.get(r) is None for r in cfg.REGIMES):
                continue
            # A GT/render resolution mismatch would silently misalign every error map.
            shapes = {images[k].shape[:2] for k in images}
            if len(shapes) != 1:
                skipped[f"{scene}/cam{cam_idx}"] = f"shape mismatch {shapes}"
                continue

            name = (per_view["stbn"].get(cam_idx) or {}).get("image_name", "")
            psnr = {r: (per_view[r].get(cam_idx) or {}).get("psnr_srgb")
                    for r in cfg.REGIMES}
            title = (f"{scene}  cam {cam_idx}  {name}  "
                     f"($\\rho$={fraction}, seed {seed})")

            sheet = sheets_dir / f"cam_{cam_idx:04d}_contact"
            error = errors_dir / f"cam_{cam_idx:04d}_error"
            _contact_sheet(images, sheet, title=title, per_view_psnr=psnr)
            _error_sheet(images, error, title=f"absolute error — {title}")

            view_rows.append({
                "scene": scene,
                "dataset_family": cfg.scene_family(scene),
                "cam_idx": cam_idx,
                "image_name": name,
                "psnr_full": psnr.get("full"),
                "psnr_iid_white": psnr.get("iid_white"),
                "psnr_indep_blue": psnr.get("indep_blue"),
                "psnr_stbn": psnr.get("stbn"),
                "delta_stbn_minus_iid": (
                    (psnr["stbn"] - psnr["iid_white"])
                    if psnr.get("stbn") is not None
                    and psnr.get("iid_white") is not None else None),
                "delta_stbn_minus_indep": (
                    (psnr["stbn"] - psnr["indep_blue"])
                    if psnr.get("stbn") is not None
                    and psnr.get("indep_blue") is not None else None),
                "contact_sheet": sheet.relative_to(out["root"]).as_posix() + ".png",
                "error_sheet": error.relative_to(out["root"]).as_posix() + ".png",
            })
            all_patches += _scan_patches(images, scene, cam_idx, name)

    tables.write_csv(qualitative / "per_view_index.csv", view_rows, VIEW_COLUMNS)

    # The "worst" ranking is failure-case evidence, produced on the same terms as the favourable ones.
    rankings = {
        "stbn_vs_iid_best": ("delta_stbn_minus_iid", True),
        "stbn_vs_indep_best": ("delta_stbn_minus_indep", True),
        "stbn_vs_indep_worst": ("delta_stbn_minus_indep", False),
    }
    candidate_rows: list[dict] = []
    for ranking, (key, descending) in rankings.items():
        ordered = sorted((p for p in all_patches if p.get(key) is not None),
                         key=lambda p: p[key], reverse=descending)
        crops_dir = qualitative / "candidates" / ranking
        for rank, patch in enumerate(ordered[:TOP_CANDIDATES], start=1):
            row = dict(patch)
            row["ranking"] = ranking
            row["rank"] = rank
            stem = (f"{rank:02d}_{patch['scene']}_cam{patch['cam_idx']:04d}"
                    f"_x{patch['x0']}_y{patch['y0']}")
            # POSIX separators: these become report.md links and cross-platform CSV paths.
            row["crop_png"] = (
                (crops_dir / stem).relative_to(out["root"]).as_posix() + ".png")
            candidate_rows.append(row)

    # Re-open each candidate's images once per view to cut the crop cost.
    by_view: dict[tuple[str, int], list[dict]] = {}
    for row in candidate_rows:
        by_view.setdefault((row["scene"], row["cam_idx"]), []).append(row)
    for (scene, cam_idx), rows in by_view.items():
        gt_root = _gt_dir(results, scene)
        if gt_root is None:
            continue
        images = {"gt": _load_png(gt_root / f"cam_{cam_idx:02d}" / "gt.png")}
        for regime in cfg.REGIMES:
            found = results.select(scene=scene, regime=regime, seed=cfg.PRIMARY_SEED,
                                   fraction=(1.0 if regime == "full" else fraction))
            images[regime] = (_load_png(found[0].render_path(cam_idx))
                              if found else None)
        if images.get("gt") is None:
            continue
        for row in rows:
            stem = Path(row["crop_png"]).stem
            _save_crop(images, row, out["root"] / Path(row["crop_png"]).parent / stem)

    tables.write_csv(qualitative / "candidate_crops.csv", candidate_rows,
                     CANDIDATE_COLUMNS)

    (qualitative / "README.md").write_text(
        "# Qualitative material\n\n"
        f"Panel order is fixed everywhere: {' | '.join(PANEL_LABELS[p] for p in PANEL_ORDER)}.\n\n"
        f"- `<scene>/contact_sheets/` one sheet per held-out view.\n"
        f"- `<scene>/error_maps/` absolute RGB error vs GT, shared colour scale "
        f"within a view (a per-panel scale would flatter the worst regime).\n"
        f"- `candidates/stbn_vs_iid_best/` patches where STBN most improves on IID.\n"
        f"- `candidates/stbn_vs_indep_best/` where STBN most improves on "
        f"independent blue noise.\n"
        f"- `candidates/stbn_vs_indep_worst/` where STBN is most BEATEN by "
        f"independent blue noise. **These are kept deliberately** — pick honest "
        f"examples, and if the method has a characteristic failure it is here.\n"
        f"- `candidate_crops.csv` every candidate with scene, view and pixel "
        f"rectangle, so a chosen figure is traceable.\n"
        f"- `per_view_index.csv` per-view PSNR for all four regimes.\n\n"
        f"Patch scan: {PATCH_SIZE}x{PATCH_SIZE} patches, stride {PATCH_STRIDE}, "
        f"patches with GT std < {MIN_PATCH_STD} skipped (a large relative delta "
        f"on a featureless region is numerical, not qualitative).\n",
        encoding="utf-8")

    def _extremes(ranking: str) -> list[dict]:
        return [{k: r[k] for k in ("scene", "cam_idx", "x0", "y0",
                                   "delta_stbn_minus_iid",
                                   "delta_stbn_minus_indep", "crop_png")}
                for r in candidate_rows if r["ranking"] == ranking][:5]

    return {
        "n_views": len(view_rows),
        "n_scenes": len({r["scene"] for r in view_rows}),
        "n_patches_scanned": len(all_patches),
        "n_candidates": len(candidate_rows),
        "skipped": skipped,
        "top_stbn_vs_iid": _extremes("stbn_vs_iid_best"),
        "top_stbn_vs_indep": _extremes("stbn_vs_indep_best"),
        "worst_stbn_vs_indep": _extremes("stbn_vs_indep_worst"),
        "artifacts": {
            "per_view_index_csv": "qualitative/per_view_index.csv",
            "candidates_csv": "qualitative/candidate_crops.csv",
            "readme": "qualitative/README.md",
        },
    }
