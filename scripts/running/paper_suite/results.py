# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Load run artifacts and index them the way the experiments ask questions.

Selection goes through one filter so failed or differently-instrumented cells never enter a paired comparison.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import config as cfg
from .manifest import CellRecord, Manifest


def read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file, skipping unparsable lines (a killed run can truncate the tail)."""
    rows: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


@dataclass
class CellResult:
    """A cell plus its on-disk detail, loaded lazily."""

    cell: cfg.Cell
    record: CellRecord
    _report: dict | None = field(default=None, repr=False)
    _curve: list[dict] | None = field(default=None, repr=False)
    _stage: dict | None = field(default=None, repr=False)

    @property
    def usable(self) -> bool:
        return self.record.usable

    @property
    def report(self) -> dict:
        if self._report is None:
            self._report = read_json(self.cell.run_dir / "run_report.json") or {}
        return self._report

    @property
    def per_view(self) -> list[dict]:
        return list((self.report.get("quality") or {}).get("per_view") or [])

    @property
    def curve(self) -> list[dict]:
        """Convergence checkpoints by iteration; their ``elapsed_s`` already excludes eval sweeps."""
        if self._curve is None:
            rows = read_jsonl(self.cell.run_dir / "eval_curve.jsonl")
            self._curve = sorted(rows, key=lambda r: r.get("iter", 0))
        return self._curve

    @property
    def stage_timing(self) -> dict:
        if self._stage is None:
            self._stage = read_json(
                self.cell.run_dir / "stage_timing.json") or {}
        return self._stage

    @property
    def renders_dir(self) -> Path:
        return self.cell.run_dir / "test_renders"

    def render_path(self, cam_idx: int) -> Path:
        return self.renders_dir / f"cam_{cam_idx:04d}.png"

    def metric(self, name: str) -> Any:
        """A headline metric by name, from the manifest row."""
        return getattr(self.record, name, None)


class ResultSet:
    """Indexed view over every cell in the plan."""

    def __init__(self, cells: Iterable[cfg.Cell], manifest: Manifest) -> None:
        self.results: dict[str, CellResult] = {}
        for cell in cells:
            record = manifest.records.get(cell.key)
            if record is None:
                continue
            self.results[cell.key] = CellResult(cell=cell, record=record)
        self.manifest = manifest

    def select(self, *, regime: str | None = None, seed: int | None = None,
               fraction: float | None = None, scene: str | None = None,
               convergence: bool | None = None,
               stage_timing: bool | None = None,
               select_mode: str | None = cfg.PRIMARY_SELECT_MODE,
               usable_only: bool = True) -> list[CellResult]:
        """Cells matching every supplied constraint.

        Timing and stratification-ablation cells are excluded by default so quality aggregates never double-count scenes.
        Dense cells ignore ``select_mode``: one dense run serves every mode.
        """
        out: list[CellResult] = []
        for result in self.results.values():
            cell = result.cell
            if regime is not None and cell.regime != regime:
                continue
            if (select_mode is not None and cell.regime != "full"
                    and cell.select_mode != select_mode):
                continue
            if seed is not None and cell.seed != seed:
                continue
            if scene is not None and cell.scene != scene:
                continue
            if fraction is not None and abs(cell.fraction - fraction) > 1e-9:
                continue
            if convergence is not None and cell.convergence != convergence:
                continue
            want_timing = False if stage_timing is None else stage_timing
            if cell.stage_timing != want_timing:
                continue
            if usable_only and not result.usable:
                continue
            out.append(result)
        return sorted(out, key=lambda r: (r.cell.scene, r.cell.seed))

    def metric_map(self, metric: str, *, regime: str, fraction: float,
                   seeds: Iterable[int] | None = None,
                   scenes: Iterable[str] | None = None,
                   convergence: bool | None = None,
                   select_mode: str | None = cfg.PRIMARY_SELECT_MODE,
                   ) -> dict[tuple[str, int], float]:
        """``{(scene, seed): value}`` for one regime at one fraction, usable cells with the metric only."""
        seed_filter = set(seeds) if seeds is not None else None
        scene_filter = set(scenes) if scenes is not None else None
        out: dict[tuple[str, int], float] = {}
        for result in self.select(regime=regime, fraction=fraction,
                                  convergence=convergence,
                                  select_mode=select_mode):
            cell = result.cell
            if seed_filter is not None and cell.seed not in seed_filter:
                continue
            if scene_filter is not None and cell.scene not in scene_filter:
                continue
            value = result.metric(metric)
            if value is None:
                continue
            out[(cell.scene, cell.seed)] = float(value)
        return out

    def dense_fraction(self) -> float:
        """The fraction the dense cells are recorded under (1.0 by construction), read from the plan."""
        for result in self.results.values():
            if result.cell.regime == "full":
                return result.cell.fraction
        return 1.0

    def scenes_present(self) -> list[str]:
        return sorted({r.cell.scene for r in self.results.values()})

    def rows(self, results: Iterable[CellResult]) -> list[dict]:
        """Flat per-cell dicts for CSV emission."""
        out = []
        for result in results:
            record, cell = result.record, result.cell
            out.append({
                "scene": cell.scene,
                "dataset_family": cell.family,
                "dataset_family_group": cell.family_group,
                "regime": cell.regime,
                "seed": cell.seed,
                "pixel_fraction": cell.fraction,
                "select_mode": cell.select_mode,
                "iters": cell.iters,
                "status": record.status,
                "psnr_srgb": record.psnr_srgb,
                "ssim_srgb": record.ssim_srgb,
                "lpips": record.lpips,
                "train_active_s": record.train_active_s,
                "train_wall_s": record.train_wall_s,
                "observation_s": record.observation_s,
                "process_wall_s": record.process_wall_s,
                "n_gaussians": record.n_gaussians,
                "render_fps": record.render_fps,
                "gpu_device_peak_mb": record.gpu_device_peak_mb,
                "gpu_device_baseline_mb": record.gpu_device_baseline_mb,
                "gpu_alloc_capacity_mb": record.gpu_alloc_capacity_mb,
                "width": record.width,
                "height": record.height,
                "n_cams": record.n_cams,
                "n_train": record.n_train,
                "n_test": record.n_test,
                "holdout_stride": record.holdout_stride,
                "has_convergence": record.has_convergence,
                "has_stage_timing": record.has_stage_timing,
                "git_commit": record.git_commit,
                "git_dirty": record.git_dirty,
                "stbn_asset_sha256": record.stbn_asset_sha256,
                "gpu_name": record.gpu_name,
                "gpu_driver_version": record.gpu_driver_version,
                "adapter_name": record.adapter_name,
                "api_name": record.api_name,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
                "config_hash": record.config_hash,
                "cell_key": record.key,
                "argv": " ".join(record.argv),
            })
        return out


RAW_COLUMNS = (
    "scene", "dataset_family", "dataset_family_group", "regime", "seed",
    "pixel_fraction", "select_mode", "iters", "status", "psnr_srgb", "ssim_srgb",
    "lpips",
    "train_active_s", "train_wall_s", "observation_s", "process_wall_s",
    "n_gaussians", "render_fps",
    "gpu_device_peak_mb", "gpu_device_baseline_mb",
    "gpu_alloc_capacity_mb", "width", "height", "n_cams", "n_train", "n_test",
    "holdout_stride", "has_convergence", "has_stage_timing", "git_commit",
    "git_dirty", "stbn_asset_sha256", "gpu_name", "gpu_driver_version",
    "adapter_name", "api_name", "started_at", "finished_at", "config_hash",
    "cell_key", "argv",
)

PER_VIEW_COLUMNS = (
    "scene", "dataset_family", "regime", "seed", "pixel_fraction", "cam_idx",
    "image_name", "psnr_srgb", "ssim_srgb", "lpips", "width", "height",
    "cell_key",
)


def per_view_rows(results: Iterable[CellResult]) -> list[dict]:
    """One row per (cell, held-out view) — the per-test-view store."""
    rows: list[dict] = []
    for result in results:
        cell = result.cell
        for view in result.per_view:
            rows.append({
                "scene": cell.scene,
                "dataset_family": cell.family,
                "regime": cell.regime,
                "seed": cell.seed,
                "pixel_fraction": cell.fraction,
                "cam_idx": view.get("cam_idx"),
                "image_name": view.get("image_name"),
                "psnr_srgb": view.get("psnr_srgb"),
                "ssim_srgb": view.get("ssim_srgb"),
                "lpips": view.get("lpips"),
                "width": view.get("width"),
                "height": view.get("height"),
                "cell_key": cell.key,
            })
    return rows
