# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The manifest: what ran, under what code and settings, and whether it may be
reused.

Same contract as the paper suite's manifest — resume, provenance, integrity — with
the adaptive parameters promoted to first-class columns and folded into the reuse
decision.

The reuse rule is the one the design insists on: a cached directory is served only
when a recorded success, a matching CONFIG HASH, and the artifacts on disk all
agree. The hash covers the full argv, so ``--stbn-importance-alpha 0.5`` can never
be served from a directory trained at 0.75 — and because the cell KEY also carries
the parameters, the two cannot even share a directory to be confused about.

``status`` is explicit rather than inferred from "does the output exist", because a
cell that crashed halfway can leave a readable ``test_psnr.json`` from an earlier
attempt; only the manifest knows which run produced the current outputs.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import config as cfg

MANIFEST_SCHEMA = 1

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_MISSING_SCENE = "missing_scene"
STATUS_PENDING = "pending"


@dataclass
class CellRecord:
    """One cell's bookkeeping row.

    A flat projection on purpose: this is the index every table is built from and
    it has to stay readable by eye. The full nested report (per-view metrics,
    every marker's timing, the device feature list) stays in the run directory.
    """

    key: str
    scene: str
    dataset_family: str
    dataset_family_group: str
    regime: str
    seed: int
    pixel_fraction: float
    # Adaptive settings. Present (as the trainer's defaults) even on a uniform
    # cell so a column is never missing; ``adaptive`` says whether they applied.
    adaptive: bool = False
    alpha: float | None = None
    sigma: float | None = None
    importance_downsample: int | None = None
    uniform_mode: str = ""
    unbiased: bool | None = None
    iters: int = 0
    consumers: list[str] = field(default_factory=list)
    status: str = STATUS_PENDING
    config_hash: str = ""
    argv: list[str] = field(default_factory=list)
    run_dir: str = ""
    # Timing. `active` excludes every mid-loop observation; `process_wall` is
    # measured by the PARENT and so includes scene load, shader compile, the
    # post-loop metrics pass and interpreter teardown.
    train_wall_s: float | None = None
    train_active_s: float | None = None
    observation_s: float | None = None
    process_wall_s: float | None = None
    # Per-stage GPU means, in ms per iteration (whole run). backward_raster is
    # the PRIMARY performance metric; loss is the image-space cost the adaptive
    # arm inflates by running dense; adaptive_sampling_ms is the feature's own
    # share, already contained in loss_ms (see stage_timing.CHILD_MARKERS).
    backward_raster_ms: float | None = None
    loss_ms: float | None = None
    adaptive_sampling_ms: float | None = None
    forward_raster_ms: float | None = None
    stage_total_ms: float | None = None
    # Results.
    psnr_srgb: float | None = None
    ssim_srgb: float | None = None
    lpips: float | None = None
    n_gaussians: int | None = None
    render_fps: float | None = None
    # Realized sampling, measured on the device (never the solver's estimate).
    p_mean: float | None = None
    p_min: float | None = None
    p_max: float | None = None
    saturated_fraction: float | None = None
    selected_fraction: float | None = None
    lambda_mean: float | None = None
    budget_error_max: float | None = None
    # Scene / protocol facts.
    width: int | None = None
    height: int | None = None
    n_cams: int | None = None
    n_train: int | None = None
    n_test: int | None = None
    holdout_stride: int | None = None
    split_rule: str = ""
    # Memory.
    gpu_alloc_capacity_mb: float | None = None
    gpu_device_peak_mb: float | None = None
    gpu_device_baseline_mb: float | None = None
    # Provenance.
    git_commit: str | None = None
    git_dirty: bool | None = None
    git_dirty_source_files: list[str] = field(default_factory=list)
    stbn_asset_sha256: str | None = None
    gpu_name: str | None = None
    gpu_memory_total: str | None = None
    gpu_driver_version: str | None = None
    adapter_name: str | None = None
    api_name: str | None = None
    # Lifecycle.
    started_at: str = ""
    finished_at: str = ""
    attempts: int = 0
    return_code: int | None = None
    stdout_path: str = ""
    stderr_path: str = ""
    error: str = ""

    @property
    def usable(self) -> bool:
        """Whether this record may contribute to an aggregate.

        Requires completion AND the headline metric: a run that finished but
        produced no PSNR is not a data point.
        """
        return self.status == STATUS_OK and self.psnr_srgb is not None


class Manifest:
    """Load/save ``manifest.json`` and answer the resume question."""

    def __init__(self, path: Path, environment: dict | None = None) -> None:
        self.path = path
        self.environment: dict = environment or {}
        self.records: dict[str, CellRecord] = {}
        self.suite: dict[str, Any] = {}
        self.stages: dict[str, Any] = {}
        self._history: list[dict] = []

    @classmethod
    def load(cls, path: Path, environment: dict | None = None) -> "Manifest":
        manifest = cls(path, environment)
        if not path.is_file():
            return manifest
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            print(f"[suite] manifest unreadable ({error}); starting a new one. "
                  f"No training is discarded — cells are re-validated by config "
                  f"hash.", flush=True)
            return manifest
        if int(data.get("schema", 0)) != MANIFEST_SCHEMA:
            print(f"[suite] manifest schema {data.get('schema')} != "
                  f"{MANIFEST_SCHEMA}; rebuilding it. Cells are re-validated by "
                  f"config hash, so nothing is retrained unnecessarily.",
                  flush=True)
            return manifest
        known = set(CellRecord.__dataclass_fields__)
        for key, row in (data.get("cells") or {}).items():
            manifest.records[key] = CellRecord(
                **{k: v for k, v in row.items() if k in known})
        manifest.suite = data.get("suite") or {}
        manifest.stages = data.get("stages") or {}
        history = list(data.get("environment_history") or [])
        if data.get("environment"):
            history.append(data["environment"])
        manifest._history = history[-20:]
        return manifest

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": MANIFEST_SCHEMA,
            "suite_schema": cfg.SUITE_SCHEMA,
            "suite": self.suite,
            # Per-stage completion records (sampling / gradients / figures), so
            # `report` can say what it is missing instead of silently omitting it.
            "stages": self.stages,
            "environment": self.environment,
            "environment_history": self._history,
            "cells": {k: asdict(v) for k, v in sorted(self.records.items())},
        }
        # Write-then-rename: a suite killed mid-write must not lose the manifest
        # that records hours of completed training.
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2))
        temp.replace(self.path)

    # -- resume ----------------------------------------------------------

    def outputs_present(self, cell: cfg.Cell) -> bool:
        run_dir = cell.run_dir
        required = [run_dir / "run_report.json", run_dir / "test_psnr.json",
                    run_dir / "stage_timing.json", run_dir / "test_renders"]
        return all(p.exists() for p in required)

    def reuse_decision(self, cell: cfg.Cell, python: str) -> tuple[bool, str]:
        """``(reuse, reason)`` for one cell.

        Anything short of "recorded ok, hash matches, artifacts on disk" re-runs,
        with the reason logged. A cache that silently serves a mismatched cell is
        worse than no cache — and for this study specifically, serving an
        alpha=0.75 directory for an alpha=0.5 request would fabricate a
        hyperparameter result.
        """
        expected = cell.config_hash(python)
        record = self.records.get(cell.key)
        present = self.outputs_present(cell)

        if record is None:
            if present:
                return False, "outputs present but not recorded in the manifest"
            return False, "not yet run"
        if record.status == STATUS_MISSING_SCENE:
            return False, "scene was missing previously; re-checking"
        if record.status != STATUS_OK:
            return False, f"previous attempt {record.status}"
        if record.config_hash != expected:
            return False, (f"config hash changed "
                           f"({record.config_hash or 'none'} -> {expected})")
        if not present:
            return False, "recorded ok but artifacts are missing on disk"
        return True, "cached"

    # -- integrity -------------------------------------------------------

    def usable_records(self) -> dict[str, CellRecord]:
        return {k: r for k, r in self.records.items() if r.usable}

    def integrity_report(self, cells: list[cfg.Cell]) -> dict:
        planned = {c.key: c for c in cells}
        ok, failed, missing, no_scene = [], [], [], []
        for key in planned:
            record = self.records.get(key)
            if record is None:
                missing.append(key)
            elif record.usable:
                ok.append(key)
            elif record.status == STATUS_MISSING_SCENE:
                no_scene.append(key)
            else:
                failed.append(key)
        return {
            "planned": len(planned),
            "ok": len(ok), "failed": len(failed), "missing": len(missing),
            "missing_scene": len(no_scene),
            "failed_keys": sorted(failed), "missing_keys": sorted(missing),
            "missing_scene_keys": sorted(no_scene),
            "complete": not (failed or missing or no_scene),
        }

    def record_for(self, cell: cfg.Cell) -> CellRecord:
        existing = self.records.get(cell.key)
        if existing is not None:
            existing.consumers = list(cell.consumers)
            return existing
        record = CellRecord(
            key=cell.key, scene=cell.scene, dataset_family=cell.family,
            dataset_family_group=cell.family_group, regime=cell.regime,
            seed=cell.seed, pixel_fraction=cell.fraction,
            adaptive=cell.is_adaptive,
            alpha=(cell.alpha if cell.is_adaptive else None),
            sigma=(cell.sigma if cell.is_adaptive else None),
            importance_downsample=(cell.importance_downsample
                                   if cell.is_adaptive else None),
            iters=cell.iters, consumers=list(cell.consumers),
            run_dir=str(cell.run_dir),
        )
        self.records[cell.key] = record
        return record
