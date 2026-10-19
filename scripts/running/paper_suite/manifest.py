# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The manifest: what ran, under what code, and whether it may be reused.

``manifest.json`` is the suite's memory. It answers three questions:

  1. **Resume.** Is this cell already done, with outputs that match the config
     hash it was recorded under? If so, skip it. If the hash differs, the cached
     run was produced by a DIFFERENT invocation and is not interchangeable — it is
     reported as stale and re-run, never quietly reused.
  2. **Provenance.** For every cell: the exact argv, the commit and whether the
     tree was dirty, the STBN asset hash, the GPU and driver, resolution, image
     counts, the split rule, timestamps, active and total wall seconds, the final
     Gaussian count, and on failure the return code plus the path to the captured
     stderr.
  3. **Integrity.** Which cells are missing or failed, so aggregation can refuse
     to average an incomplete group instead of silently reporting a mean over
     whatever happened to succeed.

That last point is the reason ``status`` is explicit rather than inferred from
"does the output file exist". A cell that crashed halfway can leave a readable
``test_psnr.json`` from an earlier attempt; only the manifest knows that the run
which produced the current outputs completed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import config as cfg

MANIFEST_SCHEMA = 2

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_MISSING_SCENE = "missing_scene"
STATUS_PENDING = "pending"


@dataclass
class CellRecord:
    """One cell's full bookkeeping row."""

    key: str
    scene: str
    dataset_family: str
    dataset_family_group: str
    regime: str
    seed: int
    pixel_fraction: float
    # "global" or "tile_rank": two different sampling processes, never pooled.
    select_mode: str = ""
    iters: int = 0
    status: str = STATUS_PENDING
    config_hash: str = ""
    argv: list[str] = field(default_factory=list)
    run_dir: str = ""
    # Timing. active excludes every mid-loop observation; process_wall is
    # measured by the PARENT and therefore includes scene load, shader compile,
    # the post-loop metrics pass and interpreter teardown.
    train_wall_s: float | None = None
    train_active_s: float | None = None
    observation_s: float | None = None
    process_wall_s: float | None = None
    # Results.
    psnr_srgb: float | None = None
    ssim_srgb: float | None = None
    lpips: float | None = None
    n_gaussians: int | None = None
    render_fps: float | None = None
    # Scene / protocol facts, per cell (resolution differs by dataset because
    # T&T and Deep Blending ship only full-res images and are resized on load).
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
    # Instrumentation profile.
    has_convergence: bool = False
    has_stage_timing: bool = False
    has_saved_renders: bool = False
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

        Requires completion AND the headline metric — a run that finished but
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

    # -- persistence -----------------------------------------------------

    @classmethod
    def load(cls, path: Path, environment: dict | None = None) -> "Manifest":
        manifest = cls(path, environment)
        if not path.is_file():
            return manifest
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            print(f"[suite] manifest unreadable ({error}); starting a new one. "
                  f"Completed cells will be re-detected from their outputs.",
                  flush=True)
            return manifest
        if int(data.get("schema", 0)) != MANIFEST_SCHEMA:
            print(f"[suite] manifest schema {data.get('schema')} != "
                  f"{MANIFEST_SCHEMA}; ignoring it and rebuilding. No training "
                  f"is discarded — cells are re-validated by config hash.",
                  flush=True)
            return manifest
        known = set(CellRecord.__dataclass_fields__)
        for key, row in (data.get("cells") or {}).items():
            manifest.records[key] = CellRecord(
                **{k: v for k, v in row.items() if k in known})
        manifest.suite = data.get("suite") or {}
        # Keep the ORIGINAL environment of a resumed run visible: if the commit
        # moved between sessions, that is exactly what a reader needs to know.
        history = data.get("environment_history") or []
        if data.get("environment"):
            history.append(data["environment"])
        manifest._history = history[-20:]
        return manifest

    _history: list[dict] = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": MANIFEST_SCHEMA,
            "suite_schema": cfg.SUITE_SCHEMA,
            "suite": self.suite,
            "environment": self.environment,
            # Every session that contributed cells, newest last.
            "environment_history": getattr(self, "_history", []),
            "cells": {k: asdict(v) for k, v in sorted(self.records.items())},
        }
        # Write-then-rename: a suite killed mid-write must not lose the manifest
        # that records 200 completed hours of training.
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2))
        temp.replace(self.path)

    # -- resume ----------------------------------------------------------

    def outputs_present(self, cell: cfg.Cell) -> bool:
        """Whether the cell's REQUIRED artifacts all exist on disk."""
        run_dir = cell.run_dir
        required = [run_dir / "run_report.json", run_dir / "test_psnr.json"]
        if cell.convergence:
            required.append(run_dir / "eval_curve.jsonl")
        if cell.stage_timing:
            required.append(run_dir / "stage_timing.json")
        if cell.save_renders:
            required.append(run_dir / "test_renders")
        return all(p.exists() for p in required)

    def reuse_decision(self, cell: cfg.Cell, python: str) -> tuple[bool, str]:
        """``(reuse, reason)`` for one cell.

        Reuse requires all three of: a recorded successful run, a matching config
        hash, and the artifacts still on disk. Anything else re-runs, with the
        reason logged — a cache that silently serves a mismatched cell is worse
        than no cache.
        """
        expected = cell.config_hash(python)
        record = self.records.get(cell.key)
        present = self.outputs_present(cell)

        if record is None:
            if present:
                # Outputs exist but this suite never recorded them: could be from
                # an older schema or a hand-run. Re-run rather than trust them.
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
        """What is complete, what failed, and what is missing.

        Consumed by the aggregation step, which refuses to report a group mean
        unless the group is complete.
        """
        planned = {c.key: c for c in cells}
        ok, failed, missing, stale = [], [], [], []
        for key, cell in planned.items():
            record = self.records.get(key)
            if record is None:
                missing.append(key)
            elif record.usable:
                ok.append(key)
            elif record.status == STATUS_MISSING_SCENE:
                stale.append(key)
            else:
                failed.append(key)
        return {
            "planned": len(planned),
            "ok": len(ok),
            "failed": len(failed),
            "missing": len(missing),
            "missing_scene": len(stale),
            "failed_keys": sorted(failed),
            "missing_keys": sorted(missing),
            "missing_scene_keys": sorted(stale),
            "complete": not (failed or missing or stale),
        }

    def record_for(self, cell: cfg.Cell) -> CellRecord:
        """The record for a cell, created (as pending) if absent."""
        existing = self.records.get(cell.key)
        if existing is not None:
            return existing
        record = CellRecord(
            key=cell.key,
            scene=cell.scene,
            dataset_family=cell.family,
            dataset_family_group=cell.family_group,
            regime=cell.regime,
            seed=cell.seed,
            pixel_fraction=cell.fraction,
            select_mode=cell.select_mode,
            iters=cell.iters,
            run_dir=str(cell.run_dir),
            has_convergence=cell.convergence,
            has_stage_timing=cell.stage_timing,
            has_saved_renders=cell.save_renders,
        )
        self.records[cell.key] = record
        return record
