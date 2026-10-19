# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Execute one cell in a fresh subprocess and fold its outputs into the manifest.

Every cell is its own process, as the original ablation script did, and for the
same reasons: a clean GPU device, clean densification state, and no chance that
250 sequential runs accumulate leaked memory or a poisoned RNG. It also means a
cell that hard-crashes the driver costs exactly one cell.

Unattended means never dying on one bad cell. Anything a cell can do wrong — a
missing scene, a non-zero exit, a driver hang, an output file that never
appeared, a JSON file that will not parse — is caught, recorded with its reason
and the path to the captured stderr, and the suite moves on. Aggregation later
refuses to average incomplete groups, so a swallowed failure cannot become a
silently-thinner mean.

Timing is measured from the parent, so ``process_wall_s`` is the honest total:
scene load, shader compilation, training, the post-loop metrics pass, and
interpreter teardown. The trainer's own ``train_active_s`` is the training-only
figure, and both are recorded — a paper that quotes one without the other is
easy to misread.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from . import config as cfg
from .env import GpuMemoryPoller, utc_now
from .manifest import (STATUS_FAILED, STATUS_MISSING_SCENE, STATUS_OK, CellRecord,
                       Manifest)

# A 3k-iteration cell at quarter resolution takes minutes; an hour means it hung
# (a driver stall, a lost fence) and the suite should reclaim the GPU and move on.
DEFAULT_CELL_TIMEOUT_S = 3600.0


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _stamp_environment(record: CellRecord, environment: dict) -> None:
    git = environment.get("git") or {}
    gpu = environment.get("gpu") or {}
    stbn = environment.get("stbn_asset") or {}
    record.git_commit = git.get("commit")
    record.git_dirty = git.get("dirty")
    record.git_dirty_source_files = list(git.get("dirty_source_files") or [])
    record.stbn_asset_sha256 = stbn.get("sha256")
    record.gpu_name = gpu.get("name")
    record.gpu_memory_total = gpu.get("memory_total")
    record.gpu_driver_version = gpu.get("driver_version")


def _fold_run_report(record: CellRecord, report: dict) -> None:
    """Copy the trainer's report into the manifest row.

    Deliberately a flat projection of a handful of fields: the manifest is the
    index every table is built from, and it should stay readable by eye. The full
    nested report (per-view metrics, per-marker timings, the device feature list)
    stays in the run directory and is loaded on demand.
    """
    timing = report.get("timing") or {}
    quality = report.get("quality") or {}
    render = report.get("render") or {}
    memory = report.get("memory") or {}
    device = report.get("device") or {}
    scene = report.get("scene") or {}
    model = report.get("model") or {}

    record.train_wall_s = timing.get("train_wall_s")
    record.train_active_s = timing.get("train_active_s")
    record.observation_s = timing.get("observation_s")
    record.psnr_srgb = quality.get("psnr_srgb")
    record.ssim_srgb = quality.get("ssim_srgb")
    record.lpips = quality.get("lpips")
    record.n_gaussians = model.get("n_gaussians")
    record.render_fps = render.get("fps")
    record.gpu_alloc_capacity_mb = memory.get("total_mb")
    record.adapter_name = device.get("adapter_name")
    record.api_name = device.get("api_name")
    record.width = scene.get("width")
    record.height = scene.get("height")
    record.n_cams = scene.get("n_cams")
    record.n_train = scene.get("n_train")
    record.n_test = scene.get("n_test")
    record.holdout_stride = scene.get("holdout_stride")
    record.split_rule = scene.get("split_rule") or ""


def run_cell(cell: cfg.Cell, manifest: Manifest, *, python: str,
             logs_dir: Path, force: bool = False,
             timeout_s: float = DEFAULT_CELL_TIMEOUT_S,
             dry_run: bool = False) -> CellRecord:
    """Train one cell (or reuse it) and return its updated manifest record."""
    record = manifest.record_for(cell)
    record.config_hash = cell.config_hash(python)

    if cfg.scene_dir(cell.scene) is None:
        record.status = STATUS_MISSING_SCENE
        record.error = (f"scene {cell.scene!r} not found under "
                        + ", ".join(f"datasets/{r}" for r in cfg.DATASET_ROOTS))
        print(f"[suite] SKIP {cell.key}: {record.error}", flush=True)
        return record

    record.argv = cell.command(python)

    if not force:
        reuse, reason = manifest.reuse_decision(cell, python)
        if reuse:
            # Re-fold the outputs even when cached: the manifest row is derived
            # data, and re-deriving it keeps a hand-edited or older row honest.
            report = _read_json(cell.run_dir / "run_report.json")
            if report is not None:
                _fold_run_report(record, report)
                record.status = STATUS_OK
                print(f"[suite] cached {cell.key}  "
                      f"psnr={record.psnr_srgb:.3f} dB  "
                      f"active={record.train_active_s:.1f}s", flush=True)
                return record
            reason = "run_report.json unreadable"
        print(f"[suite] rerun  {cell.key}: {reason}", flush=True)

    if dry_run:
        print(f"[suite] DRY   {cell.key}\n        "
              + " ".join(record.argv), flush=True)
        return record

    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{cell.key}.stdout.log"
    stderr_path = logs_dir / f"{cell.key}.stderr.log"
    record.stdout_path = str(stdout_path.relative_to(cfg.REPO))
    record.stderr_path = str(stderr_path.relative_to(cfg.REPO))
    record.attempts += 1
    record.started_at = utc_now()
    record.error = ""
    _stamp_environment(record, manifest.environment)

    poller = GpuMemoryPoller()
    poller.start()
    started = time.perf_counter()
    print(f"[suite] TRAIN {cell.key}", flush=True)

    try:
        with stdout_path.open("w", encoding="utf-8", errors="replace") as out, \
                stderr_path.open("w", encoding="utf-8", errors="replace") as err:
            proc = subprocess.run(record.argv, cwd=str(cfg.REPO), stdout=out,
                                  stderr=err, timeout=timeout_s)
        record.return_code = proc.returncode
    except subprocess.TimeoutExpired:
        record.return_code = None
        record.status = STATUS_FAILED
        record.error = f"timed out after {timeout_s:.0f}s"
    except (OSError, subprocess.SubprocessError) as error:
        record.return_code = None
        record.status = STATUS_FAILED
        record.error = f"{type(error).__name__}: {error}"
    finally:
        record.process_wall_s = time.perf_counter() - started
        memory = poller.stop()
        record.finished_at = utc_now()
        if memory.get("available"):
            record.gpu_device_peak_mb = memory.get("peak_mb")
            record.gpu_device_baseline_mb = memory.get("baseline_mb")

    if record.error:
        print(f"[suite] FAILED {cell.key}: {record.error}  "
              f"(stderr: {record.stderr_path})", flush=True)
        return record

    if record.return_code != 0:
        record.status = STATUS_FAILED
        record.error = f"exit code {record.return_code}"
        tail = _tail(stderr_path)
        print(f"[suite] FAILED {cell.key}: {record.error}  "
              f"(stderr: {record.stderr_path})"
              + (f"\n         last stderr: {tail}" if tail else ""), flush=True)
        return record

    report = _read_json(cell.run_dir / "run_report.json")
    if report is None:
        record.status = STATUS_FAILED
        record.error = ("exited 0 but run_report.json is missing or unreadable")
        print(f"[suite] FAILED {cell.key}: {record.error}", flush=True)
        return record

    _fold_run_report(record, report)
    if record.psnr_srgb is None:
        record.status = STATUS_FAILED
        record.error = "run_report.json has no held-out PSNR"
        print(f"[suite] FAILED {cell.key}: {record.error}", flush=True)
        return record

    if not manifest.outputs_present(cell):
        record.status = STATUS_FAILED
        record.error = "expected artifacts missing after a clean exit"
        print(f"[suite] FAILED {cell.key}: {record.error}", flush=True)
        return record

    record.status = STATUS_OK
    print(f"[suite] done   {cell.key}  psnr={record.psnr_srgb:.3f} dB  "
          f"ssim={(record.ssim_srgb or 0):.4f}  "
          f"lpips={record.lpips if record.lpips is None else round(record.lpips, 4)}  "
          f"active={record.train_active_s:.1f}s  "
          f"process={record.process_wall_s:.1f}s  n_g={record.n_gaussians}",
          flush=True)
    return record


def _tail(path: Path, lines: int = 3, width: int = 400) -> str:
    """Last few stderr lines, for a failure message that is actionable without
    opening the log."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    if not text:
        return ""
    return " | ".join(text.splitlines()[-lines:])[:width]


def run_plan(cells: list[cfg.Cell], manifest: Manifest, *, python: str,
             logs_dir: Path, force: bool = False,
             timeout_s: float = DEFAULT_CELL_TIMEOUT_S,
             dry_run: bool = False,
             save_every: int = 1) -> dict:
    """Run every cell, persisting the manifest as it goes.

    The manifest is saved after each cell so an interrupted suite resumes from
    the last completed one rather than from the beginning. Progress lines carry a
    running ETA from the cells already measured, which is the only honest source
    for one (scene sizes vary by 2.5x).
    """
    total = len(cells)
    completed_durations: list[float] = []
    counts = {"ok": 0, "cached": 0, "failed": 0, "skipped": 0}
    suite_started = time.perf_counter()

    for index, cell in enumerate(cells, start=1):
        before = manifest.records.get(cell.key)
        was_ok = before is not None and before.status == STATUS_OK

        elapsed = time.perf_counter() - suite_started
        eta = ""
        if completed_durations:
            mean = sum(completed_durations) / len(completed_durations)
            remaining = mean * (total - index + 1)
            eta = (f"  eta {remaining / 3600:.1f}h"
                   f"  (mean {mean / 60:.1f} min/cell)")
        print(f"\n[suite] === cell {index}/{total} "
              f"({elapsed / 3600:.2f}h elapsed){eta} ===", flush=True)

        record = run_cell(cell, manifest, python=python, logs_dir=logs_dir,
                          force=force, timeout_s=timeout_s, dry_run=dry_run)

        if record.status == STATUS_OK:
            if was_ok and record.process_wall_s is None:
                counts["cached"] += 1
            else:
                counts["ok"] += 1
                if record.process_wall_s:
                    completed_durations.append(record.process_wall_s)
        elif record.status == STATUS_MISSING_SCENE:
            counts["skipped"] += 1
        else:
            counts["failed"] += 1

        if not dry_run and (index % max(1, save_every) == 0 or index == total):
            manifest.save()

    if not dry_run:
        manifest.save()
    counts["wall_hours"] = (time.perf_counter() - suite_started) / 3600.0
    return counts
