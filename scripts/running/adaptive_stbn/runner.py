# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Execute one cell in a fresh subprocess and fold its outputs into the manifest.

Every cell is its own process: a clean GPU device, clean densification state, and
no chance that seventy sequential runs accumulate leaked memory or a poisoned RNG.
It also means a cell that hard-crashes the driver costs exactly one cell.

Unattended means never dying on one bad cell. A missing scene, a non-zero exit, a
driver hang, an absent output, an unparseable JSON — each is caught, recorded with
its reason and the path to the captured stderr, and the suite moves on.
Aggregation later refuses to average incomplete groups, so a swallowed failure
cannot become a silently thinner mean.

Timing is measured from the parent, so ``process_wall_s`` is the honest total. The
trainer's own ``train_active_s`` is the training-only figure and the per-stage GPU
means are the ones the performance claims use; all are recorded, because a paper
quoting one without the others is easy to misread.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from . import config as cfg
from .manifest import (STATUS_FAILED, STATUS_MISSING_SCENE, STATUS_OK, CellRecord,
                       Manifest)

# A 3k-iteration cell at the published resolutions takes a couple of minutes; an
# hour means it hung (a driver stall, a lost fence) and the suite should reclaim
# the GPU and move on.
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
    timing = report.get("timing") or {}
    quality = report.get("quality") or {}
    render = report.get("render") or {}
    memory = report.get("memory") or {}
    device = report.get("device") or {}
    scene = report.get("scene") or {}
    model = report.get("model") or {}
    adaptive = report.get("adaptive_stbn") or {}

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

    if adaptive.get("enabled"):
        config = adaptive.get("config") or {}
        measured = adaptive.get("measured") or {}
        record.uniform_mode = str(config.get("uniform") or "")
        record.unbiased = config.get("unbiased")
        record.p_mean = measured.get("p_mean")
        record.p_min = measured.get("p_min")
        record.p_max = measured.get("p_max")
        record.saturated_fraction = measured.get("saturated_fraction_mean")
        record.selected_fraction = measured.get("selected_fraction_mean")
        record.lambda_mean = measured.get("lambda_mean")
        record.budget_error_max = measured.get("budget_error_max")
    elif record.regime != cfg.REGIME_FULL:
        # A uniform sparse arm has a constant p and selects exactly its quantized
        # density, so its columns are known analytically. Filled in rather than
        # left blank so the sampling table can be read straight down: a blank
        # would look like a measurement that failed.
        from trainer.backend.render.stbn import pixels_per_tile
        from trainer.contracts.codegen import TILE_AREA
        rho = pixels_per_tile(record.pixel_fraction, TILE_AREA) / TILE_AREA
        record.uniform_mode = "uniform"
        record.unbiased = True
        record.p_mean = record.p_min = record.p_max = rho
        record.saturated_fraction = 0.0
        record.selected_fraction = rho
        record.budget_error_max = 0.0


def _fold_stage_timing(record: CellRecord, timing: dict) -> None:
    """Whole-run per-stage GPU means, in ms per iteration.

    ``adaptive_sampling_ms`` comes from the CHILD markers: the sampler is recorded
    inside the loss group, so its cost is already inside ``loss_ms`` and adding it
    to the stage sum would double count. Reported separately so the report can say
    what the feature costs AND what the whole image-space pass costs.
    """
    whole = timing.get("whole_run") or {}
    stages = whole.get("stages_ms") or {}
    children = whole.get("nested_child_markers_ms") or {}
    record.backward_raster_ms = stages.get("backward_raster")
    record.loss_ms = stages.get("loss")
    record.forward_raster_ms = stages.get("forward_raster")
    record.stage_total_ms = whole.get("stage_total_ms")
    share = [children.get(name) for name in ("adaptive_importance",
                                             "adaptive_apply")]
    present = [v for v in share if v is not None]
    record.adaptive_sampling_ms = (round(sum(present), 6) if present
                                   else (0.0 if record.adaptive is False else None))


def run_cell(cell: cfg.Cell, manifest: Manifest, *, python: str,
             logs_dir: Path, force: bool = False,
             timeout_s: float = DEFAULT_CELL_TIMEOUT_S,
             dry_run: bool = False) -> CellRecord:
    """Train one cell (or reuse it) and return its updated manifest record."""
    from paper_suite.env import GpuMemoryPoller, utc_now

    record = manifest.record_for(cell)
    record.config_hash = cell.config_hash(python)
    record.iters = cell.iters

    if cfg.scene_dir(cell.scene) is None:
        record.status = STATUS_MISSING_SCENE
        record.error = (f"scene {cell.scene!r} not found under "
                        + ", ".join(f"datasets/{r}" for r in cfg.DATASET_ROOTS))
        print(f"[suite] SKIP {cell.key}: {record.error}", flush=True)
        return record

    record.argv = cell.command(python)

    def finalize_from_disk() -> bool:
        report = _read_json(cell.run_dir / "run_report.json")
        if report is None:
            return False
        _fold_run_report(record, report)
        stage = _read_json(cell.run_dir / "stage_timing.json")
        if stage is not None:
            _fold_stage_timing(record, stage)
        return record.psnr_srgb is not None

    if not force:
        reuse, reason = manifest.reuse_decision(cell, python)
        if reuse:
            # Re-fold even when cached: the manifest row is derived data, and
            # re-deriving it keeps a hand-edited or older row honest.
            if finalize_from_disk():
                record.status = STATUS_OK
                print(f"[suite] cached {cell.key}  psnr={record.psnr_srgb:.3f} dB"
                      f"  bwd={_ms(record.backward_raster_ms)}", flush=True)
                return record
            reason = "run_report.json unreadable or without a PSNR"
        print(f"[suite] rerun  {cell.key}: {reason}", flush=True)

    if dry_run:
        print(f"[suite] DRY   {cell.key}\n        " + " ".join(record.argv),
              flush=True)
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
        print(f"[suite] FAILED {cell.key}: {record.error} "
              f"(stderr: {record.stderr_path})"
              + (f"\n         last stderr: {tail}" if tail else ""), flush=True)
        return record
    if not finalize_from_disk():
        record.status = STATUS_FAILED
        record.error = "exited 0 but run_report.json is missing or has no PSNR"
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
          f"bwd={_ms(record.backward_raster_ms)}  "
          f"loss={_ms(record.loss_ms)}  "
          f"adapt={_ms(record.adaptive_sampling_ms)}  "
          f"active={record.train_active_s:.1f}s  n_g={record.n_gaussians}"
          + (f"  p=[{record.p_min:.3f},{record.p_mean:.4f},{record.p_max:.3f}]"
             if record.p_mean is not None else ""), flush=True)
    return record


def _ms(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}ms"


def _tail(path: Path, lines: int = 3, width: int = 400) -> str:
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
             dry_run: bool = False, save_every: int = 1) -> dict:
    """Run every cell, persisting the manifest as it goes.

    Saved after each cell so an interrupted suite resumes from the last completed
    one rather than from the beginning. The ETA comes from the cells already
    measured, which is the only honest source for one (scene sizes vary ~2.5x).
    """
    total = len(cells)
    durations: list[float] = []
    counts = {"ok": 0, "cached": 0, "failed": 0, "skipped": 0}
    started = time.perf_counter()

    for index, cell in enumerate(cells, start=1):
        before = manifest.records.get(cell.key)
        was_ok = before is not None and before.status == STATUS_OK
        elapsed = time.perf_counter() - started
        eta = ""
        if durations:
            mean = sum(durations) / len(durations)
            eta = (f"  eta {mean * (total - index + 1) / 60:.0f} min"
                   f"  (mean {mean / 60:.1f} min/cell)")
        print(f"\n[suite] === cell {index}/{total} "
              f"({elapsed / 60:.1f} min elapsed){eta} ===", flush=True)

        record = run_cell(cell, manifest, python=python, logs_dir=logs_dir,
                          force=force, timeout_s=timeout_s, dry_run=dry_run)
        if record.status == STATUS_OK:
            if was_ok and record.process_wall_s is None:
                counts["cached"] += 1
            else:
                counts["ok"] += 1
                if record.process_wall_s:
                    durations.append(record.process_wall_s)
        elif record.status == STATUS_MISSING_SCENE:
            counts["skipped"] += 1
        else:
            counts["failed"] += 1

        if not dry_run and (index % max(1, save_every) == 0 or index == total):
            manifest.save()

    if not dry_run:
        manifest.save()
    counts["wall_hours"] = (time.perf_counter() - started) / 3600.0
    return counts
