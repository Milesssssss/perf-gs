# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Sampled per-stage GPU timing across a whole run: one iteration in ``every`` drains and reads
the step timer. Raw marker sums are persisted; :data:`STAGE_MAP` groups them only for analysis."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from trainer.backend.compute.timing import StepTimer
from trainer.frontend.profiling.mutation_timing import MutationTimingRecorder

# Default marker -> paper stage grouping; parents are excluded since they span their children.
STAGE_MAP: dict[str, tuple[str, ...]] = {
    # Per-splat setup: view transform, 2D covariance, SH evaluation, cull.
    "preprocess_projection": ("clear", "cluster_cull", "bin_count", "sh_forward"),
    # Building the per-tile splat lists: prefix scans, key emit, sort, ranges.
    "tile_binning_sort": ("depth_sort", "scan_tile_offsets", "bin_emit", "radix_sort",
                          "identify_tile_ranges", "bucket_scan"),
    "forward_raster": ("rasterize",),
    "loss": ("loss",),
    "backward_raster": ("rasterize_backward",),
    "backward_projection_sh": ("projection_backward", "sh_backward"),
    "optimizer_update": ("adam", "regularization"),
    # Densification bookkeeping that runs inside the step (gradient/eta accum).
    "densify_accum": ("structure_accum", "accumulate_iter"),
}

# Containers for other markers: reported, never summed into a stage.
PARENT_MARKERS = ("forward", "backward")

# Markers nested inside one a stage already counts; summing them would double count.
CHILD_MARKERS = ("live_bucket_compaction", "adam_main", "adam_rest",
                 "ssim_maps", "ssim_grad", "l1_l2_loss")


def _is_child(marker: str) -> bool:
    return marker in CHILD_MARKERS or marker.startswith("pass_")

DEFAULT_SAMPLE_EVERY = 25


def _empty_accumulator() -> dict[str, Any]:
    return {"samples": 0, "markers_ms": {}, "host_ms": {}}


class StageTimingRecorder:
    """Installs the GPU step timer on a sampled cadence and accumulates results.
    ``before_step``/``after_step`` bracket ``train_step``; disabled, every method is a no-op."""

    def __init__(self, *, every: int, total_iters: int,
                 windows: tuple[tuple[int, int], ...] | None = None) -> None:
        self.every = int(every)
        self.enabled = self.every > 0
        self.total_iters = int(total_iters)
        self.mutations = MutationTimingRecorder(enabled=self.enabled)
        # Paper-facing windows: early (pre-densify-heavy), middle, late.
        self.windows = windows or ((1, 1000), (1001, 2000), (2001, 3000))
        self._timer: StepTimer | None = None
        self._pipeline: Any = None
        self._sampling = False
        self._sample_started = 0.0
        self.samples: list[dict] = []
        self.sampled_iters = 0
        self.sample_overhead_s = 0.0

    def _is_sample_step(self, step: int) -> bool:
        return self.enabled and (step % self.every == 0)

    def before_step(self, step: int, pipeline: Any) -> None:
        if not self._is_sample_step(step):
            self._sampling = False
            return
        try:
            if self._timer is None:
                self._timer = StepTimer(pipeline.device)
            self._timer.begin_step()
            pipeline.commands.timer = self._timer
            self._pipeline = pipeline
            self._sampling = True
            self._sample_started = time.perf_counter()
        except Exception as error:                         # noqa: BLE001
            # Timing is an observation: the run must complete even without a query pool.
            print(f"[stage-timing] disabled: {type(error).__name__}: {error}",
                  flush=True)
            self.enabled = False
            self._sampling = False
            self.mutations.enabled = False

    def after_step(self, step: int, pipeline: Any) -> None:
        """Drain the sampled submit and read its per-marker durations.
        The drain is this feature's entire overhead, so it is measured and reported."""
        if not self._sampling or self._timer is None:
            return
        self._sampling = False
        try:
            pipeline.runtime.wait_for_idle()
            timing = self._timer.read()
            markers = {k: float(v) for k, v in timing.markers_ms.items()}
            self.samples.append({
                "iter": int(step),
                "markers_ms": markers,
                "gpu_total_ms": round(sum(
                    v for k, v in markers.items()
                    if k not in PARENT_MARKERS and not _is_child(k)), 6),
            })
            self.sampled_iters += 1
        except Exception as error:                         # noqa: BLE001
            print(f"[stage-timing] read failed at iter {step}: "
                  f"{type(error).__name__}: {error}", flush=True)
        finally:
            try:
                pipeline.commands.timer = None
            except Exception:                              # noqa: BLE001
                pass
            self.sample_overhead_s += time.perf_counter() - self._sample_started

    def _aggregate(self, samples: list[dict], label: str,
                   lo: int, hi: int) -> dict:
        markers: dict[str, float] = {}
        for sample in samples:
            for key, value in sample["markers_ms"].items():
                markers[key] = markers.get(key, 0.0) + value
        n = max(1, len(samples))

        mean_markers = {k: v / n for k, v in markers.items()}
        stages = {
            stage: round(sum(mean_markers.get(m, 0.0) for m in members), 6)
            for stage, members in STAGE_MAP.items()
        }
        attributed = sum(stages.values())
        mapped = {m for members in STAGE_MAP.values() for m in members}
        # Gaps in STAGE_MAP stay visible so a new pipeline pass cannot vanish from the breakdown.
        unmapped = {k: round(v, 6) for k, v in mean_markers.items()
                    if k not in mapped and k not in PARENT_MARKERS
                    and not _is_child(k)}
        children = {k: round(v, 6) for k, v in mean_markers.items()
                    if _is_child(k)}
        return {
            "window": label,
            "iter_from": lo,
            "iter_to": hi,
            "samples": len(samples),
            "stages_ms": stages,
            "stage_total_ms": round(attributed, 6),
            "unmapped_markers_ms": unmapped,
            "parent_markers_ms": {
                k: round(mean_markers.get(k, 0.0), 6) for k in PARENT_MARKERS
                if k in mean_markers},
            "nested_child_markers_ms": children,
            "raw_markers_ms": {k: round(v, 6)
                               for k, v in sorted(mean_markers.items())},
        }

    def summary(self) -> dict:
        """Per-window and whole-run means, plus the measured overhead."""
        windows = []
        for lo, hi in self.windows:
            picked = [s for s in self.samples if lo <= s["iter"] <= hi]
            if picked:
                windows.append(self._aggregate(picked, f"{lo}-{hi}", lo, hi))
        whole = (self._aggregate(self.samples, "all", 1, self.total_iters)
                 if self.samples else None)
        return {
            "sample_every": self.every,
            "sampled_iters": self.sampled_iters,
            "total_iters": self.total_iters,
            "sample_coverage": (self.sampled_iters / self.total_iters
                                if self.total_iters else None),
            "sample_overhead_s": round(self.sample_overhead_s, 4),
            "stage_map": {k: list(v) for k, v in STAGE_MAP.items()},
            "mutation_timings": self.mutations.summary(),
            "windows": windows,
            "whole_run": whole,
            "note": ("stages_ms are MEANS PER SAMPLED ITERATION in ms. Parent "
                     "markers (forward/backward) span their children and are "
                     "excluded from stage sums to avoid double counting. "
                     "Densify, Morton and Mip run in separate submits and are "
                     "timed per event in mutation_timings."),
        }

    def write(self, run_dir: Path) -> None:
        if not self.samples and not self.mutations.samples:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        self.mutations.write(run_dir)
        with (run_dir / "stage_timing.jsonl").open("w") as handle:
            for sample in self.samples:
                handle.write(json.dumps(sample) + "\n")
        (run_dir / "stage_timing.json").write_text(
            json.dumps(self.summary(), indent=2))
