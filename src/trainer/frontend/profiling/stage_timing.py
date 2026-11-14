# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Sampled per-stage GPU timing across a whole training run.

``bench_iter.py`` answers "how fast is one frozen iteration"; this answers "where
did the 90 seconds go, and how did that shift as the scene densified". It reuses
the existing :class:`~trainer.backend.compute.timing.StepTimer` and the marker
groups ``Pipeline`` already declares, so it adds no instrumentation call sites —
only a policy for when to install the timer and how to accumulate what it reads.

Cost control is the whole design. Reading timestamps requires the submit that
wrote them to have COMPLETED, which means a host wait that a normal iteration
never performs — so timing every iteration would measure a pipeline we broke.
Instead one iteration in ``every`` is sampled (default 25, i.e. 120 samples over
a 3k run) and the sampled iterations are counted, so the overhead is bounded,
measurable, and reported rather than assumed:

    overhead_ratio ~= sampled_iters / total_iters * (drain cost / iter cost)

RAW MARKER SUMS ARE WHAT GETS PERSISTED. Mapping markers onto paper-facing stage
names ("binning/sorting", "optimizer") is an editorial choice that tends to get
revised while writing, and re-running a 14-hour suite to relabel a bar chart
would be absurd. ``stage_timing.jsonl`` therefore holds every marker the pipeline
declared, per sample; :data:`STAGE_MAP` is the *default* grouping applied at
analysis time and can be changed without touching a single measurement.

Host-side costs the GPU markers cannot see (mask-slice selection, the densify
pass, which lives in its own submit outside ``train_step``) are recorded
alongside as explicit ``host_*_ms`` fields, never folded into a GPU number.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from trainer.backend.compute.timing import StepTimer
from trainer.frontend.profiling.mutation_timing import MutationTimingRecorder

# Default marker -> paper stage grouping. Keys are the ``_marker_group`` labels in
# trainer.backend.render.pipeline; a marker absent from a given run simply
# contributes nothing. Parent markers ("forward", "backward") are intentionally
# EXCLUDED from the leaf stages: they span their children, so summing both would
# double count. They are kept in the raw record and used to cross-check the
# leaves (see unattributed_ms).
STAGE_MAP: dict[str, tuple[str, ...]] = {
    # Per-splat setup: view transform, 2D covariance, SH evaluation, cull.
    "preprocess_projection": ("clear", "cluster_cull", "bin_count", "sh_forward"),
    # Building the per-tile splat lists: prefix scans, key emit, sort, ranges.
    "tile_binning_sort": ("scan_tile_offsets", "bin_emit", "radix_sort",
                          "identify_tile_ranges", "bucket_scan"),
    "forward_raster": ("rasterize",),
    # Includes the importance-guided sampling passes when --stbn-adaptive is on:
    # they are recorded INSIDE this group, so "loss" is the full image-space cost
    # of a step in every regime and stays directly comparable across them. Their
    # own share is reported separately as child markers — see CHILD_MARKERS.
    "loss": ("loss",),
    "backward_raster": ("rasterize_backward",),
    "backward_projection_sh": ("projection_backward", "sh_backward"),
    "optimizer_update": ("adam", "regularization"),
    # Densification bookkeeping that runs INSIDE the step (gradient/eta accum).
    "densify_accum": ("structure_accum", "accumulate_iter"),
}

# Markers that are containers for other markers. Reported, never summed into a
# stage, and used to compute how much of the parent the leaves account for.
PARENT_MARKERS = ("forward", "backward")

# Markers nested INSIDE a marker that a stage already counts. Reported for
# detail, excluded from stage sums — adding them would double count their
# parent. ``live_bucket_compaction`` sits inside ``rasterize_backward``;
# ``adam_main``/``adam_rest`` inside ``adam``; ``pass_N`` are the prefix-scan
# sub-passes inside ``scan_tile_offsets`` / ``bucket_scan``;
# ``adaptive_importance``/``adaptive_apply`` inside ``loss`` (the adaptive
# sampler is recorded from within the loss pass, so its cost is already in the
# loss stage — listing it here is what keeps it from being counted twice while
# still reporting what the feature itself costs).
CHILD_MARKERS = ("live_bucket_compaction", "adam_main", "adam_rest",
                 "adaptive_importance", "adaptive_apply")


def _is_child(marker: str) -> bool:
    return marker in CHILD_MARKERS or marker.startswith("pass_")

DEFAULT_SAMPLE_EVERY = 25


def _empty_accumulator() -> dict[str, Any]:
    return {"samples": 0, "markers_ms": {}, "host_ms": {}}


class StageTimingRecorder:
    """Installs the GPU step timer on a sampled cadence and accumulates results.

    Lifecycle per iteration, driven by ``TrainingLoop``:

        recorder.before_step(step)          # installs the timer if sampling
        ...pipeline.train_step(context.camera_entry.camera, lrs=context.learning_rates,
                                contextual=context.iteration_contextual_variables)...
        recorder.after_step(step, pipeline) # drains + reads if sampled
        recorder.add_host(step, "densify", ms)

    ``enabled`` is False when the feature is off, and every method is then a
    cheap no-op — the hot path pays one boolean test.
    """

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
        # Host-side per-iteration costs, accumulated for EVERY iteration (they
        # are free to measure), keyed by iteration.
        self._host_pending: dict[str, float] = {}
        self.host_totals_ms: dict[str, float] = {}
        self.host_counts: dict[str, int] = {}
        self.sampled_iters = 0
        self.sample_overhead_s = 0.0

    # -- host-side accounting (always on when enabled) --------------------

    def add_host(self, label: str, milliseconds: float) -> None:
        """Record a host-measured cost for the CURRENT iteration.

        Used for work the GPU markers structurally cannot see: the mask-slice
        selection (pure CPU) and the densify pass (its own submit, outside
        ``train_step``'s marker tree).
        """
        if not self.enabled:
            return
        self._host_pending[label] = self._host_pending.get(label, 0.0) + milliseconds
        self.host_totals_ms[label] = self.host_totals_ms.get(label, 0.0) + milliseconds
        self.host_counts[label] = self.host_counts.get(label, 0) + 1

    # -- GPU sampling ----------------------------------------------------

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
            pipeline._step_timer = self._timer
            self._pipeline = pipeline
            self._sampling = True
            self._sample_started = time.perf_counter()
        except Exception as error:                         # noqa: BLE001
            # Timing is an observation. If the query pool cannot be created on
            # this backend, the run must still complete normally.
            print(f"[stage-timing] disabled: {type(error).__name__}: {error}",
                  flush=True)
            self.enabled = False
            self._sampling = False
            self.mutations.enabled = False

    def after_step(self, step: int, pipeline: Any) -> None:
        """Drain the sampled submit and read its per-marker durations.

        The drain is the entire overhead of this feature and is measured, so the
        report can state it instead of claiming it is negligible.
        """
        pending, self._host_pending = dict(self._host_pending), {}
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
                "host_ms": pending,
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
                pipeline._step_timer = None
            except Exception:                              # noqa: BLE001
                pass
            self.sample_overhead_s += time.perf_counter() - self._sample_started

    # -- aggregation -----------------------------------------------------

    def _aggregate(self, samples: list[dict], label: str,
                   lo: int, hi: int) -> dict:
        markers: dict[str, float] = {}
        host: dict[str, float] = {}
        for sample in samples:
            for key, value in sample["markers_ms"].items():
                markers[key] = markers.get(key, 0.0) + value
            for key, value in sample["host_ms"].items():
                host[key] = host.get(key, 0.0) + value
        n = max(1, len(samples))

        mean_markers = {k: v / n for k, v in markers.items()}
        stages = {
            stage: round(sum(mean_markers.get(m, 0.0) for m in members), 6)
            for stage, members in STAGE_MAP.items()
        }
        attributed = sum(stages.values())
        # Every marker that is neither a parent, a scan sub-pass, nor mapped.
        mapped = {m for members in STAGE_MAP.values() for m in members}
        # A marker here is a real gap in STAGE_MAP: neither mapped, nor a
        # container, nor nested inside something already counted. It stays
        # visible so a new pipeline pass cannot quietly vanish from the
        # breakdown.
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
            "host_ms": {k: round(v / n, 6) for k, v in host.items()},
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
            # The honest cost of this instrumentation, for the "is the overhead
            # negligible?" decision.
            "sample_overhead_s": round(self.sample_overhead_s, 4),
            "stage_map": {k: list(v) for k, v in STAGE_MAP.items()},
            "host_totals_ms": {k: round(v, 4)
                               for k, v in sorted(self.host_totals_ms.items())},
            "host_counts": dict(sorted(self.host_counts.items())),
            "mutation_timings": self.mutations.summary(),
            "windows": windows,
            "whole_run": whole,
            "note": ("stages_ms are MEANS PER SAMPLED ITERATION in ms. Parent "
                     "markers (forward/backward) span their children and are "
                     "excluded from stage sums to avoid double counting. "
                     "host_ms are host wall-clock costs the GPU markers cannot "
                     "see (mask selection is pure CPU; densify is a separate "
                     "submit)."),
        }

    def write(self, run_dir: Path) -> None:
        if not self.samples and not self.host_totals_ms and not self.mutations.samples:
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        self.mutations.write(run_dir)
        with (run_dir / "stage_timing.jsonl").open("w") as handle:
            for sample in self.samples:
                handle.write(json.dumps(sample) + "\n")
        (run_dir / "stage_timing.json").write_text(
            json.dumps(self.summary(), indent=2))
