# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Opt-in timing of mutation submits outside the training-step marker tree."""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

from trainer.backend.compute.timing import StepTimer


_ROOT_MARKERS = ("densify_structure", "densify_simple", "densify_tracking",
                 "morton_sort", "mip_filter")


class MutationTimingRecorder:
    """Measure every expensive mutation event when stage timing is enabled.

    These events are rare and need their own samples: the step recorder removes
    its timer BEFORE densification, and submission time alone does not measure
    asynchronous Morton/Mip work. Normal, uninstrumented runs never wait here.
    """

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.samples: list[dict] = []
        self._timer: StepTimer | None = None

    @contextlib.contextmanager
    def measure(self, step: int, pipeline: Any, kind: str, *, active: bool = True):
        if not self.enabled or not active:
            yield
            return
        previous_timer = pipeline._step_timer
        try:
            if self._timer is None:
                self._timer = StepTimer(pipeline.device, pool_size=256)
            self._timer.begin_step()
            pipeline._step_timer = self._timer
        except Exception as error:                         # noqa: BLE001
            self.enabled = False
            pipeline._step_timer = previous_timer
            print(f"[mutation-timing] disabled: {error}", flush=True)
            yield
            return

        started = time.perf_counter()
        try:
            yield
        finally:
            try:
                recorded_ms = (time.perf_counter() - started) * 1000
                pipeline.runtime.wait_for_idle()
                completed_ms = (time.perf_counter() - started) * 1000
                markers = dict(self._timer.read().markers_ms)
                self.samples.append({
                    "iter": int(step), "kind": kind,
                    "markers_ms": markers,
                    "gpu_root_ms": sum(markers.get(k, 0.0) for k in _ROOT_MARKERS),
                    "host_record_ms": recorded_ms,
                    "host_complete_ms": completed_ms,
                    "instrumentation_wait_ms": completed_ms - recorded_ms,
                })
            except Exception as error:                     # noqa: BLE001
                print(f"[mutation-timing] read failed at iter {step}: {error}", flush=True)
            finally:
                pipeline._step_timer = previous_timer

    def summary(self) -> dict:
        kinds = {}
        for kind in sorted({sample["kind"] for sample in self.samples}):
            picked = [sample for sample in self.samples if sample["kind"] == kind]
            markers = {name for sample in picked for name in sample["markers_ms"]}
            kinds[kind] = {
                "events": len(picked),
                "mean_markers_ms": {
                    name: sum(s["markers_ms"].get(name, 0.0) for s in picked) / len(picked)
                    for name in sorted(markers)},
                "mean_host_complete_ms": sum(s["host_complete_ms"] for s in picked) / len(picked),
            }
        return {
            "events": len(self.samples), "by_kind": kinds,
            "instrumentation_wait_ms": sum(s["instrumentation_wait_ms"] for s in self.samples),
            "note": "GPU markers measure execution; host completion also includes queue backlog/readbacks. "
                    "Nested markers are not additive. Instrumentation waits occur only when enabled.",
        }

    def write(self, run_dir: Path) -> None:
        if not self.samples:
            return
        with (run_dir / "mutation_timing.jsonl").open("w") as handle:
            for sample in self.samples:
                handle.write(json.dumps(sample) + "\n")