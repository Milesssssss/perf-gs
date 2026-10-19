# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Mutation timing sees separate submits without changing normal scheduling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from trainer.frontend.profiling import mutation_timing
from trainer.frontend.profiling.mutation_timing import MutationTimingRecorder
from trainer.frontend.profiling.stage_timing import StageTimingRecorder


class _Timer:
    def __init__(self, device, pool_size):
        self.reset = 0

    def begin_step(self):
        self.reset += 1

    def read(self):
        return SimpleNamespace(markers_ms={
            "densify_structure": 20.0, "morton_sort": 8.0,
            "arena_copy_back": 6.0, "mip_filter": 12.0})


def _pipeline():
    waits = []
    pipeline = SimpleNamespace(device=object(), _step_timer=None,
                               runtime=SimpleNamespace(wait_for_idle=lambda: waits.append(1)))
    return pipeline, waits


@pytest.mark.parametrize("enabled,active", [(False, True), (True, False)])
def test_disabled_mutation_timer_never_waits(enabled, active, monkeypatch) -> None:
    pipeline, waits = _pipeline()
    recorder = MutationTimingRecorder(enabled=enabled)
    monkeypatch.setattr(mutation_timing, "StepTimer",
                        lambda *a, **k: pytest.fail("must not allocate timer"))
    with recorder.measure(100, pipeline, "structure", active=active):
        assert pipeline._step_timer is None
    assert waits == [] and recorder.samples == []


def test_mutation_timer_covers_completion_without_double_counting(monkeypatch) -> None:
    monkeypatch.setattr(mutation_timing, "StepTimer", _Timer)
    clock = iter([1.0, 1.1, 1.4])
    monkeypatch.setattr(mutation_timing.time, "perf_counter", lambda: next(clock))
    pipeline, waits = _pipeline()
    original = pipeline._step_timer = object()
    recorder = MutationTimingRecorder(enabled=True)
    with recorder.measure(500, pipeline, "structure"):
        assert pipeline._step_timer is recorder._timer
    assert pipeline._step_timer is original and waits == [1]
    sample = recorder.samples[0]
    assert sample["iter"] == 500 and sample["kind"] == "structure"
    assert sample["gpu_root_ms"] == 40.0  # not 46: copies are nested.
    assert sample["host_record_ms"] == pytest.approx(100.0)
    assert sample["host_complete_ms"] == pytest.approx(400.0)
    assert sample["instrumentation_wait_ms"] == pytest.approx(300.0)
    assert recorder.summary()["by_kind"]["structure"]["events"] == 1


def test_failed_mutation_restores_timer_and_propagates(monkeypatch) -> None:
    monkeypatch.setattr(mutation_timing, "StepTimer", _Timer)
    pipeline, _ = _pipeline()
    recorder = MutationTimingRecorder(enabled=True)
    with pytest.raises(ValueError, match="mutation failed"):
        with recorder.measure(100, pipeline, "structure"):
            raise ValueError("mutation failed")
    assert pipeline._step_timer is None


def test_timing_failure_does_not_break_mutations(monkeypatch) -> None:
    def unsupported(*args, **kwargs):
        raise RuntimeError("timestamps unavailable")
    monkeypatch.setattr(mutation_timing, "StepTimer", unsupported)
    pipeline, waits = _pipeline()
    recorder = MutationTimingRecorder(enabled=True)
    ran = []
    with recorder.measure(100, pipeline, "structure"):
        ran.append(1)
    assert ran == [1] and waits == [] and not recorder.enabled
    assert pipeline._step_timer is None


def test_mutation_samples_do_not_pollute_step_means(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(mutation_timing, "StepTimer", _Timer)
    pipeline, _ = _pipeline()
    recorder = StageTimingRecorder(every=25, total_iters=500)
    with recorder.mutations.measure(500, pipeline, "structure"):
        pass
    summary = recorder.summary()
    assert summary["sampled_iters"] == 0 and summary["whole_run"] is None
    assert summary["mutation_timings"]["events"] == 1
    recorder.write(tmp_path)
    assert (tmp_path / "mutation_timing.jsonl").exists()