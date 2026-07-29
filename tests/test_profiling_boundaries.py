"""Profiling frame boundaries must align with completed GPU iterations."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from _v2_profiling import Profiler
from trainer.backend.compute.gpu import SubmitHandle


class _FakeEncoder:
    def __init__(self, events) -> None:
        self._events = events

    def blit(self, destination, source) -> None:
        self._events.append(("blit", destination, source))


class _FakeBatch:
    def __init__(self, label, events) -> None:
        self.label = label
        self.encoder = _FakeEncoder(events)


class _FakeRuntime:
    def __init__(self, events) -> None:
        self.events = events
        self.idle_waits = 0

    def command_batch(self, label):
        self.events.append(("command_batch", label))
        return _FakeBatch(label, self.events)

    def submit(self, batch, after_submit=None):
        self.events.append(("submit", batch.label))
        if after_submit is not None:
            after_submit()
        return SubmitHandle(99)

    def wait_for_submit(self, handle):
        submit_id = handle.submit_id if isinstance(handle, SubmitHandle) else handle
        self.events.append(("wait_for_submit", submit_id))

    def wait_for_idle(self):
        self.idle_waits += 1
        self.events.append("wait_for_idle")


class _FakeSurface:
    def __init__(self, events) -> None:
        self.events = events

    def acquire_next_image(self):
        self.events.append("acquire")
        return "surface_texture"

    def present(self):
        self.events.append("present")


class _FakeWindow:
    width = 64
    height = 32

    def __init__(self, events) -> None:
        self.events = events

    def process_events(self):
        self.events.append("process_events")

    def should_close(self):
        return False

    def close(self):
        self.events.append("close")


class _FakeCounter:
    def to_numpy(self):
        return np.array([7], dtype=np.uint32)


class _FakePipeline:
    def __init__(self, events) -> None:
        self.runtime = _FakeRuntime(events)
        self.buffers = SimpleNamespace(
            output_image="output_image",
            num_gaussians_counter=_FakeCounter(),
        )
        self.last_submit_id = 0
        self.train_calls = 0

    def train_step(self, camera, lrs):
        self.train_calls += 1
        self.last_submit_id = self.train_calls


class _FakeQueueLimiter:
    def __init__(self) -> None:
        self.cleared = 0

    def clear(self):
        self.cleared += 1


def test_present_waits_for_iteration_before_cpu_frame_boundary() -> None:
    events = []
    profiler = Profiler(SimpleNamespace())
    profiler.pipeline = _FakePipeline(events)
    profiler.window = _FakeWindow(events)
    profiler.surface = _FakeSurface(events)

    profiler.present_frame(42)

    assert events == [
        "process_events",
        ("wait_for_submit", 42),
        "acquire",
        ("command_batch", "profile_present"),
        ("blit", "surface_texture", "output_image"),
        ("submit", "profile_present"),
        "present",
        ("wait_for_submit", 99),
    ]


def test_capture_loop_presents_each_completed_frozen_submit(monkeypatch) -> None:
    events = []
    args = SimpleNamespace(
        capture_external=True,
        capture_iter=5,
        capture_frames=2,
    )
    profiler = Profiler(args)
    profiler.window = _FakeWindow(events)
    pipeline = _FakePipeline(events)
    queue_limiter = _FakeQueueLimiter()
    presented = []
    monkeypatch.setattr(profiler, "present_frame", presented.append)

    captured = profiler.run_capture_if_due(
        pipeline=pipeline,
        it=5,
        entry=SimpleNamespace(camera="camera"),
        cam_idx=3,
        queue_limiter=queue_limiter,
    )

    assert captured is True
    assert pipeline.train_calls == 10  # 8 warm-up + 2 captured frozen frames
    assert presented == list(range(1, 11))
    assert pipeline.runtime.idle_waits == 3  # initial drain, post-warmup, teardown
    assert queue_limiter.cleared == 1
