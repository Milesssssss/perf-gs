# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""CPU unit test for ``Pipeline._recording`` exception safety.

The recording scope must clear the active command batch in a ``finally`` so a
mid-recording exception can't leave the pipeline wedged with a stale batch, and
must submit only when the body completes without raising. Exercised here with a
stub runtime so it needs no GPU.
"""

from __future__ import annotations

import pytest

from trainer.backend.compute.gpu import SubmitHandle
from trainer.backend.render.pipeline import TrainingPipeline


class _StubRuntime:
    def __init__(self) -> None:
        self.submits = 0
        self.fail_submit = False

    def command_batch(self, label):
        return f"batch:{label}"

    def submit(self, batch):
        self.submits += 1
        if self.fail_submit:
            raise RuntimeError("submit failed")
        return SubmitHandle(42)


class _StubPipeline:
    # Bind the real context manager onto a minimal stub with just the state it
    # touches (runtime / _batch / last_submit_id) — no device, no buffers.
    _recording = TrainingPipeline._recording

    def __init__(self) -> None:
        self.runtime = _StubRuntime()
        self._batch = None
        self.last_submit_id = 0


def test_recording_submits_and_clears_on_success() -> None:
    p = _StubPipeline()
    committed = []
    with p._recording("train_step") as transaction:
        assert p._batch == "batch:train_step"
        assert transaction.batch == "batch:train_step"
        transaction.after_submit(lambda: committed.append("committed"))
    assert p._batch is None            # active batch cleared
    assert p.runtime.submits == 1      # submitted exactly once
    assert p.last_submit_id == 42
    assert committed == ["committed"]


def test_recording_clears_batch_and_skips_submit_on_exception() -> None:
    p = _StubPipeline()
    committed = []
    with pytest.raises(ValueError):
        with p._recording("train_step") as transaction:
            assert p._batch == "batch:train_step"
            transaction.after_submit(lambda: committed.append("committed"))
            raise ValueError("boom during recording")
    assert p._batch is None            # cleared in finally despite the exception
    assert p.runtime.submits == 0      # never submitted a partial batch
    assert p.last_submit_id == 0       # unchanged
    assert committed == []             # host ownership was not committed


def test_recording_skips_commit_when_submit_fails() -> None:
    p = _StubPipeline()
    p.runtime.fail_submit = True
    committed = []
    with pytest.raises(RuntimeError, match="submit failed"):
        with p._recording("densify") as transaction:
            transaction.after_submit(lambda: committed.append("committed"))
    assert p._batch is None
    assert p.runtime.submits == 1
    assert p.last_submit_id == 0
    assert committed == []
