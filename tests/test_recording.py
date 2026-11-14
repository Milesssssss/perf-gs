# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""CommandRecorder.recording submits once on success and never leaves a stale batch."""

from __future__ import annotations

import pytest

from trainer.backend.compute.gpu import CommandRecorder, SubmitHandle


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


def test_recording_submits_and_clears_on_success() -> None:
    commands = CommandRecorder(_StubRuntime())
    committed = []
    with commands.recording("train_step") as transaction:
        assert commands.batch == "batch:train_step"
        assert transaction.batch == "batch:train_step"
        transaction.after_submit(lambda: committed.append("committed"))
    assert commands.batch is None
    assert commands.runtime.submits == 1
    assert commands.last_submit_id == 42
    assert committed == ["committed"]


def test_recording_clears_batch_and_skips_submit_on_exception() -> None:
    commands = CommandRecorder(_StubRuntime())
    committed = []
    with pytest.raises(ValueError):
        with commands.recording("train_step") as transaction:
            assert commands.batch == "batch:train_step"
            transaction.after_submit(lambda: committed.append("committed"))
            raise ValueError("boom during recording")
    assert commands.batch is None
    assert commands.runtime.submits == 0
    assert commands.last_submit_id == 0
    assert committed == []


def test_recording_skips_commit_when_submit_fails() -> None:
    commands = CommandRecorder(_StubRuntime())
    commands.runtime.fail_submit = True
    committed = []
    with pytest.raises(RuntimeError, match="submit failed"):
        with commands.recording("densify") as transaction:
            transaction.after_submit(lambda: committed.append("committed"))
    assert commands.batch is None
    assert commands.runtime.submits == 1
    assert commands.last_submit_id == 0
    assert committed == []


def test_attached_batch_is_restored() -> None:
    commands = CommandRecorder(_StubRuntime())
    with commands.attached("viewer_frame"):
        assert commands.batch == "viewer_frame"
    assert commands.batch is None
    assert commands.runtime.submits == 0
