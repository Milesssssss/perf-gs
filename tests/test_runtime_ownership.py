"""Enforce GpuRuntime as the sole owner of command encoders and queue waits."""

from __future__ import annotations

import ast
from pathlib import Path

from trainer.backend.compute.gpu import CommandBatch, GpuRuntime


class _TrackingLock:
    def __init__(self) -> None:
        self.depth = 0

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.depth -= 1


class _Encoder:
    def finish(self):
        return "command-buffer"


class _Device:
    def __init__(self, lock: _TrackingLock) -> None:
        self.lock = lock
        self.events: list[object] = []

    def submit_command_buffer(self, command):
        assert self.lock.depth > 0
        self.events.append(("submit", command))
        return 17

    def wait_for_submit(self, submit_id):
        assert self.lock.depth > 0
        self.events.append(("wait_submit", submit_id))

    def wait_for_idle(self):
        assert self.lock.depth > 0
        self.events.append("wait_idle")


def test_runtime_holds_lock_for_submit_callback_and_waits() -> None:
    lock = _TrackingLock()
    device = _Device(lock)
    runtime = GpuRuntime(device, lock)

    def after_submit() -> None:
        assert lock.depth > 0
        device.events.append("present")

    handle = runtime.submit(CommandBatch(_Encoder()), after_submit=after_submit)
    runtime.wait_for_submit(handle)
    runtime.wait_for_idle()

    assert device.events == [
        ("submit", "command-buffer"), "present", ("wait_submit", 17), "wait_idle"]


def test_no_raw_queue_operations_outside_runtime() -> None:
    root = Path(__file__).resolve().parent.parent / "src" / "trainer"
    banned = {
        "create_command_encoder", "submit_command_buffer",
        "wait_for_idle", "wait_for_submit",
    }
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "gpu.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in banned:
                continue
            receiver = node.func.value
            is_device = (
                isinstance(receiver, ast.Name) and receiver.id == "device"
                or isinstance(receiver, ast.Attribute) and receiver.attr in {"device", "_device"}
            )
            if is_device:
                offenders.append(
                    f"{path.relative_to(root)}:{node.lineno}: {node.func.attr}")
    assert offenders == [], "raw queue operations outside GpuRuntime:\n" + "\n".join(offenders)


def test_no_implicit_dispatch_outside_runtime() -> None:
    root = Path(__file__).resolve().parent.parent / "src" / "trainer"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "gpu.py":
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "command_encoder=None" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{path.relative_to(root)}:{line_no}: {line.strip()}")
    assert offenders == [], "implicit SlangPy submissions:\n" + "\n".join(offenders)
