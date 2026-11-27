# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU runtime (queue + submission) and command recording: each recorded batch
becomes one command buffer submitted under the shared queue lock."""

from __future__ import annotations

import contextlib
import hashlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
import slangpy as spy


@lru_cache(maxsize=None)
def marker_color(name: str) -> tuple[float, float, float]:
    """Run-stable RGB for a debug-marker name (blake2s: builtin hash() is salted per process)."""
    digest = hashlib.blake2s(name.encode("utf-8"), digest_size=3).digest()
    # Bias into [0.35, 1.0] so colors stay legible (never near-black).
    return tuple(0.35 + (b / 255.0) * 0.65 for b in digest)  # type: ignore[return-value]


_METAL_INDIRECT_NOTE = False


def indirect_group_counts(words: np.ndarray, offset: int) -> tuple[int, int, int]:
    """Three dispatch group counts stored as uint32s at ``offset`` bytes."""
    if offset < 0 or offset % 4 != 0:
        raise ValueError(f"indirect args offset {offset} is not a uint32 boundary")
    flat = np.asarray(words, dtype=np.uint32).reshape(-1)
    start = offset // 4
    return int(flat[start]), int(flat[start + 1]), int(flat[start + 2])


def thread_count_for_groups(kernel, groups: tuple[int, int, int]) -> list[int]:
    """Thread counts that launch exactly ``groups`` of this kernel's reflected group size.

    Indirect args are group counts. ``ComputeKernel.dispatch`` takes threads and divides
    by ``[numthreads]``, so the product is the same launch Vulkan's indirect dispatch runs.
    """
    entry = kernel.program.layout.entry_points[0]
    size = entry.compute_thread_group_size
    return [int(groups[axis]) * int(size[axis]) for axis in range(3)]


def _note_metal_indirect_once() -> None:
    global _METAL_INDIRECT_NOTE
    if _METAL_INDIRECT_NOTE:
        return
    _METAL_INDIRECT_NOTE = True
    print(
        "Metal: slangpy 0.42.0 drops indirect compute dispatches; "
        "replaying them from the same group counts.",
        file=sys.stderr,
    )


@dataclass(frozen=True)
class SubmitHandle:
    """A submitted command buffer; wait on it to bound queue depth without a full device idle."""
    submit_id: int


class CommandBatch:
    """One command buffer's worth of recorded GPU work (wraps a ``spy.CommandEncoder``)."""

    def __init__(self, encoder: spy.CommandEncoder, label: str = "",
                 runtime: GpuRuntime | None = None) -> None:
        self._encoder = encoder
        self._label = label
        self._runtime = runtime
        self._debug_groups: list[str] = []
        # Tests set this to exercise the readback path without a Metal device.
        self.emulate_indirect = False

    @property
    def encoder(self) -> spy.CommandEncoder:
        return self._encoder

    def _emulate_indirect(self) -> bool:
        if self.emulate_indirect:
            return True
        try:
            return self._encoder.device.info.type == spy.DeviceType.metal
        except AttributeError:
            return False

    def dispatch(self, kernel, marker: str, **kwargs) -> None:
        self._encoder.insert_debug_marker(marker, spy.math.float3(*marker_color(marker)))
        kernel.dispatch(command_encoder=self._encoder, **kwargs)

    def dispatch_indirect(self, kernel, marker: str, args_buf: spy.NDBuffer,
                          offset: int = 0, **kwargs) -> None:
        """Record an indirect dispatch; groups_x/y/z are uint32s at ``offset`` in ``args_buf``.

        No barrier before the fetch: sgl's resource tracking covers the ``args_buf`` write.
        On Metal, slangpy 0.42.0's command recorder ignores that fetch, so the same group
        counts are read back and launched with an ordinary dispatch.
        """
        if self._emulate_indirect():
            self._dispatch_indirect_replayed(kernel, marker, args_buf, offset, **kwargs)
            return
        self._encoder.insert_debug_marker(marker, spy.math.float3(*marker_color(marker)))
        pass_enc = self._encoder.begin_compute_pass()
        try:
            shader_obj = pass_enc.bind_pipeline(kernel.pipeline)
            cursor = spy.ShaderCursor(shader_obj).find_entry_point(0)
            cursor.write(kwargs)
            pass_enc.dispatch_compute_indirect(spy.BufferOffsetPair(args_buf.storage, offset))
        finally:
            pass_enc.end()

    def _dispatch_indirect_replayed(self, kernel, marker: str, args_buf: spy.NDBuffer,
                                    offset: int, **kwargs) -> None:
        _note_metal_indirect_once()
        self._flush_recorded_work()
        groups = indirect_group_counts(args_buf.to_numpy(), offset)
        if groups[0] <= 0 or groups[1] <= 0 or groups[2] <= 0:
            return
        self.dispatch(kernel, marker, thread_count=thread_count_for_groups(kernel, groups), **kwargs)

    def _flush_recorded_work(self) -> None:
        """Submit the open encoder so a later read sees its writes, then keep recording."""
        device = self._encoder.device
        for _label in reversed(self._debug_groups):
            self._encoder.pop_debug_group()
        command = self._encoder.finish()
        if self._runtime is not None:
            with self._runtime.gpu_lock:
                submit_id = device.submit_command_buffer(command)
            self._runtime.wait_for_submit(submit_id)
        else:
            submit_id = device.submit_command_buffer(command)
            device.wait_for_submit(submit_id)
        self._encoder = device.create_command_encoder()
        for label in self._debug_groups:
            self._encoder.push_debug_group(label, spy.math.float3(*marker_color(label)))

    @contextlib.contextmanager
    def marker_group(self, label: str):
        self._encoder.push_debug_group(label, spy.math.float3(*marker_color(label)))
        self._debug_groups.append(label)
        try:
            yield
        finally:
            self._debug_groups.pop()
            self._encoder.pop_debug_group()

    def finish(self):
        return self._encoder.finish()


class GpuRuntime:
    """Owns the device and hands out CommandBatches; submissions serialize under ``gpu_lock``.

    The lock (shared with the viewer; a no-op by default) covers submits and ``wait_for_idle``,
    not single-submission fence waits or the viewer's ``present``.
    """

    def __init__(self, device: spy.Device,
                 gpu_lock: "contextlib.AbstractContextManager | None" = None) -> None:
        self.device = device
        self.gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()

    def command_batch(self, label: str) -> CommandBatch:
        return CommandBatch(self.device.create_command_encoder(), label, runtime=self)

    def submit(self, batch: CommandBatch,
               after_submit: Callable[[], None] | None = None) -> SubmitHandle:
        """Submit under the queue lock; ``after_submit`` (viewer present, serialized by sgl)
        runs after unlocking, so a stalled present cannot hold the queue against the trainer.
        """
        cmd = batch.finish()
        with self.gpu_lock:
            submit_id = self.device.submit_command_buffer(cmd)
        if after_submit is not None:
            after_submit()
        return SubmitHandle(submit_id)

    def wait_for_submit(self, handle: SubmitHandle | int) -> None:
        """Wait for one submission's fence; not under ``gpu_lock``, which would serialize
        the viewer and the trainer for the whole GPU wait.
        """
        submit_id = handle.submit_id if isinstance(handle, SubmitHandle) else handle
        self.device.wait_for_submit(submit_id)

    def wait_for_idle(self) -> None:
        """Queue-wide idle wait; a queue operation, so it takes the lock."""
        with self.gpu_lock:
            self.device.wait_for_idle()

    def queue_scope(self):
        """Lock scope for compound queue-adjacent operations (reload/reconfigure)."""
        return self.gpu_lock


class RecordingTransaction:
    """A recorded batch plus host-state commits that run once its submit succeeds."""

    def __init__(self, batch: CommandBatch) -> None:
        self.batch = batch
        self._commits: list[Callable[[], None]] = []

    def after_submit(self, commit: Callable[[], None]) -> None:
        self._commits.append(commit)

    def commit(self) -> None:
        for commit in self._commits:
            commit()


class CommandRecorder:
    """Records GPU work into the active batch, or submits each dispatch on its own.

    Shared by every GPU component, so a training step is one command buffer; ``timer``
    (a ``StepTimer``) is set only while profiling and timestamps every marker group.
    """

    def __init__(self, runtime: GpuRuntime) -> None:
        self.runtime = runtime
        self.batch: CommandBatch | None = None
        self.timer = None
        self.last_submit_id = 0

    def dispatch(self, kernel, marker: str, **kwargs) -> None:
        if self.batch is not None:
            self.batch.dispatch(kernel, marker, **kwargs)
            return
        transient = self.runtime.command_batch(marker)
        transient.dispatch(kernel, marker, **kwargs)
        self.last_submit_id = self.runtime.submit(transient).submit_id

    def dispatch_indirect(self, kernel, marker: str, args_buf: spy.NDBuffer,
                          offset: int = 0, **kwargs) -> None:
        if self.batch is not None:
            self.batch.dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)
            return
        transient = self.runtime.command_batch(marker)
        transient.dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)
        self.last_submit_id = self.runtime.submit(transient).submit_id

    @contextlib.contextmanager
    def marker_group(self, label: str):
        if self.batch is None:
            yield
            return
        batch, timer = self.batch, self.timer
        interval = timer.open(label, batch.encoder) if timer is not None else -1
        with batch.marker_group(label):
            try:
                yield
            finally:
                if timer is not None:
                    timer.close(interval, batch.encoder)

    @contextlib.contextmanager
    def recording(self, label: str):
        """Record into a fresh batch and submit it once, only if the body succeeds."""
        batch = self.runtime.command_batch(label)
        transaction = RecordingTransaction(batch)
        self.batch = batch
        try:
            yield transaction
            self.last_submit_id = self.runtime.submit(batch).submit_id
            transaction.commit()
        finally:
            self.batch = None

    @contextlib.contextmanager
    def attached(self, batch: CommandBatch):
        """Record into a batch the caller owns and submits (the viewer frame)."""
        previous, self.batch = self.batch, batch
        try:
            yield
        finally:
            self.batch = previous

    def upload_texture(self, texture: spy.Texture, pixels) -> None:
        if self.batch is not None:
            self.batch.encoder.upload_texture_data(texture, 0, 0, pixels)
        else:
            texture.copy_from_numpy(pixels)
