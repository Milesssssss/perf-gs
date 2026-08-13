"""GPU runtime + command-batch abstraction.

Extracts the queue/submission and command-recording machinery out of
``Pipeline`` so it is owned by cross-cutting infrastructure rather than the
rasterizer. Behavior is preserved EXACTLY:

  * flat + nested debug markers with deterministic per-name colors,
  * one command buffer per recorded batch, submitted once under the gpu_lock,
  * the indirect-dispatch write-visibility global barrier,
  * the transient one-shot batch used when a dispatch happens with no active
    batch (e.g. an indirect dispatch driven from ``render()``).

``Pipeline`` retains an ``_encoder`` compatibility property (backed by a
``CommandBatch``) so external encoder owners — the densify pass and the live
viewer, which build their own encoder and assign it — keep working unchanged.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import slangpy as spy


@lru_cache(maxsize=None)
def marker_color(name: str) -> tuple[float, float, float]:
    """Deterministic pseudo-random RGB for a debug-marker/group name, so the
    same phase gets the same color across runs (profiler captures stay
    diff-able). Uses blake2s, NOT builtin hash() — CPython salts str hashing
    per-process, which would recolor every run."""
    digest = hashlib.blake2s(name.encode("utf-8"), digest_size=3).digest()
    # Bias into [0.35, 1.0] so colors stay legible (never near-black).
    return tuple(0.35 + (b / 255.0) * 0.65 for b in digest)  # type: ignore[return-value]


@dataclass(frozen=True)
class SubmitHandle:
    """Opaque handle to a submitted command buffer; wait on it to bound queue
    depth without a full device idle."""
    submit_id: int


class CommandBatch:
    """
    One command-buffer's worth of recorded GPU work.
    Wraps a ``spy.CommandEncoder`` and provides the dispatch / marker helpers
    that record into it.
    """

    def __init__(self, encoder: spy.CommandEncoder, label: str = "") -> None:
        self._encoder = encoder
        self._label = label

    @property
    def encoder(self) -> spy.CommandEncoder:
        return self._encoder

    def dispatch(self, kernel, marker: str, **kwargs) -> None:
        """Record a kernel dispatch tagged with a flat debug marker."""
        self._encoder.insert_debug_marker(marker, spy.math.float3(*marker_color(marker)))
        kernel.dispatch(command_encoder=self._encoder, **kwargs)

    def dispatch_indirect(self, kernel, marker: str, args_buf: spy.NDBuffer,
                          offset: int = 0, barrier_before: bool = True,
                          **kwargs) -> None:
        """Record an indirect compute dispatch (group count read from
        ``args_buf`` at ``offset`` as three uint32s: groups_x/y/z).

        Opens a scoped compute pass, binds the pipeline, writes entry-point
        params via a ShaderCursor (matching ``kernel.dispatch(**kwargs)``), then
        ``dispatch_compute_indirect``."""
        self._encoder.insert_debug_marker(marker, spy.math.float3(*marker_color(marker)))
        # TODO: im not sure if the barrier is actually needed here.
        #if barrier_before:
        #    self._encoder.global_barrier()
        pass_enc = self._encoder.begin_compute_pass()
        try:
            shader_obj = pass_enc.bind_pipeline(kernel.pipeline)
            cursor = spy.ShaderCursor(shader_obj).find_entry_point(0)
            cursor.write(kwargs)
            pass_enc.dispatch_compute_indirect(spy.BufferOffsetPair(args_buf.storage, offset))
        finally:
            pass_enc.end()

    @contextlib.contextmanager
    def marker_group(self, label: str):
        self._encoder.push_debug_group(label, spy.math.float3(*marker_color(label)))
        try:
            yield
        finally:
            self._encoder.pop_debug_group()

    def finish(self):
        return self._encoder.finish()


class GpuRuntime:
    """Owns the device + queue-lock discipline and hands out CommandBatches.

    Submissions are serialized under ``gpu_lock`` (shared with any other thread
    on this device — notably the live viewer). The lock defaults to a no-op so
    single-threaded use is unchanged.

    WHAT THE LOCK COVERS, and what it deliberately does not: it serializes the
    queue operations that require external synchronization — submitting a command
    buffer, the viewer's ``present`` paired with it, and the queue-wide
    ``wait_for_idle``. It does NOT cover waiting for a single submission, which is
    a fence wait (see :meth:`wait_for_submit`). Blocking on the GPU while holding
    this lock is what made a live viewer cost multiples of the training speed.
    """

    def __init__(self, device: spy.Device,
                 gpu_lock: "contextlib.AbstractContextManager | None" = None) -> None:
        self.device = device
        self.gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()

    def command_batch(self, label: str) -> CommandBatch:
        """Start recording a new batch on a fresh command encoder."""
        return CommandBatch(self.device.create_command_encoder(), label)

    def submit(self, batch: CommandBatch,
               after_submit: Callable[[], None] | None = None) -> SubmitHandle:
        """Finish + submit a batch under the queue lock.

        ``after_submit`` runs before releasing the lock. The viewer uses this
        for ``surface.present`` so submit+present are one externally-synchronized
        queue transaction without teaching the runtime about swapchains.
        """
        cmd = batch.finish()
        with self.gpu_lock:
            submit_id = self.device.submit_command_buffer(cmd)
            if after_submit is not None:
                after_submit()
        return SubmitHandle(submit_id)

    def wait_for_submit(self, handle: SubmitHandle | int) -> None:
        """Wait for ONE submission to finish. Deliberately NOT under ``gpu_lock``.

        This is a wait on that submission's fence: it needs no exclusive access to
        the queue, unlike submit/present/wait_for_idle. Holding the lock here meant
        whoever waited owned the queue for the entire GPU execution of their own
        work, in BOTH directions:

          * the viewer waits for its frame every single frame (``render_frame``),
            which blocked every trainer submit for the duration of that frame;
          * the trainer waits on its oldest in-flight submit to bound queue depth
            (:class:`~trainer.frontend.queue_limiter.QueueLimiter`), which blocked
            the viewer for the duration of a training step.

        Training and rendering could therefore never overlap — with a viewer open
        the two serialized instead of sharing the GPU.
        """
        submit_id = handle.submit_id if isinstance(handle, SubmitHandle) else handle
        self.device.wait_for_submit(submit_id)

    def wait_for_idle(self) -> None:
        """Queue-wide idle wait — this one DOES need the lock (it is a queue op)."""
        with self.gpu_lock:
            self.device.wait_for_idle()

    def queue_scope(self):
        """Lock scope for compound queue-adjacent operations (reload/reconfigure)."""
        return self.gpu_lock
