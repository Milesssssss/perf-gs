"""Lock-free shared-memory stream of gaussian snapshots (trainer -> viewer).

Decouples the live viewer from the trainer completely: the trainer is a
single WRITER that pushes a CPU snapshot of the gaussian buffer into shared
memory every N iterations; a standalone viewer PROCESS is a single READER
that polls for new snapshots and renders them. No shared GPU device, no
shared VkQueue, no cross-thread submission — and the viewer crashing (or
being closed/reopened) cannot touch the training run.

Protocol: a seqlock over a double-buffered payload.
  - Two data slots. The writer fills the INACTIVE slot (never the one the
    reader is looking at), then commits by bumping `seq` odd -> writing the
    header pointers -> bumping `seq` even.
  - The reader samples `seq` before and after copying the active slot; if it
    changed (or was odd), a commit raced the copy, so it retries. Because the
    write cadence (N training iters, seconds) is orders of magnitude slower
    than a snapshot copy (ms), a retry essentially never fires in practice,
    but it guarantees the reader never sees a torn snapshot.

Single-writer / single-reader only. Aligned 64-bit stores are atomic on the
x86-64 / ARM64 targets we run on, which is all the seqlock relies on.

Layout (little-endian):
  header (HEADER_BYTES):
    0  magic   u32   'SPLT'
    4  version u32
    8  capacity u32  max splats per slot
    12 floats_per_splat u32  (== FLOATS_PER_SPLAT)
    16 seq     u64   seqlock counter (odd = commit in progress)
    24 active  u32   index of the slot holding the latest complete snapshot
    28 count   u32   live splats in the active slot
    32 iter    u32   training iteration (for display)
    36 closed  u32   set to 1 by the writer on shutdown
    40 start   u32   set to 1 by the READER (viewer) to release the trainer
                     from its pre-training wait. Reader->writer direction, the
                     mirror of `closed`. Ignored unless the trainer opted into
                     --wait-for-viewer.
    44..HEADER_BYTES  reserved
  slot 0: capacity * FLOATS_PER_SPLAT float32
  slot 1: capacity * FLOATS_PER_SPLAT float32
"""
from __future__ import annotations

from multiprocessing import shared_memory

import numpy as np

from trainer.contracts import GAUSSIAN_FLOAT_COUNT

MAGIC = 0x53504C54  # 'SPLT'
VERSION = 1
HEADER_BYTES = 64
# Packed Gaussian layout width — from the authoritative contract, not a local 14.
FLOATS_PER_SPLAT = GAUSSIAN_FLOAT_COUNT  # mean3, log_scale3, rot_quat4, opacity1, sh0_3

# The trainer writes its live stream name here; the viewer reads it when no
# --stream is given. Single source of truth for both processes.
STREAM_LOCK_FILE_DEFAULT = "outputs/.splat_stream"

# Header field byte offsets.
_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_CAPACITY = 8
_OFF_FLOATS = 12
_OFF_SEQ = 16
_OFF_ACTIVE = 24
_OFF_COUNT = 28
_OFF_ITER = 32
_OFF_CLOSED = 36
_OFF_START = 40


def _u32(buf, off):
    return np.frombuffer(buf, dtype=np.uint32, count=1, offset=off)


def _u64(buf, off):
    return np.frombuffer(buf, dtype=np.uint64, count=1, offset=off)


def _slot_size_bytes(capacity: int) -> int:
    return capacity * FLOATS_PER_SPLAT * 4


def _total_size_bytes(capacity: int) -> int:
    return HEADER_BYTES + 2 * _slot_size_bytes(capacity)


class SplatStreamWriter:
    """Trainer-side writer. Owns the shared-memory block.

    Usage:
        writer = SplatStreamWriter(capacity=2_000_000)
        print(writer.name)              # pass this to the viewer process
        ...
        writer.push(pipeline, iteration)  # every N iters
        ...
        writer.close()                    # on shutdown (unlinks the block)
    """

    def __init__(self, capacity: int = 2_000_000, name: str | None = None):
        self.capacity = int(capacity)
        size = _total_size_bytes(self.capacity)
        self._shm = shared_memory.SharedMemory(create=True, size=size, name=name)
        self.name = self._shm.name
        buf = self._shm.buf

        # Views onto header scalars and the two data slots.
        self._magic = _u32(buf, _OFF_MAGIC)
        self._version = _u32(buf, _OFF_VERSION)
        self._capacity_h = _u32(buf, _OFF_CAPACITY)
        self._floats_h = _u32(buf, _OFF_FLOATS)
        self._seq = _u64(buf, _OFF_SEQ)
        self._active = _u32(buf, _OFF_ACTIVE)
        self._count = _u32(buf, _OFF_COUNT)
        self._iter = _u32(buf, _OFF_ITER)
        self._closed = _u32(buf, _OFF_CLOSED)
        self._start = _u32(buf, _OFF_START)

        slot_floats = self.capacity * FLOATS_PER_SPLAT
        s0 = HEADER_BYTES
        s1 = HEADER_BYTES + _slot_size_bytes(self.capacity)
        self._slots = [
            np.frombuffer(buf, dtype=np.float32, count=slot_floats, offset=s0),
            np.frombuffer(buf, dtype=np.float32, count=slot_floats, offset=s1),
        ]

        self._magic[0] = MAGIC
        self._version[0] = VERSION
        self._capacity_h[0] = self.capacity
        self._floats_h[0] = FLOATS_PER_SPLAT
        self._seq[0] = 0
        self._active[0] = 0
        self._count[0] = 0
        self._iter[0] = 0
        self._closed[0] = 0
        self._start[0] = 0
        self._warned_overflow = False

    def write_snapshot(self, floats: np.ndarray, count: int, iteration: int) -> None:
        """Commit a snapshot. ``floats`` is a flat float32 array holding at
        least ``count * FLOATS_PER_SPLAT`` values in the 14-float layout."""
        count = int(count)
        if count > self.capacity:
            if not self._warned_overflow:
                print(f"[SplatStream] snapshot count {count} exceeds stream capacity "
                      f"{self.capacity}; streaming first {self.capacity} splats. "
                      f"Raise --stream-max-splats to see all.")
                self._warned_overflow = True
            count = self.capacity

        n_floats = count * FLOATS_PER_SPLAT
        inactive = 1 - int(self._active[0])
        # Fill the slot the reader is NOT looking at.
        self._slots[inactive][:n_floats] = floats[:n_floats]

        # Commit: odd seq (writing) -> flip pointers -> even seq (done).
        self._seq[0] = self._seq[0] + 1        # -> odd
        self._active[0] = inactive
        self._count[0] = count
        self._iter[0] = iteration
        self._seq[0] = self._seq[0] + 1        # -> even

    def push(self, pipeline, iteration: int) -> None:
        """Read the live gaussian buffer off the GPU and stream it.

        Piggyback this on a point where the training loop is already syncing
        (log/preview/densify) so the GPU->CPU readback is not an extra stall.
        """
        pipeline.runtime.wait_for_idle()
        raw = pipeline.buffers.gaussians.to_numpy()
        floats = np.frombuffer(raw, dtype=np.float32)
        count = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
        self.write_snapshot(floats, count, iteration)

    def start_requested(self) -> bool:
        """True once a viewer has pressed 'Start Training'. Reader->writer
        signal; polled by the trainer when it opted into --wait-for-viewer.
        Always False if no viewer ever attaches, so the trainer must provide
        its own escape hatch (keypress / timeout) to avoid hanging headless."""
        try:
            return int(self._start[0]) == 1
        except Exception:
            return True  # stream torn down -> don't block training forever

    def signal_closed(self) -> None:
        """Set the shared `closed` flag so a reading viewer exits on its own.
        Does NOT unlink — call close() after the viewer has had a chance to
        see the flag, so the reader never dereferences freed memory."""
        try:
            self._closed[0] = 1
        except Exception:
            pass

    def close(self) -> None:
        self.signal_closed()
        # Drop numpy views before releasing the buffer (they hold references).
        self._magic = self._version = self._capacity_h = self._floats_h = None
        self._seq = self._active = self._count = self._iter = self._closed = None
        self._start = None
        self._slots = None
        try:
            self._shm.close()
            self._shm.unlink()
        except Exception:
            pass


class SplatStreamReader:
    """Viewer-side reader. Attaches to an existing shared-memory block by name.

    ``latest()`` returns ``(floats, count, iteration)`` for a NEW snapshot
    since the last successful read, or ``None`` if nothing changed / the read
    could not be made consistent this poll (caller just tries again next
    frame).
    """

    def __init__(self, name: str, retries: int = 8):
        self._shm = shared_memory.SharedMemory(create=False, name=name)
        buf = self._shm.buf
        self._retries = retries

        magic = int(_u32(buf, _OFF_MAGIC)[0])
        if magic != MAGIC:
            raise ValueError(f"shared memory {name!r} is not a splat stream "
                             f"(magic {magic:#x} != {MAGIC:#x})")
        self._version = int(_u32(buf, _OFF_VERSION)[0])
        self.capacity = int(_u32(buf, _OFF_CAPACITY)[0])
        self._floats_per = int(_u32(buf, _OFF_FLOATS)[0])

        self._seq = _u64(buf, _OFF_SEQ)
        self._active = _u32(buf, _OFF_ACTIVE)
        self._count = _u32(buf, _OFF_COUNT)
        self._iter = _u32(buf, _OFF_ITER)
        self._closed = _u32(buf, _OFF_CLOSED)
        self._start = _u32(buf, _OFF_START)

        slot_floats = self.capacity * self._floats_per
        s0 = HEADER_BYTES
        s1 = HEADER_BYTES + _slot_size_bytes(self.capacity)
        self._slots = [
            np.frombuffer(buf, dtype=np.float32, count=slot_floats, offset=s0),
            np.frombuffer(buf, dtype=np.float32, count=slot_floats, offset=s1),
        ]
        self._last_seq = -1

    def is_closed(self) -> bool:
        return int(self._closed[0]) == 1

    def request_start(self) -> None:
        """Signal the (waiting) trainer to begin training. Idempotent."""
        try:
            self._start[0] = 1
        except Exception:
            pass

    def start_requested(self) -> bool:
        try:
            return int(self._start[0]) == 1
        except Exception:
            return False

    def latest(self):
        """Return (floats_copy, count, iteration) for a new snapshot, else None."""
        for _ in range(self._retries):
            s0 = int(self._seq[0])
            if s0 == 0:
                return None  # nothing committed yet
            if s0 & 1:
                continue  # commit in progress
            if s0 == self._last_seq:
                return None  # nothing new since last read
            slot = int(self._active[0])
            count = int(self._count[0])
            iteration = int(self._iter[0])
            if slot not in (0, 1) or count < 0 or count > self.capacity:
                continue  # header not yet consistent
            n_floats = count * self._floats_per
            data = self._slots[slot][:n_floats].copy()
            s1 = int(self._seq[0])
            if s0 == s1:
                self._last_seq = s0
                return data, count, iteration
            # else a commit raced our copy — retry
        return None

    def close(self) -> None:
        self._seq = self._active = self._count = self._iter = self._closed = None
        self._start = None
        self._slots = None
        try:
            self._shm.close()
        except Exception:
            pass
