# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Indirect dispatch group-count helpers, and the 0.42.0 Metal readback fallback.

slangpy 0.42.0 records the indirect command and then drops it on Metal. The
fallback submits the writer, reads the three uint32 group counts, and launches
that many groups of the same kernel. Newer slangpy builds that implement Metal
``dispatchComputeIndirect`` use the hardware path by default. This test forces
the fallback on the CPU device.
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from trainer.backend.memory.ndbuffer import make_ndbuffer

from trainer.backend.compute.device import SHADERS_DIR
from trainer.backend.compute.gpu import (
    GpuRuntime,
    indirect_group_counts,
    thread_count_for_groups,
)


def test_group_counts_honor_the_byte_offset() -> None:
    words = np.array([9, 8, 7, 4, 2, 1], dtype=np.uint32)
    assert indirect_group_counts(words, 0) == (9, 8, 7)
    assert indirect_group_counts(words, 12) == (4, 2, 1)


def test_thread_count_is_groups_times_reflected_size() -> None:
    class _Entry:
        compute_thread_group_size = (256, 1, 1)

    class _Layout:
        entry_points = (_Entry(),)

    class _Program:
        layout = _Layout()

    class _Kernel:
        program = _Program()

    assert thread_count_for_groups(_Kernel(), (3, 2, 1)) == [768, 2, 1]


def test_replay_matches_the_indirect_group_counts() -> None:
    """A GPU-written dispatch grid copies the same words through the replay path."""
    try:
        device = spy.create_device(type=spy.DeviceType.cpu, include_paths=[SHADERS_DIR])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cpu device unavailable: {exc}")
    path = str(SHADERS_DIR / "densify" / "copy_active.slang")
    try:
        prepare = device.create_compute_kernel(device.load_program(path, ["prepareActiveCopyArgs"]))
        copy = device.create_compute_kernel(device.load_program(path, ["copyActiveRecords"]))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"copy kernels unavailable on cpu: {exc}")

    capacity, count, record_words = 8, 5, 4
    source = np.arange(capacity * record_words, dtype=np.uint32)
    original = np.full(source.shape, 0xDEADBEEF, np.uint32)
    src = make_ndbuffer(device, "uint", source.shape)
    dst = make_ndbuffer(device, "uint", original.shape)
    count_buf = make_ndbuffer(device, "uint", (1))
    args = make_ndbuffer(device, "uint", (3),
                        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
                        | spy.BufferUsage.indirect_argument)
    src.copy_from_numpy(source)
    dst.copy_from_numpy(original)
    count_buf.copy_from_numpy(np.array([count], np.uint32))
    args.copy_from_numpy(np.zeros(3, np.uint32))

    runtime = GpuRuntime(device)
    batch = runtime.command_batch("replay_copy")
    batch.emulate_indirect = True
    batch.dispatch(prepare, "prepare", thread_count=[1, 1, 1],
                   count_buf=count_buf.storage, dispatch_args=args.storage,
                   record_words=record_words, max_splats=capacity)
    batch.dispatch_indirect(copy, "copy", args, source=src.storage,
                            destination=dst.storage, count_buf=count_buf.storage,
                            record_words=record_words, soa=0,
                            destination_word_offset=0, max_splats=capacity)
    runtime.wait_for_submit(runtime.submit(batch))

    expected = original.copy()
    expected[:count * record_words] = source[:count * record_words]
    np.testing.assert_array_equal(dst.to_numpy().view(np.uint32), expected)
    # prepare wrote one group in x for 20 words (group size 256).
    assert tuple(int(v) for v in args.to_numpy()[:3]) == (1, 1, 1)
