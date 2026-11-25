# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Prefix-scan primitive test.

Verifies the three-kernel Blelloch scan (``scanBlock``, ``scanBlockSums``,
``addBlockOffsets``) produces correct exclusive prefix sums on:
- trivial inputs (single block, many blocks)
- edge cases (all zeros, all ones, size exactly at block boundary)

Runs against CPU's ``np.cumsum`` as reference, on both backends.
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from trainer.backend.compute.execution import SCAN_BLOCK_SIZE
from trainer.backend.compute.device import SHADERS_DIR


@pytest.fixture(scope="module")
def scan_kernels(gpu_device):
    path = str(SHADERS_DIR / "common" / "scan.slang")
    kernels = {}
    for ep in ("scanBlock", "scanBlockSums", "addBlockOffsets"):
        program = gpu_device.load_program(path, [ep])
        kernels[ep] = gpu_device.create_compute_kernel(program)
    return gpu_device, kernels


def _scan(device: spy.Device, kernels: dict, data: np.ndarray) -> tuple[np.ndarray, int]:
    """Run the three-kernel scan on a uint32 array and return
    ``(exclusive_scan, total_sum)``."""
    n = data.shape[0]
    num_blocks = (n + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE
    assert num_blocks <= SCAN_BLOCK_SIZE, "test exceeds two-level scan capacity"

    in_buf = spy.NDBuffer(device=device, dtype="uint", shape=(n,))
    in_buf.copy_from_numpy(data.astype(np.uint32))
    out_buf = spy.NDBuffer(device=device, dtype="uint", shape=(n,))
    block_sums = spy.NDBuffer(device=device, dtype="uint", shape=(max(num_blocks, 1),))
    total_out = spy.NDBuffer(device=device, dtype="uint", shape=(1,))

    kernels["scanBlock"].dispatch(
        thread_count=[num_blocks * SCAN_BLOCK_SIZE, 1, 1],
        in_buf=in_buf.storage, out_buf=out_buf.storage,
        block_sums=block_sums.storage, n=n,
    )
    kernels["scanBlockSums"].dispatch(
        thread_count=[SCAN_BLOCK_SIZE, 1, 1],
        block_sums=block_sums.storage, total_out=total_out.storage,
        num_blocks=num_blocks,
    )
    kernels["addBlockOffsets"].dispatch(
        thread_count=[num_blocks * SCAN_BLOCK_SIZE, 1, 1],
        out_buf=out_buf.storage, block_sums=block_sums.storage, n=n,
    )
    device.wait_for_idle()

    result_raw = out_buf.to_numpy()
    result = np.frombuffer(result_raw.tobytes(), dtype=np.uint32)[:n]
    total = int(np.frombuffer(total_out.to_numpy().tobytes(), dtype=np.uint32)[0])
    return result, total


def _reference_exclusive(data: np.ndarray) -> tuple[np.ndarray, int]:
    """CPU reference: out[i] = sum(in[0..i-1]), total = sum(in)."""
    exc = np.concatenate([[0], np.cumsum(data[:-1])]).astype(np.uint32) if len(data) else np.zeros(0, np.uint32)
    return exc, int(data.sum())


@pytest.mark.parametrize("n", [1, 7, 256, 1023, 1024, 1025, 3000, 100_000])
def test_scan_correctness(scan_kernels, n: int) -> None:
    device, kernels = scan_kernels
    rng = np.random.default_rng(n)
    data = rng.integers(0, 4, size=n, dtype=np.uint32)

    got, got_total = _scan(device, kernels, data)
    want, want_total = _reference_exclusive(data)

    np.testing.assert_array_equal(got, want, err_msg=f"mismatch at n={n}")
    assert got_total == want_total, f"total mismatch at n={n}: got {got_total}, want {want_total}"


def test_scan_all_ones(scan_kernels) -> None:
    """All-ones input: exclusive scan is the arange, total = n."""
    device, kernels = scan_kernels
    n = 5000
    data = np.ones(n, dtype=np.uint32)
    got, total = _scan(device, kernels, data)
    np.testing.assert_array_equal(got, np.arange(n, dtype=np.uint32))
    assert total == n


def test_scan_all_zeros(scan_kernels) -> None:
    device, kernels = scan_kernels
    n = 5000
    data = np.zeros(n, dtype=np.uint32)
    got, total = _scan(device, kernels, data)
    np.testing.assert_array_equal(got, np.zeros(n, dtype=np.uint32))
    assert total == 0
