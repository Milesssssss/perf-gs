"""Radix sort correctness test.

Sorts random (key64, value) pairs with our 8-pass LSB radix sort and
compares against numpy reference (sort by full 64-bit key).
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from trainer.device import SHADERS_DIR, create_device


BACKENDS = ["vulkan", "d3d12"]


SORT_BLOCK_SIZE = 256  # must match radix_sort.slang


def _make_device_and_kernels(backend: str):
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    path = str(SHADERS_DIR / "sort" / "radix_sort.slang")
    kernels = {}
    for ep in ("blockSort", "globalScan", "scanDigitTotals", "scatterFinal"):
        program = device.load_program(path, [ep])
        kernels[ep] = device.create_compute_kernel(program)
    return device, kernels


def _radix_sort(device, kernels, keys_lo: np.ndarray, keys_hi: np.ndarray,
                values: np.ndarray):
    # Buffers sized to a block-aligned upper bound so the shader can
    # dispatch whole blocks without checking per-thread overrun.
    n = keys_lo.shape[0]
    cap = max(SORT_BLOCK_SIZE, ((n + SORT_BLOCK_SIZE - 1) // SORT_BLOCK_SIZE) * SORT_BLOCK_SIZE)
    num_blocks = cap // SORT_BLOCK_SIZE

    kA_lo = spy.NDBuffer(device=device, dtype="uint", shape=(cap,))
    kA_hi = spy.NDBuffer(device=device, dtype="uint", shape=(cap,))
    vA    = spy.NDBuffer(device=device, dtype="uint", shape=(cap,))
    kB_lo = spy.NDBuffer(device=device, dtype="uint", shape=(cap,))
    kB_hi = spy.NDBuffer(device=device, dtype="uint", shape=(cap,))
    vB    = spy.NDBuffer(device=device, dtype="uint", shape=(cap,))

    # Pad input arrays up to cap with 0xFFFFFFFF so any OOB reads
    # are treated as sentinels by the shader (matches its n_buf-gated
    # logic).
    def _pad(a):
        out = np.full(cap, 0xFFFFFFFF, dtype=np.uint32)
        out[:n] = a.astype(np.uint32)
        return out
    kA_lo.copy_from_numpy(_pad(keys_lo))
    kA_hi.copy_from_numpy(_pad(keys_hi))
    vA.copy_from_numpy(_pad(values))

    block_hist   = spy.NDBuffer(device=device, dtype="uint", shape=(num_blocks * 256,))
    block_start  = spy.NDBuffer(device=device, dtype="uint", shape=(num_blocks * 256,))
    block_prefix = spy.NDBuffer(device=device, dtype="uint", shape=(num_blocks * 256,))
    digit_total  = spy.NDBuffer(device=device, dtype="uint", shape=(256,))
    digit_base   = spy.NDBuffer(device=device, dtype="uint", shape=(256,))

    n_buf = spy.NDBuffer(device=device, dtype="uint", shape=(1,))
    n_buf.copy_from_numpy(np.array([n], dtype=np.uint32))

    src_lo, src_hi, src_v = kA_lo, kA_hi, vA
    dst_lo, dst_hi, dst_v = kB_lo, kB_hi, vB
    for pass_idx in range(8):
        kernels["blockSort"].dispatch(
            thread_count=[num_blocks * 256, 1, 1],
            keys_lo=src_lo.storage, keys_hi=src_hi.storage, values=src_v.storage,
            block_hist=block_hist.storage, block_start=block_start.storage,
            n_buf=n_buf.storage, pass_idx=pass_idx,
        )
        kernels["globalScan"].dispatch(
            thread_count=[256, 1, 1],
            block_hist=block_hist.storage,
            block_prefix=block_prefix.storage,
            digit_total=digit_total.storage,
            n_buf=n_buf.storage,
        )
        kernels["scanDigitTotals"].dispatch(
            thread_count=[256, 1, 1],
            digit_total=digit_total.storage,
            digit_global_base=digit_base.storage,
        )
        kernels["scatterFinal"].dispatch(
            thread_count=[cap, 1, 1],
            keys_lo_in=src_lo.storage, keys_hi_in=src_hi.storage, values_in=src_v.storage,
            block_start=block_start.storage,
            block_prefix=block_prefix.storage,
            digit_global_base=digit_base.storage,
            keys_lo_out=dst_lo.storage, keys_hi_out=dst_hi.storage, values_out=dst_v.storage,
            n_buf=n_buf.storage, pass_idx=pass_idx,
        )
        src_lo, dst_lo = dst_lo, src_lo
        src_hi, dst_hi = dst_hi, src_hi
        src_v, dst_v = dst_v, src_v

    device.wait_for_idle()
    # Only the first n slots hold real sorted data; slots [n, cap) still
    # hold whatever sentinel state the sort left there, which we ignore.
    out_lo = np.frombuffer(src_lo.to_numpy().tobytes(), dtype=np.uint32)[:n]
    out_hi = np.frombuffer(src_hi.to_numpy().tobytes(), dtype=np.uint32)[:n]
    out_v  = np.frombuffer(src_v.to_numpy().tobytes(), dtype=np.uint32)[:n]
    return out_lo, out_hi, out_v


def _reference_sort(keys_lo: np.ndarray, keys_hi: np.ndarray, values: np.ndarray):
    """Sort by full 64-bit key ascending; return reordered arrays."""
    keys64 = (keys_hi.astype(np.uint64) << np.uint64(32)) | keys_lo.astype(np.uint64)
    # numpy argsort is not stable by default for integer but is for uint64
    order = np.argsort(keys64, kind="stable")
    return keys_lo[order], keys_hi[order], values[order]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [1, 257, 1000, 100_000])
def test_radix_sort(backend: str, n: int) -> None:
    device, kernels = _make_device_and_kernels(backend)
    rng = np.random.default_rng(n)
    keys_lo = rng.integers(0, 2**32, size=n, dtype=np.uint32)
    keys_hi = rng.integers(0, 2**32, size=n, dtype=np.uint32)
    values = rng.integers(0, 2**32, size=n, dtype=np.uint32)

    got_lo, got_hi, got_v = _radix_sort(device, kernels, keys_lo, keys_hi, values)
    want_lo, want_hi, want_v = _reference_sort(keys_lo, keys_hi, values)

    np.testing.assert_array_equal(got_lo, want_lo, err_msg=f"keys_lo mismatch @ n={n}")
    np.testing.assert_array_equal(got_hi, want_hi, err_msg=f"keys_hi mismatch @ n={n}")
    np.testing.assert_array_equal(got_v, want_v,  err_msg=f"values mismatch @ n={n}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_radix_sort_all_same_key(backend: str) -> None:
    """Degenerate case: all keys identical. Sort should be stable-ish
    (preserving value order isn't guaranteed under atomics, but values
    should all be preserved as a set)."""
    device, kernels = _make_device_and_kernels(backend)
    n = 1000
    keys_lo = np.full(n, 42, dtype=np.uint32)
    keys_hi = np.full(n, 7, dtype=np.uint32)
    values = np.arange(n, dtype=np.uint32)
    got_lo, got_hi, got_v = _radix_sort(device, kernels, keys_lo, keys_hi, values)
    np.testing.assert_array_equal(got_lo, keys_lo)
    np.testing.assert_array_equal(got_hi, keys_hi)
    # Values: the set should be identical (a permutation).
    np.testing.assert_array_equal(np.sort(got_v), values)


@pytest.mark.parametrize("backend", BACKENDS)
def test_radix_sort_tile_depth_shape(backend: str) -> None:
    """Realistic use case: keys_hi is tile_id, keys_lo is sortable depth.
    After sort, keys with same tile_id should be contiguous, and within
    each tile group, keys_lo should be ascending."""
    device, kernels = _make_device_and_kernels(backend)
    n = 50_000
    rng = np.random.default_rng(7)
    # 2000 tiles, depth as arbitrary uint32.
    keys_hi = rng.integers(0, 2000, size=n, dtype=np.uint32)
    keys_lo = rng.integers(0, 2**32, size=n, dtype=np.uint32)
    values = rng.integers(0, 2**32, size=n, dtype=np.uint32)

    got_lo, got_hi, got_v = _radix_sort(device, kernels, keys_lo, keys_hi, values)
    # Monotonic tile_id.
    assert np.all(got_hi[1:] >= got_hi[:-1]), "tile_id not monotonic"
    # Within each tile, depth ascending.
    for tile_id in np.unique(got_hi):
        mask = got_hi == tile_id
        seg = got_lo[mask]
        assert np.all(seg[1:] >= seg[:-1]), f"depth not ascending within tile {tile_id}"
