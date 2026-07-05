"""Radix sort correctness test.

Sorts random (key64, value) pairs with the ported DeviceRadixSort
(shaders/sort/device_radix_sort.slang) and compares against a numpy
reference (sort by full 64-bit key).
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from trainer.device import SHADERS_DIR, create_device


BACKENDS = ["vulkan", "d3d12"]


# Must match device_radix_sort.slang.
PART_SIZE = 2048
US_DIM = 128
SCAN_DIM = 128
D_DIM = 256
RADIX = 256
GLOBAL_HIST_SIZE = RADIX * 8  # 2048 — one 256-region per pass


def _make_device_and_kernels(backend: str):
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    path = str(SHADERS_DIR / "sort" / "device_radix_sort.slang")
    kernels = {}
    for ep in ("initGlobalHist", "upsweep", "scan", "downsweep"):
        program = device.load_program(path, [ep])
        kernels[ep] = device.create_compute_kernel(program)
    return device, kernels


def _radix_sort(device, kernels, keys_lo: np.ndarray, keys_hi: np.ndarray,
                values: np.ndarray):
    n = keys_lo.shape[0]
    # Capacity rounded up to a partition multiple; max_blocks derived from it.
    cap = max(PART_SIZE, ((n + PART_SIZE - 1) // PART_SIZE) * PART_SIZE)
    max_blocks = cap // PART_SIZE

    kA_lo = spy.NDBuffer(device=device, dtype="uint", shape=(cap,))
    kA_hi = spy.NDBuffer(device=device, dtype="uint", shape=(cap,))
    vA    = spy.NDBuffer(device=device, dtype="uint", shape=(cap,))
    kB_lo = spy.NDBuffer(device=device, dtype="uint", shape=(cap,))
    kB_hi = spy.NDBuffer(device=device, dtype="uint", shape=(cap,))
    vB    = spy.NDBuffer(device=device, dtype="uint", shape=(cap,))

    # Pad up to cap with SENTINEL (0xFFFFFFFF) so partial-partition loads
    # match what the shader would read out of a fully-allocated buffer.
    def _pad(a):
        out = np.full(cap, 0xFFFFFFFF, dtype=np.uint32)
        out[:n] = a.astype(np.uint32)
        return out
    kA_lo.copy_from_numpy(_pad(keys_lo))
    kA_hi.copy_from_numpy(_pad(keys_hi))
    vA.copy_from_numpy(_pad(values))

    pass_hist   = spy.NDBuffer(device=device, dtype="uint", shape=(RADIX * max_blocks,))
    global_hist = spy.NDBuffer(device=device, dtype="uint", shape=(GLOBAL_HIST_SIZE,))

    n_buf = spy.NDBuffer(device=device, dtype="uint", shape=(1,))
    n_buf.copy_from_numpy(np.array([n], dtype=np.uint32))

    # Clear global_hist once at the start of the whole sort.
    kernels["initGlobalHist"].dispatch(
        thread_count=[GLOBAL_HIST_SIZE, 1, 1],
        b_global_hist=global_hist.storage,
    )

    src_lo, src_hi, src_v = kA_lo, kA_hi, vA
    dst_lo, dst_hi, dst_v = kB_lo, kB_hi, vB
    for pass_idx in range(8):
        radix_shift = (pass_idx % 4) * 8
        key_word = 0 if pass_idx < 4 else 1

        kernels["upsweep"].dispatch(
            thread_count=[max_blocks * US_DIM, 1, 1],
            keys_lo=src_lo.storage, keys_hi=src_hi.storage,
            b_pass_hist=pass_hist.storage,
            b_global_hist=global_hist.storage,
            n_buf=n_buf.storage,
            radix_shift=radix_shift, key_word=key_word,
            pass_idx=pass_idx, max_blocks=max_blocks,
        )
        kernels["scan"].dispatch(
            thread_count=[RADIX * SCAN_DIM, 1, 1],
            b_pass_hist=pass_hist.storage,
            n_buf=n_buf.storage,
            max_blocks=max_blocks,
        )
        kernels["downsweep"].dispatch(
            thread_count=[max_blocks * D_DIM, 1, 1],
            keys_lo_in=src_lo.storage, keys_hi_in=src_hi.storage, values_in=src_v.storage,
            keys_lo_out=dst_lo.storage, keys_hi_out=dst_hi.storage, values_out=dst_v.storage,
            b_pass_hist=pass_hist.storage,
            b_global_hist=global_hist.storage,
            n_buf=n_buf.storage,
            radix_shift=radix_shift, key_word=key_word,
            pass_idx=pass_idx, max_blocks=max_blocks,
        )
        src_lo, dst_lo = dst_lo, src_lo
        src_hi, dst_hi = dst_hi, src_hi
        src_v, dst_v = dst_v, src_v

    device.wait_for_idle()
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
