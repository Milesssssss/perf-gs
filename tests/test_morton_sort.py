# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU Morton-order sort of the gaussian + adam-state buffers.

Verifies that after one ``pipeline.morton_sort(world_min, world_max)``:

- Every input gaussian appears exactly once in the output (permutation,
  not lossy filter).
- The resulting sequence of Morton codes is monotonically non-decreasing.
- adam_state is permuted with the SAME permutation gaussians are
  (paired-move contract; if it drifts, the optimizer state desyncs).
- Tail slots [num_gaussians, max_splats) are left untouched (the render
  path never reads them).
"""

from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.backend.compute.device import create_device


BACKENDS = ["vulkan", "d3d12"]


def _setup(backend: str, max_splats: int = 64) -> TestPipeline:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return TestPipeline(device=device, width=32, height=32, max_splats=max_splats)


def _spread21(v: int) -> int:
    """Python mirror of _spread21 in morton_sort.slang (for the oracle)."""
    x = v & 0x1FFFFF
    x = (x | (x << 32)) & 0x1F00000000FFFF
    x = (x | (x << 16)) & 0x1F0000FF0000FF
    x = (x | (x <<  8)) & 0x100F00F00F00F00F
    x = (x | (x <<  4)) & 0x10C30C30C30C30C3
    x = (x | (x <<  2)) & 0x1249249249249249
    return x


def _morton63(x: int, y: int, z: int) -> int:
    return _spread21(x) | (_spread21(y) << 1) | (_spread21(z) << 2)


def _quantize(p: np.ndarray, world_min: np.ndarray,
              world_max: np.ndarray) -> np.ndarray:
    extent = np.maximum(world_max - world_min, 1e-6)
    t = np.clip((p - world_min) / extent, 0.0, 1.0)
    return (t * 2097151.0).astype(np.uint32)  # 2^21 - 1


def _cpu_morton(means: np.ndarray, world_min: np.ndarray,
                world_max: np.ndarray) -> np.ndarray:
    q = _quantize(means, world_min, world_max)
    return np.array([_morton63(int(qi[0]), int(qi[1]), int(qi[2])) for qi in q],
                    dtype=np.uint64)


@pytest.mark.parametrize("backend", BACKENDS)
def test_morton_sort_permutes_gaussians(backend: str) -> None:
    max_splats = 64
    pipeline = _setup(backend, max_splats=max_splats)

    # 32 gaussians deliberately scrambled in memory. Colors + opacities
    # made unique per index so we can identify each one after sorting.
    n = 32
    rng = np.random.default_rng(0)
    means = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
    log_scales = np.log(np.full((n, 3), 0.05, dtype=np.float32))
    rot_quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    opacities = np.linspace(0.1, 0.9, n).astype(np.float32)
    # Distinctive per-slot fingerprint in sh0 so we can verify pairing.
    color_logits = np.stack([np.arange(n, dtype=np.float32),
                             np.arange(n, dtype=np.float32) * 2.0,
                             np.arange(n, dtype=np.float32) * 3.0], axis=1)

    pipeline.set_primitive_from_numpy(
        means=means, log_scales=log_scales, rot_quats=rot_quats,
        opacities=opacities, color_logits=color_logits,
    )

    # Fingerprint adam_state per slot too, so a permute-mismatch shows up
    # as a paired-move violation. AdamState is 28 float32s = 112 bytes.
    adam_raw = np.zeros((max_splats, 28), dtype=np.float32)
    # Encode slot index in adam_state.mean_m.x for later cross-check.
    adam_raw[:, 0] = np.arange(max_splats, dtype=np.float32)
    pipeline.buffers.adam_state.copy_from_numpy(adam_raw)

    world_min = means.min(axis=0) - 1.0
    world_max = means.max(axis=0) + 1.0
    pipeline.morton_sort(world_min, world_max)
    pipeline.device.wait_for_idle()

    # Read back post-sort state.
    raw = pipeline.buffers.gaussians.to_numpy()
    floats = np.frombuffer(raw.tobytes(), dtype=np.float32).reshape(-1, 14)
    sorted_means = floats[:n, 0:3]
    sorted_sh0 = floats[:n, 11:14]

    # (1) Permutation, not lossy: the multiset of means matches input.
    #     Sort both by lex(mean.xyz) and compare.
    def _lexsort(arr):
        return arr[np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))]
    assert np.allclose(_lexsort(sorted_means), _lexsort(means), atol=1e-5), \
        "morton_sort lost or duplicated a gaussian"

    # (2) Morton codes of the sorted means are monotonically non-decreasing.
    codes = _cpu_morton(sorted_means, world_min, world_max)
    assert np.all(codes[1:] >= codes[:-1]), \
        f"Morton codes not monotonic: first violation at {int(np.argmax(codes[1:] < codes[:-1]))}"

    # (3) adam_state permutation matches gaussians permutation. We stamped
    #     mean_m.x = slot_index; sh0.r = slot_index (color_logits[..., 0]).
    #     After sort, each output slot i must have mean_m.x == sh0.r.
    adam_out = pipeline.buffers.adam_state.to_numpy()
    adam_floats = np.frombuffer(adam_out.tobytes(), dtype=np.float32).reshape(-1, 28)
    sorted_mean_m_x = adam_floats[:n, 0]
    sorted_sh0_r = sorted_sh0[:, 0]
    assert np.allclose(sorted_mean_m_x, sorted_sh0_r), (
        "adam_state not moved in lockstep with gaussians: "
        f"first mismatch idx {int(np.argmax(sorted_mean_m_x != sorted_sh0_r))}")

    # (4) num_gaussians_counter is unchanged (permutation, not resize).
    counter = pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0]
    assert int(counter) == n, f"counter changed after Morton: {counter} != {n}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_morton_sort_empty_scene_is_noop(backend: str) -> None:
    """n=0 must not crash and must leave state alone."""
    pipeline = _setup(backend, max_splats=32)
    world_min = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    world_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    pipeline.morton_sort(world_min, world_max)
    pipeline.device.wait_for_idle()
    counter = pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0]
    assert int(counter) == 0
