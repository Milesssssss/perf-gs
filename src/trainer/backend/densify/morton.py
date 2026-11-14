# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Morton (Z-order) re-sort of the live primitives for cache locality."""

from __future__ import annotations

import numpy as np
import slangpy as spy

from trainer.backend.compute.algorithms import PrefixScan, RadixSort
from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.arena import ArenaCommitter
from trainer.backend.memory.buffers import (
    AdamResources,
    EtaResources,
    PrimitiveStorage,
    SphericalHarmonicsResources,
)

_MORTON = "densify/morton_sort.slang"


def morton_bounds(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The quantization box: the points' AABB padded 20% per axis.

    Means that later drift outside it saturate, which costs locality, not correctness.
    """
    means = means.astype(np.float32).reshape(-1, 3)
    low, high = means.min(axis=0), means.max(axis=0)
    extent = np.maximum(high - low, 1e-6)
    return low - 0.2 * extent, high + 0.2 * extent


class MortonReorder:
    """Permutes primitives (and their optimizer state) into Morton order for cache locality.

    Slot-indexed densification statistics are not permuted: the caller clears them afterwards.
    """

    KERNELS = {
        "snapshot_count": ("densify/densify_dispatch.slang", "snapshotCount"),
        "encode": (_MORTON, "mortonEncode"),
        "scatter_gaussian": (_MORTON, "mortonScatterGaussian"),
        "scatter_adam": (_MORTON, "mortonScatterAdam"),
        "scatter_features": (_MORTON, "mortonScatterFeatures"),
        "scatter_adam_rest": (_MORTON, "mortonScatterAdamRestPlanes"),
        "scatter_eta": (_MORTON, "mortonScatterEta"),
    }

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary, *,
                 primitive: PrimitiveStorage, adam: AdamResources,
                 eta: EtaResources | None, sh: SphericalHarmonicsResources | None,
                 sort: RadixSort, scan: PrefixScan, arena: ArenaCommitter,
                 capacity: int) -> None:
        self._commands = commands
        self._k = kernels.load(self.KERNELS)
        self._primitive = primitive
        self._adam = adam
        self._eta = eta
        self._sh = sh
        self._sort = sort
        self._count = scan.resources.total
        self._arena = arena
        self._capacity = capacity

    def reorder(self, world_min, world_max, carry_eta: bool = False) -> None:
        """Sort by each mean's Morton code inside the given box; a stale box only degrades locality."""
        world_min = np.asarray(world_min, dtype=np.float32).reshape(3)
        world_max = np.asarray(world_max, dtype=np.float32).reshape(3)
        inv_extent = (1.0 / np.maximum(world_max - world_min, 1e-6)).astype(np.float32)
        commands, k, capacity = self._commands, self._k, self._capacity
        primitive, adam, sh, arena = self._primitive, self._adam, self._sh, self._arena
        count = self._count
        pairs = self._sort.resources

        with commands.recording("morton_sort") as recording, commands.marker_group("morton_sort"):
            commands.dispatch(k.snapshot_count, "morton_snapshot",
                thread_count=[1, 1, 1],
                src=primitive.active_count.storage,
                dst=count.storage)
            commands.dispatch(k.encode, "morton_encode",
                thread_count=[capacity, 1, 1],
                gaussians=primitive.values.storage,
                count_buf=count.storage,
                keys_lo=pairs.keys_lo.storage,
                keys_hi=pairs.keys_hi.storage,
                values=pairs.values.storage,
                world_min=spy.math.float3(*world_min.tolist()),
                inv_extent=spy.math.float3(*inv_extent.tolist()),
                max_splats=capacity)
            _, _, order = self._sort.sort(count)

            commands.dispatch(k.scatter_gaussian, "morton_scatter_gaussian",
                thread_count=[capacity, 1, 1],
                gaussians_in=primitive.values.storage,
                sorted_values=order.storage,
                count_buf=count.storage,
                gaussians_out=arena.storage,
                max_splats=capacity)
            arena.commit(primitive.values, count)
            commands.dispatch(k.scatter_adam, "morton_scatter_adam",
                thread_count=[capacity, 1, 1],
                adam_state_in=adam.state.storage,
                sorted_values=order.storage,
                count_buf=count.storage,
                adam_state_out=arena.storage,
                max_splats=capacity)
            arena.commit(adam.state, count)
            if sh is not None:
                commands.dispatch(k.scatter_features, "morton_scatter_features",
                    thread_count=[capacity, 1, 1],
                    features_rest_in=sh.features_rest.storage,
                    sorted_values=order.storage,
                    count_buf=count.storage,
                    features_rest_out=arena.storage,
                    rest_floats=sh.rest_floats,
                    max_splats=capacity)
                arena.commit(sh.features_rest, count)
                arena.reorder_planes(
                    sh.adam_rest, k.scatter_adam_rest, "morton_scatter_adam_rest",
                    sh.moment_planes, count,
                    sorted_values=order.storage, count_buf=count.storage)
            if carry_eta:
                commands.dispatch(k.scatter_eta, "morton_scatter_eta",
                    thread_count=[capacity, 1, 1],
                    sorted_values=order.storage,
                    count_buf=count.storage,
                    **self._eta.scatter_kwargs(),
                    max_splats=capacity)
                recording.after_submit(self._eta.swap)
