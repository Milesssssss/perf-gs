# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Commits scatters staged in the shared scratch arena back into live buffers."""

from __future__ import annotations

import slangpy as spy

from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import ScratchArena


class ArenaCommitter:
    """Copies the live prefix of a staged scatter back, using the GPU-resident count."""

    KERNELS = {
        "prepare_args": ("densify/copy_active.slang", "prepareActiveCopyArgs"),
        "copy_records": ("densify/copy_active.slang", "copyActiveRecords"),
    }

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary,
                 arena: ScratchArena, capacity: int) -> None:
        self._commands = commands
        self._k = kernels.load(self.KERNELS)
        self.arena = arena
        self.capacity = capacity

    @property
    def storage(self):
        """Bind target for a scatter kernel's ``*_out`` parameter."""
        return self.arena.storage

    def commit(self, dst: spy.NDBuffer, live_count: spy.NDBuffer, *,
               nbytes: int | None = None, soa: bool = False,
               dst_offset_bytes: int = 0) -> None:
        """Copy ``live_count[0]`` staged records into ``dst`` inside the active batch.

        ``nbytes`` (default: all of ``dst``) spans ``capacity`` records; SoA
        resources keep their capacity stride, and an offset commits one batch of planes.
        """
        assert self._commands.batch is not None, "arena commits need an active batch"
        if nbytes is None:
            nbytes = dst.storage.size
        if (dst_offset_bytes < 0 or dst_offset_bytes % 4
                or dst_offset_bytes + nbytes > dst.storage.size):
            raise ValueError("arena copy must fit an aligned destination span")
        assert nbytes <= self.arena.capacity_bytes, "scratch arena too small"
        record_bytes, remainder = divmod(nbytes, self.capacity)
        if remainder or record_bytes % 4:
            raise ValueError("active copy requires whole uint-sized records")
        commands = self._commands
        with commands.marker_group("arena_copy_back"):
            commands.dispatch(self._k.prepare_args, "active_copy_prepare_args",
                thread_count=[1, 1, 1], count_buf=live_count.storage,
                dispatch_args=self.arena.copy_dispatch_args.storage,
                record_words=record_bytes // 4, max_splats=self.capacity)
            commands.dispatch_indirect(self._k.copy_records, "copy_active_records",
                self.arena.copy_dispatch_args, source=self.arena.storage,
                destination=dst.storage, count_buf=live_count.storage,
                record_words=record_bytes // 4, soa=int(soa),
                destination_word_offset=dst_offset_bytes // 4,
                max_splats=self.capacity)

    def reorder_planes(self, buffer: spy.NDBuffer, kernel, marker: str, plane_count: int,
                       live_count: spy.NDBuffer, **permutation) -> None:
        """Permute capacity-strided SoA planes in batches that fit the arena.

        Each batch reads its complete source planes before overwriting them, and
        later batches only read untouched planes, so any permutation is safe.
        """
        plane_bytes = self.capacity * 4
        batch_planes = self.arena.capacity_bytes // plane_bytes
        if batch_planes < 1:
            raise ValueError("scratch arena must hold at least one complete SoA plane")
        for first in range(0, plane_count, batch_planes):
            count = min(batch_planes, plane_count - first)
            self._commands.dispatch(
                kernel, marker, thread_count=[self.capacity, 1, 1],
                adam_rest_in=buffer.storage, adam_rest_out=self.arena.storage,
                first_plane=first, plane_count=count, max_splats=self.capacity,
                **permutation)
            self.commit(buffer, live_count, nbytes=count * plane_bytes, soa=True,
                        dst_offset_bytes=first * plane_bytes)
