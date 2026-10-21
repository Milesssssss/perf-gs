# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Bounded scratch for in-place, capacity-strided SoA permutations."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import slangpy as spy

    from trainer.backend.memory.buffers import ScratchArena


class ReorderCommands(Protocol):
    def dispatch(self, kernel: Any, marker: str, **kwargs: Any) -> None: ...

    def copy_arena_back(
        self, arena: ScratchArena, dst: spy.NDBuffer, nbytes: int, *,
        count_buf: spy.NDBuffer, soa: bool, dst_offset_bytes: int = 0,
    ) -> None: ...


def reorder_soa_planes(
    *, commands: ReorderCommands, arena: ScratchArena, buffer: spy.NDBuffer,
    kernel: Any, marker: str, plane_count: int, capacity: int,
    live_count: spy.NDBuffer, **permutation: Any,
) -> None:
    """Scatter independent FP32/uint planes in batches, then commit live prefixes.

    A batch reads complete source planes before overwriting those same planes.
    Later batches only read untouched planes, so arbitrary permutations remain
    safe. All commands stay in the caller's batch; no count readback or wait.
    """
    plane_bytes = capacity * 4
    batch_planes = arena.capacity_bytes // plane_bytes
    if batch_planes < 1:
        raise ValueError("scratch arena must hold at least one complete SoA plane")
    for first in range(0, plane_count, batch_planes):
        count = min(batch_planes, plane_count - first)
        commands.dispatch(
            kernel, marker, thread_count=[capacity, 1, 1],
            adam_rest_in=buffer.storage, adam_rest_out=arena.storage,
            first_plane=first, plane_count=count, max_splats=capacity,
            **permutation,
        )
        commands.copy_arena_back(
            arena, buffer, count * plane_bytes, count_buf=live_count, soa=True,
            dst_offset_bytes=first * plane_bytes,
        )