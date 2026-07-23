"""Rasterization forward -> backward handoff.

The tile-binned forward leaves its outputs in stable pipeline buffers EXCEPT the
radix-sort winner, which lands on whichever ping-pong slot the final sort pass
produced. That was stashed in a hidden ``Pipeline._sorted_values`` field that the
backward (and importance scoring) read implicitly. ``RasterizationContext`` makes
that dependency EXPLICIT: the forward returns it and the backward takes it — a
value passed between the two rather than hidden mutable state.

This is the seam the full ``PrimitiveRasterizer`` extraction builds on: once the
forward/backward dispatch bodies move into a dedicated rasterizer, they will
exchange exactly this context (plus a supplied gradient view) instead of reaching
into ``Pipeline`` internals.
"""

from __future__ import annotations

from dataclasses import dataclass

import slangpy as spy


@dataclass(frozen=True)
class RasterizationContext:
    """The non-stable forward output the backward must consume.

    Today that is just the sorted (tile, depth) -> primitive-id buffer — the
    radix ping-pong winner — since every other backward input (blend data, tile
    ranges, bucket offsets, checkpoints, per-pixel counts, the rendered image)
    lives in a stable buffer the backward reads directly.
    """
    sorted_ids: spy.NDBuffer
