"""Reusable GPU exclusive-scan service.

Owns scan kernels + scan scratch and records into the caller's active
``CommandBatch`` when one exists. It has no dependency on Pipeline, rasterization,
or densification; those components supply their input/output buffers explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import slangpy as spy

from trainer.buffers import SCAN_BLOCK_SIZE, ScanResources
from trainer.gpu import CommandBatch, GpuRuntime


@dataclass(frozen=True)
class ScanKernels:
    scan_block: Any
    scan_block_sums: Any
    add_block_offsets: Any


class GpuScan:
    def __init__(
        self,
        resources: ScanResources,
        kernels: ScanKernels,
        runtime: GpuRuntime,
        active_batch: Callable[[], CommandBatch | None],
    ) -> None:
        self.resources = resources
        self.kernels = kernels
        self.runtime = runtime
        self._active_batch = active_batch

    def _dispatch(self, kernel, marker: str, **kwargs) -> None:
        batch = self._active_batch()
        if batch is None:
            transient = self.runtime.command_batch(marker)
            transient.dispatch(kernel, marker, **kwargs)
            self.runtime.submit(transient)
        else:
            batch.dispatch(kernel, marker, **kwargs)

    def scan(
        self,
        n: int,
        in_buf: spy.NDBuffer,
        out_buf: spy.NDBuffer | None = None,
        total_buf: spy.NDBuffer | None = None,
    ) -> None:
        """Exclusive scan ``in_buf`` -> ``out_buf`` and write total to [0].

        Supports up to ``SCAN_BLOCK_SIZE**3`` elements through three levels.
        The optional default work buffer exists in the trainer allocation but
        not the viewer allocation; viewer callers therefore pass ``out_buf``.
        """
        if out_buf is None:
            out_buf = self.resources.work
        if out_buf is None:
            raise ValueError("scan out_buf is required when no work buffer is allocated")
        total_buf = total_buf or self.resources.total

        num_blocks = max(1, (n + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE)
        self._dispatch(
            self.kernels.scan_block, "scan_block",
            thread_count=[num_blocks * SCAN_BLOCK_SIZE, 1, 1],
            in_buf=in_buf.storage,
            out_buf=out_buf.storage,
            block_sums=self.resources.block_sums.storage,
            n=n,
        )

        if num_blocks <= SCAN_BLOCK_SIZE:
            self._dispatch(
                self.kernels.scan_block_sums, "scan_block_sums",
                thread_count=[SCAN_BLOCK_SIZE, 1, 1],
                block_sums=self.resources.block_sums.storage,
                total_out=total_buf.storage,
                num_blocks=num_blocks,
            )
        else:
            num_l2_blocks = max(
                1, (num_blocks + SCAN_BLOCK_SIZE - 1) // SCAN_BLOCK_SIZE)
            self._dispatch(
                self.kernels.scan_block, "scan_block",
                thread_count=[num_l2_blocks * SCAN_BLOCK_SIZE, 1, 1],
                in_buf=self.resources.block_sums.storage,
                out_buf=self.resources.block_sums.storage,
                block_sums=self.resources.block_sums_l2.storage,
                n=num_blocks,
            )
            self._dispatch(
                self.kernels.scan_block_sums, "scan_block_sums",
                thread_count=[SCAN_BLOCK_SIZE, 1, 1],
                block_sums=self.resources.block_sums_l2.storage,
                total_out=total_buf.storage,
                num_blocks=num_l2_blocks,
            )
            self._dispatch(
                self.kernels.add_block_offsets, "add_block_offsets",
                thread_count=[num_l2_blocks * SCAN_BLOCK_SIZE, 1, 1],
                out_buf=self.resources.block_sums.storage,
                block_sums=self.resources.block_sums_l2.storage,
                n=num_blocks,
            )

        self._dispatch(
            self.kernels.add_block_offsets, "add_block_offsets",
            thread_count=[num_blocks * SCAN_BLOCK_SIZE, 1, 1],
            out_buf=out_buf.storage,
            block_sums=self.resources.block_sums.storage,
            n=n,
        )
