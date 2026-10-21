# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Interleaved full-arena versus bounded-plane SH moment permutation benchmark.

No dataset or optimizer changes. Both variants use identical FP32/uint input,
permutations, and live counts; only their scratch batch size differs. Restoring
input and checking output happen outside timing. Both arenas coexist for this
diagnostic, so their reported sizes are not its combined process residency.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import slangpy as spy

from trainer.backend.compute.device import create_device
from trainer.backend.compute.timing import StepTimer
from trainer.backend.memory.buffers import ScratchArena
from trainer.backend.memory.reorder import reorder_soa_planes
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.config.config_types import DensificationMode
from trainer.config.config import AppearanceConfig


def _reference_arena(pipeline):
    current = pipeline.buffers.scratch_arena
    size = pipeline.buffers.adam_rest_state.storage.size
    return ScratchArena(
        buffer=spy.NDBuffer(device=pipeline.device, dtype="uint", shape=(size // 4,)),
        capacity_bytes=size, copy_dispatch_args=current.copy_dispatch_args)


def run(args):
    device = create_device(backend=args.backend)
    pipeline = TrainingPipeline(densify_mode=DensificationMode.STRUCTURE_AWARE,
        device=device, width=16, height=16, max_splats=args.capacity,
        max_sort_pairs=args.capacity,
        appearance_config=AppearanceConfig(sh_degree=3))
    b, runtime = pipeline.buffers, pipeline.runtime
    arenas = {"full": _reference_arena(pipeline), "bounded": b.scratch_arena}
    rng = np.random.default_rng(2026)
    original = rng.integers(0, 2**32, (90, args.capacity), dtype=np.uint32)
    b.adam_rest_state.copy_from_numpy(original.ravel())
    frozen = device.create_buffer(
        size=original.nbytes, usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access, label="benchmark_original_moments")
    batch = runtime.command_batch("freeze_moments")
    batch.encoder.copy_buffer(frozen, 0, b.adam_rest_state.storage, 0, original.nbytes)
    runtime.wait_for_submit(runtime.submit(batch))
    timer = StepTimer(device, pool_size=128)
    results = {}

    for kind in ("compaction", "morton"):
        b.scan_total.copy_from_numpy(np.array([args.live], np.uint32))
        if kind == "compaction":
            selected = np.sort(rng.choice(args.capacity, args.live, replace=False))
            mask = np.zeros(args.capacity, np.uint32)
            mask[selected] = 1
            b.valid_mask_buffer.copy_from_numpy(mask)
            b.scan_buffer.copy_from_numpy(np.cumsum(mask, dtype=np.uint32) - mask)
            kernel = pipeline.k_scatter_compact_adam_rest
            permutation = dict(valid_mask=b.valid_mask_buffer.storage, scan=b.scan_buffer.storage)
        else:
            selected = rng.permutation(args.live)
            indices = np.zeros(args.capacity, np.uint32)
            indices[:args.live] = selected
            b.sort_values_a.copy_from_numpy(indices)
            kernel = pipeline.k_morton_scatter_adam_rest
            permutation = dict(sorted_values=b.sort_values_a.storage, count_buf=b.scan_total.storage)
        expected = original.copy()
        expected[:, :args.live] = original[:, selected]
        samples = defaultdict(list)

        for repeat in range(args.warmup + args.repeats):
            # Alternating AB/BA avoids always giving either batch size the same
            # position relative to unrelated system load, cache, or clock drift.
            order = ("full", "bounded") if repeat % 2 == 0 else ("bounded", "full")
            for variant in order:
                restore = runtime.command_batch("restore_moments")
                restore.encoder.copy_buffer(b.adam_rest_state.storage, 0, frozen, 0, original.nbytes)
                runtime.wait_for_submit(runtime.submit(restore))
                timer.begin_step()
                pipeline._step_timer = timer
                started = time.perf_counter()
                try:
                    with pipeline.recording("benchmark_reorder"), pipeline.marker_group("reorder"):
                        reorder_soa_planes(
                            commands=pipeline, arena=arenas[variant], buffer=b.adam_rest_state,
                            kernel=kernel, marker=kind, plane_count=90,
                            capacity=args.capacity, live_count=b.scan_total, **permutation)
                    runtime.wait_for_idle()
                    wall_ms = (time.perf_counter() - started) * 1e3
                    gpu_ms = timer.read().markers_ms["reorder"]
                finally:
                    pipeline._step_timer = None
                if repeat == 0:
                    np.testing.assert_array_equal(
                        b.adam_rest_state.to_numpy().view(np.uint32), expected.ravel())
                if repeat >= args.warmup:
                    samples[variant].append((wall_ms, gpu_ms))
        results[kind] = {
            variant: {"wall_median_ms": float(np.median(values, axis=0)[0]),
                      "gpu_median_ms": float(np.median(values, axis=0)[1]),
                      "wall_min_ms": float(np.min(values, axis=0)[0]),
                      "wall_max_ms": float(np.max(values, axis=0)[0])}
            for variant, values in samples.items()
        }
    report = {
        "backend": args.backend, "adapter": device.info.adapter_name,
        "capacity": args.capacity, "live": args.live,
        "scratch_bytes": {name: arena.capacity_bytes for name, arena in arenas.items()},
        "repeats": args.repeats, "warmup": args.warmup,
        "bit_exact": True, "results": results,
        "scope": "isolated moment scatter + live-prefix commits, not full training",
    }
    device.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("vulkan", "d3d12"), default="vulkan")
    parser.add_argument("--capacity", type=int, default=250_000)
    parser.add_argument("--live", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if not 0 < args.live <= args.capacity or args.warmup < 1 or args.repeats < 1:
        parser.error("require 0 < live <= capacity and positive warmup/repeats")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
