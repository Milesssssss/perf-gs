# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Measure the post-densify Mip filter at realistic counts without a dataset.

Only primitive/filter/camera resources are allocated, so a 10M-slot capacity
can be measured without allocating a second complete training pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import slangpy as spy

from bench_structure_densify import TimedPipeline, summarize
from trainer.backend.compute.device import SHADERS_DIR, create_device
from trainer.backend.compute.gpu import GpuRuntime
from trainer.backend.compute.timing import StepTimer
from trainer.backend.memory.buffers import GaussianBuffers, PrimitiveStorage
from trainer.backend.render.kernels import KERNEL_SPECS
from trainer.contracts.codegen import GAUSSIAN_FLOAT_COUNT
from trainer.data.dataset import simple_camera


class MipFilterProbe(TimedPipeline):
    """Exercise the production filter API with only its resource dependencies."""

    def __init__(self, device, capacity: int, live: int) -> None:
        self.device = device
        self.runtime = GpuRuntime(device)
        self.max_splats = capacity
        self._batch = None
        self._step_timer = None
        self.last_submit_id = 0
        module = spy.Module.load_from_file(device, str(SHADERS_DIR / "common/types.slang"))
        gaussians = spy.NDBuffer(device=device,
                                dtype=module.layout.find_type_by_name("Gaussian3D"),
                                shape=(capacity,))
        count = spy.NDBuffer(device=device, dtype="uint", shape=(1,))
        filters = spy.NDBuffer(device=device, dtype="float", shape=(capacity,))
        self.buffers = GaussianBuffers(max_splats=capacity, width=64, height=64)
        self.buffers.primitive = PrimitiveStorage(gaussians, count, filters)
        self.buffers.mip_filter_dispatch_args = spy.NDBuffer(
            device=device, dtype="uint", shape=(6,),
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access |
            spy.BufferUsage.indirect_argument)
        self.buffers.mip_filter_previous_count = spy.NDBuffer(
            device=device, dtype="uint", shape=(2,))
        self.buffers.mip_filter_previous_count.copy_from_numpy(
            np.array([capacity, capacity], np.uint32))
        data = np.zeros((capacity, GAUSSIAN_FLOAT_COUNT), np.float32)
        data[:live, :3] = np.random.default_rng(42).uniform(-0.5, 0.5, (live, 3))
        data[:live, 2] += 3.0
        gaussians.storage.copy_from_numpy(data)
        count.copy_from_numpy(np.array([live], np.uint32))
        for attr, path, entry in KERNEL_SPECS:
            if attr in {"compute_3d_filter", "mip_filter_prepare_args", "mip_filter_clear_tail"}:
                program = device.load_program(str(SHADERS_DIR / path), [entry])
                setattr(self, f"k_{attr}", device.create_compute_kernel(program))


def run_benchmark(args: argparse.Namespace) -> dict:
    device = create_device(backend=args.backend)
    pipeline = MipFilterProbe(device, args.capacity, args.live)
    cameras = [simple_camera(width=64, height=64,
                             eye=(0.1 * np.sin(i), 0.1 * np.cos(i), 0),
                             target=(0, 0, 3)) for i in range(args.cameras)]
    pipeline.set_mipsplatting_cameras(cameras)
    pipeline.runtime.wait_for_idle()
    timer = StepTimer(device)
    operations = {"optimized": pipeline.compute_3d_filter}
    if args.compare_reference:
        source = subprocess.run(
            ["git", "show", "HEAD:shaders/mip/compute_3d_filter.slang"],
            cwd=SHADERS_DIR.parent, check=True, capture_output=True, text=True).stdout
        module = device.load_module_from_source(
            "benchmark_mip_reference", source,
            Path(SHADERS_DIR / "mip/benchmark_mip_reference.slang"))
        reference = device.create_compute_kernel(device.link_program(
            [module], [module.entry_point("computeMip3DFilter")]))

        def reference_filter() -> None:
            b = pipeline.buffers
            with pipeline.recording("mip_reference"), pipeline.marker_group("mip_reference"):
                pipeline.dispatch(reference, "compute_3d_filter_reference",
                    thread_count=[args.capacity * 32, 1, 1],
                    gaussians=b.gaussians.storage,
                    num_gaussians_counter=b.num_gaussians_counter.storage,
                    cameras=pipeline._mip_cameras_buffer, filter_3D=b.filter_3D.storage,
                    num_cameras=pipeline._mip_num_cameras,
                    filter_coef=pipeline._mip_filter_coef, max_splats=args.capacity)

        operations["reference"] = reference_filter

    samples = {name: [] for name in operations}
    fingerprints = {}
    for iteration in range(args.warmup + args.repeats):
        # Interleave A/B samples on the SAME buffers and device, so memory
        # residency and background GPU load cannot differ between processes.
        for name, operation in operations.items():
            timer.begin_step()
            pipeline._step_timer = timer
            started = time.perf_counter()
            operation()
            pipeline.runtime.wait_for_idle()
            wall_ms = (time.perf_counter() - started) * 1000
            if iteration >= args.warmup:
                samples[name].append({"wall_ms": wall_ms, "markers_ms": timer.read().markers_ms})
            pipeline._step_timer = None
            if args.compare_reference and iteration == args.warmup:
                fingerprints[name] = pipeline.buffers.filter_3D.to_numpy().copy()
    if fingerprints:
        np.testing.assert_array_equal(fingerprints["optimized"], fingerprints["reference"])
    result = {
        "backend": args.backend, "capacity": args.capacity, "live": args.live,
        "cameras": args.cameras, "repeats": args.repeats,
        "results": {name: summarize(values) for name, values in samples.items()},
        "reference_bit_exact": True if fingerprints else None,
    }
    device.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("vulkan", "d3d12"), default="vulkan")
    parser.add_argument("--capacity", type=int, default=10_000_000)
    parser.add_argument("--live", type=int, default=1_000_000)
    parser.add_argument("--cameras", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--compare-reference", action="store_true",
                        help="A/B against the committed shader on the same resources")
    args = parser.parse_args()
    if not 0 < args.live <= args.capacity:
        parser.error("live must be positive and no larger than capacity")
    if min(args.cameras, args.warmup, args.repeats) < 1:
        parser.error("cameras, warmup, and repeats must be positive")
    print(json.dumps(run_benchmark(args), indent=2), flush=True)


if __name__ == "__main__":
    main()