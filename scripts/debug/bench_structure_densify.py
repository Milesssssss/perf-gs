# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Dataset-free GPU benchmark for SAD scoring, densification, and Morton sort.

Each sample restores the same device-local population before timing. Uploads,
restoration, and shader warmup are excluded; GPU timestamps separate kernels
from arena copies, while wall time includes the production readback overhead.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from functools import partial

import numpy as np
import slangpy as spy

from trainer.backend.compute.device import SHADERS_DIR, create_device
from trainer.backend.compute.timing import StepTimer
from trainer.backend.densify.densify import StructureConfig, accumulate_structure_scores
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.config.config import AppearanceConfig
from trainer.config.config_types import DensificationMode
from trainer.data.dataset import simple_camera


class TimedPipeline(TrainingPipeline):
    """Add per-dispatch timestamps locally, without changing production timing."""

    def _dispatch(self, kernel, marker: str, **kwargs) -> None:
        with self.marker_group(marker):
            super()._dispatch(kernel, marker, **kwargs)

    def _dispatch_indirect(self, kernel, marker: str, args_buf,
                          offset: int = 0, **kwargs) -> None:
        with self.marker_group(marker):
            super()._dispatch_indirect(kernel, marker, args_buf, offset, **kwargs)

    def copy_arena_back(self, arena, dst, nbytes: int, **kwargs) -> None:
        with self.marker_group("arena_copy"):
            super().copy_arena_back(arena, dst, nbytes, **kwargs)


class FrozenPopulation:
    """Restore mutated primitive and tracking state entirely on the GPU."""

    def __init__(self, pipeline: TrainingPipeline) -> None:
        self.pipeline = pipeline
        b = pipeline.buffers
        names = ["gaussians", "adam_state", "num_gaussians_counter",
                 "grad_mean_2d_mag_accum", "visibility_count_accum",
                 "tile_count_per_gaussian", "eta_n_total", "valid_mask_buffer", "scan_total",
                 "densify_pre_count", "action_buffer", "split_factor_x", "split_factor_y", "split_factor_z"]
        if b.sh_enabled:
            names.extend(["features_rest", "adam_rest_state"])
        resources = [partial(getattr, b, name) for name in names]
        resources.extend(partial(_eta_component, b, name) for name in b.eta_buffers)
        self.copies = []
        batch = pipeline.runtime.command_batch("benchmark_freeze")
        for index, resource in enumerate(resources):
            storage = resource().storage
            frozen = pipeline.device.create_buffer(
                size=storage.size,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                label=f"benchmark_frozen_{index}")
            batch.encoder.copy_buffer(frozen, 0, storage, 0, storage.size)
            self.copies.append((resource, frozen))
        pipeline.runtime.submit(batch)
        pipeline.runtime.wait_for_idle()

    def restore(self) -> None:
        batch = self.pipeline.runtime.command_batch("benchmark_restore")
        for resource, frozen in self.copies:
            storage = resource().storage
            batch.encoder.copy_buffer(storage, 0, frozen, 0, storage.size)
        self.pipeline.runtime.submit(batch)
        self.pipeline.runtime.wait_for_idle()


def _eta_component(buffers, name):
    return buffers.eta_buffers[name]


def initial_population(n: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    means = rng.uniform(-0.5, 0.5, (n, 3)).astype(np.float32)
    means[:, 2] += 3.0
    scales = np.full((n, 3), 0.05, dtype=np.float32)
    scales[1::10] = 0.005  # Triggered small gaussians clone instead of splitting.
    return {
        "means": means,
        "log_scales": np.log(scales),
        "rot_quats": np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        "opacities": np.full(n, 0.7, dtype=np.float32),
        "color_logits": np.zeros((n, 3), dtype=np.float32),
    }


def seed_decisions(pipeline: TrainingPipeline, n: int, factor: int) -> None:
    """10% split, 10% clone, 10% prune, and 70% keep; deterministic votes."""
    b = pipeline.buffers
    capacity = pipeline.max_splats
    total = np.zeros(capacity, np.uint32)
    total[:n] = 10
    b.eta_n_total.copy_from_numpy(total)
    high = np.zeros(capacity, np.uint32)
    high[:n:10] = high[1:n:10] = 10
    b.eta_buffers["n_high"].copy_from_numpy(high)
    low = np.zeros(capacity, np.uint32)
    low[2:n:10] = 10
    b.eta_buffers["n_low"].copy_from_numpy(low)
    eta = np.zeros(capacity, np.uint32)
    eta[:n:10] = eta[1:n:10] = factor * factor * 65536
    for axis in ("max_x", "max_y", "max_z"):
        b.eta_buffers[axis].copy_from_numpy(eta)
    visible = np.zeros(capacity, np.uint32)
    visible[:n] = 1
    b.visibility_count_accum.copy_from_numpy(visible)
    gradients = np.zeros(capacity, np.int64)
    gradients[:n] = 65536
    b.grad_mean_2d_mag_accum.copy_from_numpy(gradients)
    b.tile_count_per_gaussian.copy_from_numpy(visible)
    actions = np.zeros(capacity, np.uint32)
    actions[:n:10], actions[1:n:10], actions[2:n:10] = 1, 2, 3
    b.action_buffer.copy_from_numpy(actions)
    factors = np.ones(capacity, np.uint32)
    factors[:n:10] = factor
    for buffer in (b.split_factor_x, b.split_factor_y, b.split_factor_z):
        buffer.copy_from_numpy(factors)
    selected = np.flatnonzero((actions == 1) | (actions == 2)).astype(np.uint32)
    parents = np.zeros(capacity, np.uint32)
    parents[:len(selected)] = selected
    b.valid_mask_buffer.copy_from_numpy(parents)
    b.scan_total.copy_from_numpy(np.array([len(selected)], np.uint32))
    b.densify_pre_count.copy_from_numpy(np.array([n], np.uint32))


def sample_operation(pipeline: TimedPipeline, frozen: FrozenPopulation,
                     timer: StepTimer, operation) -> dict:
    frozen.restore()
    timer.begin_step()
    pipeline._step_timer = timer
    started = time.perf_counter()
    try:
        operation()
        pipeline.runtime.wait_for_idle()
        wall_ms = (time.perf_counter() - started) * 1000
        return {"wall_ms": wall_ms, "markers_ms": timer.read().markers_ms}
    finally:
        pipeline._step_timer = None


def summarize(samples: list[dict]) -> dict:
    markers = defaultdict(list)
    for sample in samples:
        for label, duration in sample["markers_ms"].items():
            markers[label].append(duration)
    wall = [sample["wall_ms"] for sample in samples]
    return {
        "wall_median_ms": float(np.median(wall)),
        "wall_min_ms": float(np.min(wall)),
        "wall_p90_ms": float(np.percentile(wall, 90)),
        "gpu_median_ms": {label: float(np.median(values))
                          for label, values in markers.items()},
    }


def score_view(pipeline: TrainingPipeline, structure) -> None:
    with pipeline.recording("benchmark_score"):
        accumulate_structure_scores(pipeline, structure, pass_idx=1)


def split_only(pipeline: TrainingPipeline, cfg: StructureConfig, reference=None) -> None:
    b = pipeline.buffers
    params = dict(
        gaussians=b.gaussians.storage, adam_state=b.adam_state.storage,
        num_gaussians_counter=b.num_gaussians_counter.storage,
        action=b.action_buffer.storage, split_factor_x=b.split_factor_x.storage,
        split_factor_y=b.split_factor_y.storage, split_factor_z=b.split_factor_z.storage,
        pre_count_buf=b.densify_pre_count.storage, ks_scale_power=cfg.ks_scale_power,
        max_splats=pipeline.max_splats)
    if b.sh_enabled:
        params.update(features_rest=b.features_rest.storage, adam_rest=b.adam_rest_state.storage,
                      rest_floats=b.sh.rest_floats)
    with pipeline.recording("split_probe"), pipeline.marker_group("split_probe"):
        pipeline.dispatch(pipeline.k_densify_prepare_args if reference else
                          pipeline.k_structure_split_prepare_args, "prepare_split",
                          thread_count=[1, 1, 1],
                          count_buf=b.densify_pre_count.storage if reference else b.scan_total.storage,
                          dispatch_args=b.densify_dispatch_args.storage, max_splats=pipeline.max_splats)
        if reference is None:
            params.update(selected_parents=b.valid_mask_buffer.storage,
                          selected_count=b.scan_total.storage)
        kernel = reference if reference is not None else (
            pipeline.k_split_anisotropic_sh if b.sh_enabled else pipeline.k_split_anisotropic)
        pipeline.dispatch_indirect(kernel, "split_kernel", b.densify_dispatch_args, **params)


def _canonical_children(pipeline: TrainingPipeline) -> np.ndarray:
    count = int(pipeline.buffers.num_gaussians_counter.to_numpy()[0])
    from trainer.contracts.codegen import GAUSSIAN_FLOAT_COUNT
    g = pipeline.buffers.gaussians.to_numpy().view(np.float32).reshape(-1, GAUSSIAN_FLOAT_COUNT)[:count]
    # Tail reservations are intentionally unordered on both implementations.
    return g[np.lexsort((g[:, 2], g[:, 1], g[:, 0]))].copy()


def run_benchmark(args: argparse.Namespace) -> dict:
    device = create_device(backend=args.backend)
    pipeline = TimedPipeline(
        device=device, width=64, height=64, max_splats=args.capacity,
        max_sort_pairs=max(args.capacity, args.live * 16),
        appearance_config=AppearanceConfig(sh_degree=args.sh_degree),
        initial_state=initial_population(args.live),
        densify_mode=DensificationMode.STRUCTURE_AWARE)
    camera = simple_camera(width=64, height=64, eye=(0, 0, 0),
                           target=(0, 0, 1), up=(0, 1, 0))
    pipeline.rasterize(camera)
    seed_decisions(pipeline, args.live, args.split_factor)
    field = np.zeros((64, 64, 4), np.float32)
    field[..., 0] = 0.25
    structure = device.create_texture(
        type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
        width=64, height=64, usage=spy.TextureUsage.shader_resource |
        spy.TextureUsage.unordered_access, label="benchmark_structure")
    structure.copy_from_numpy(field)
    frozen = FrozenPopulation(pipeline)
    timer = StepTimer(device, pool_size=512)
    cfg = StructureConfig(dense_scale=0.01, max_split_factor=args.split_factor)
    operations = {
        "score": partial(score_view, pipeline, structure),
        "structure": partial(pipeline.densify_structure_aware, cfg),
        "morton": partial(pipeline.morton_sort,
                          np.array([-1, -1, 2], np.float32),
                          np.array([1, 1, 4], np.float32), carry_eta=True),
    }
    parity = None
    if args.compare_reference:
        source = subprocess.run(
            ["git", "show", "HEAD:shaders/densify/split_anisotropic.slang"],
            cwd=SHADERS_DIR.parent, check=True, capture_output=True, text=True).stdout
        module = device.load_module_from_source("benchmark_split_reference", source,
                                                SHADERS_DIR / "densify/benchmark_split_reference.slang")
        entry = "splitAnisotropicSH" if pipeline.buffers.sh_enabled else "splitAnisotropic"
        reference = device.create_compute_kernel(device.link_program([module], [module.entry_point(entry)]))
        operations["split_optimized"] = partial(split_only, pipeline, cfg)
        operations["split_reference"] = partial(split_only, pipeline, cfg, reference)
        children = {}
        for name in ("split_optimized", "split_reference"):
            frozen.restore()
            operations[name]()
            pipeline.runtime.wait_for_idle()
            children[name] = _canonical_children(pipeline)
        np.testing.assert_allclose(children["split_optimized"], children["split_reference"],
                                   rtol=2e-6, atol=2e-6)
        parity = True
    results = {}
    samples = {name: [] for name in operations}
    for iteration in range(args.warmup + args.repeats):
        for name, operation in operations.items():
            sample = sample_operation(pipeline, frozen, timer, operation)
            if iteration >= args.warmup:
                samples[name].append(sample)
    for name, values in samples.items():
        results[name] = summarize(values)
        print(f"{name}: {results[name]['wall_median_ms']:.3f} ms wall", flush=True)
    device.close()
    return {
        "backend": args.backend, "capacity": args.capacity, "live": args.live,
        "sh_degree": args.sh_degree, "split_factor": args.split_factor,
        "warmup": args.warmup, "repeats": args.repeats, "results": results,
        "reference_geometry_matches": parity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("vulkan", "d3d12"), default="vulkan")
    parser.add_argument("--capacity", type=int, default=250_000)
    parser.add_argument("--live", type=int, default=25_000)
    parser.add_argument("--sh-degree", type=int, choices=range(4), default=3)
    parser.add_argument("--split-factor", type=int, choices=range(2, 9), default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--compare-reference", action="store_true",
                        help="A/B the split against its committed serial kernel")
    args = parser.parse_args()
    if not 0 < args.live <= args.capacity:
        parser.error("live must be positive and no larger than capacity")
    if args.warmup < 1 or args.repeats < 1:
        parser.error("warmup and repeats must be positive")
    splits = (args.live + 9) // 10
    clones = (args.live + 8) // 10
    if args.live + splits * (args.split_factor ** 3 - 1) + clones > args.capacity:
        parser.error("population would exceed capacity; lower live or split-factor")
    return args


def main() -> None:
    print(json.dumps(run_benchmark(parse_args()), indent=2), flush=True)


if __name__ == "__main__":
    main()