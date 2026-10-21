# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Full-training A/B of the densification optimization, without reverting files.

The reference uses the current renderer, optimizer, sampling, data, and frontend.
Its pre-optimization densify/controller/split/Mip/copy paths are restored from
isolated snapshots; SH uses the full-sized arena those old scatters require.
Setup, quality checks, and
exports are outside the timing. Run variants in separate processes, in ABBA
order, with the same scene, seed, and settings.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import slangpy as spy

_SCRIPTS = Path(__file__).resolve().parents[1]
for _sub in ("running", "debug", "plots"):
    sys.path.append(str(_SCRIPTS / _sub))

from densify_reference import densify as reference_densify
from densify_reference.controller import DensificationController as ReferenceController
from trainer.backend.compute.device import SHADERS_DIR
from trainer.backend.memory.buffers import ScratchArena
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.common.metrics import psnr
from trainer.frontend import app as training_app
from trainer.frontend.cli import build_parser
from trainer.frontend.densification import DensificationController


REFERENCE_REVISION = "1a9045655b9efb0691d24cf71b55ece8a2e6eab6"
REFERENCE_DIR = Path(__file__).resolve().parent / "densify_reference"


class ReferencePipeline(TrainingPipeline):
    """Restore only the old densification implementation on current resources."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        entries = (
            ("decide_actions_structure", "decide_actions_structure", "decideActionsStructure", "densify"),
            ("split_anisotropic", "split_anisotropic", "splitAnisotropic", "densify"),
            ("split_anisotropic_sh", "split_anisotropic", "splitAnisotropicSH", "densify"),
            ("compute_3d_filter", "compute_3d_filter", "computeMip3DFilter", "mip"),
        )
        modules = {}
        for attr, name, entry, folder in entries:
            if name not in modules:
                modules[name] = self.device.load_module_from_source(
                    f"training_reference_{name}",
                    (REFERENCE_DIR / f"{name}.slang").read_text(encoding="utf-8"),
                    SHADERS_DIR / folder / f"training_reference_{name}.slang")
            module = modules[name]
            program = self.device.link_program([module], [module.entry_point(entry)])
            setattr(self, f"k_{attr}", self.device.create_compute_kernel(program))

        if self.buffers.sh_enabled:
            # Snapshot code scatters every moment plane at once. Keep that
            # reference isolated: production uses bounded plane batches.
            resources = self.buffers._require_densification()
            arena = resources.scratch_arena
            nbytes = max(arena.capacity_bytes, self.buffers.adam_rest_state.storage.size)
            resources.scratch_arena = ScratchArena(
                buffer=spy.NDBuffer(device=self.device, dtype="uint", shape=(nbytes // 4,)),
                capacity_bytes=nbytes, copy_dispatch_args=arena.copy_dispatch_args)
            for attr, path, entry in (
                ("scatter_compact_adam_rest", "scatter_compact.slang", "scatterCompactAdamRest"),
                ("morton_scatter_adam_rest", "morton_sort.slang", "mortonScatterAdamRest"),
            ):
                program = self.device.load_program(str(SHADERS_DIR / "densify" / path), [entry])
                setattr(self, f"k_{attr}", self.device.create_compute_kernel(program))

    def copy_arena_back(self, arena, dst, nbytes: int, *, count_buf=None,
                        soa: bool = False, dst_offset_bytes: int = 0) -> None:
        # Both SAD and the current filler service commit through this contract.
        # Ignoring the added count restores the old full-capacity byte copy.
        assert self._batch is not None
        arena.copy_back(self._batch.encoder, dst, nbytes,
                dst_offset_bytes=dst_offset_bytes)

    def densify_structure_aware(self, cfg) -> dict:
        return reference_densify.densify_structure_aware(self, cfg)

    def morton_sort(self, world_min, world_max, carry_eta=False) -> None:
        return reference_densify.morton_sort(self, world_min, world_max, carry_eta)

    def compute_3d_filter(self) -> None:
        if self._mip_cameras_buffer is None:
            raise RuntimeError("Mip cameras must be configured")
        b = self.buffers
        self.dispatch(self.k_compute_3d_filter, "compute_3d_filter",
            thread_count=[self.max_splats * 32, 1, 1],
            gaussians=b.gaussians.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            cameras=self._mip_cameras_buffer, filter_3D=b.filter_3D.storage,
            num_cameras=self._mip_num_cameras, filter_coef=self._mip_filter_coef,
            max_splats=self.max_splats)


class _TickAccounting:
    """Cheap host accounting; no timing queries or extra GPU waits."""

    def run(self, it, events, count):
        started = time.perf_counter()
        outcome = super().run(it, events, count)
        if not hasattr(self, "benchmark_ticks"):
            self.benchmark_ticks = []
        self.benchmark_ticks.append({
            "iter": it, "kind": outcome.kind,
            "host_ms": (time.perf_counter() - started) * 1000,
            "overbudget": bool(outcome.stats.get("overbudget", False)),
        })
        return outcome


class MeasuredController(_TickAccounting, DensificationController):
    pass


class MeasuredReferenceController(_TickAccounting, ReferenceController):
    pass


@contextlib.contextmanager
def benchmark_implementation(variant: str):
    """Process-local factory selection; existing workspace files never change."""
    pipeline_class = training_app.TrainingPipeline
    controller_class = training_app.DensificationController
    try:
        if variant == "reference":
            training_app.TrainingPipeline = ReferencePipeline
            training_app.DensificationController = MeasuredReferenceController
        else:
            training_app.DensificationController = MeasuredController
        yield
    finally:
        training_app.TrainingPipeline = pipeline_class
        training_app.DensificationController = controller_class


def gpu_snapshot() -> dict:
    fields = ("memory.total", "memory.used", "utilization.gpu",
              "clocks.current.graphics", "clocks.current.memory")
    try:
        response = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True)
        rows = [line.split(",") for line in response.stdout.strip().splitlines()]
        return {"devices": [dict(zip(fields, (v.strip() for v in row))) for row in rows]}
    except (OSError, subprocess.CalledProcessError):
        return {}


def source_fingerprint() -> dict:
    root = SHADERS_DIR.parent
    paths = [
        "src/trainer/backend/render/pipeline.py", "src/trainer/backend/render/kernels.py",
        "src/trainer/backend/render/stbn.py", "src/trainer/frontend/app.py",
        "src/trainer/frontend/densification.py", "src/trainer/frontend/settings.py",
        "src/trainer/frontend/cli.py", "src/trainer/backend/densify/densify.py",
        "src/trainer/backend/densify/simple_densifier.py", "src/trainer/backend/memory/buffers.py",
        "src/trainer/backend/memory/reorder.py", "shaders/densify/scatter_compact.slang",
        "shaders/densify/morton_sort.slang",
        "shaders/rasterize/tile_binned/backward_persplat.slang",
        "shaders/densify/decide_actions_structure.slang", "shaders/densify/split_anisotropic.slang",
        "shaders/densify/copy_active.slang", "shaders/mip/compute_3d_filter.slang",
    ]
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}


def sample_quality(application, views: int) -> dict:
    pipeline = application.pipeline
    count = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    result = {"n_gaussians": count, "views": []}
    if views <= 0:
        return result
    if application.args.mip_splatting:
        pipeline.compute_3d_filter()
    scene = application.scene_context
    available = scene.test_indices or scene.train_indices
    indices = np.linspace(0, len(available) - 1, min(views, len(available)), dtype=int)
    for at in indices:
        camera_index = available[int(at)]
        pipeline.render(scene.colmapScene.images[camera_index].camera)
        image = pipeline.read_output_image()[..., :3]
        if not np.all(np.isfinite(image)):
            raise RuntimeError("non-finite final render")
        target = application.targets.srgb[camera_index][..., :3]
        result["views"].append({
            "camera_index": camera_index,
            "psnr_srgb": float(psnr(np.clip(image, 0, 1), target)),
        })
    result["mean_sample_psnr_srgb"] = float(np.mean([v["psnr_srgb"] for v in result["views"]]))
    return result


def run(args: argparse.Namespace) -> dict:
    # Observe an ordinary complete training loop, not a profiled/frozen step.
    args.no_viewer = True
    args.hub = False
    args.wait_for_viewer = False
    args.profile_run = False
    args.capture_external = False
    args.profile_every = 0
    args.stage_timing_every = 0
    args.preview_every = args.log_every = args.eval_every = args.render_all_every = 0
    args.eval_iters = None
    args.evaluation = False
    args.psnr_curve = False
    args.densify_diag = False
    args.write_ground_truths = False
    args.compute_final_metrics = False
    args.final_test_psnr = args.run_report = False
    before_sources = source_fingerprint()
    external_gpu_before = gpu_snapshot()
    application = None
    with benchmark_implementation(args.bench_variant):
        try:
            application = training_app.TrainingApplication(args)
            setup_started = time.perf_counter()
            application.prepare_training_context()
            application.pipeline.runtime.wait_for_idle()
            setup_s = time.perf_counter() - setup_started
            print(f"[training-bench] {args.bench_variant}: setup complete; "
                  f"starting {args.iters} iterations", flush=True)
            loop = application.build_training_loop()
            loop_started = time.perf_counter()
            trained = loop.launch_training_loop()
            completed_wall_s = time.perf_counter() - loop_started
            if trained.iterations != args.iters:
                raise RuntimeError("benchmark training ended early")
            quality = sample_quality(application, args.bench_quality_views)
            after_sources = source_fingerprint()
            if before_sources != after_sources:
                raise RuntimeError("production sources changed during the benchmark")
            result = {
                "variant": args.bench_variant, "reference_revision": REFERENCE_REVISION,
                "scene": Path(args.scene).name, "backend": args.backend,
                "iters": trained.iterations, "seed": args.seed,
                "sh_degree": args.sh_degree, "capacity": args.max_splats,
                "resolution": [application.scene_context.width, application.scene_context.height],
                "camera_count": application.scene_context.camera_count,
                "stbn_pixel_fraction": args.stbn_pixel_fraction,
                "stbn_source": args.stbn_source, "stbn_adaptive": args.stbn_adaptive,
                "settings": {name: getattr(args, name) for name in (
                    "downsample", "sample_mode", "densify_mode", "densify_from", "densify_to",
                    "densify_every", "sad_structure_every", "sad_accum_every", "sad_tau_split",
                    "sad_tau_prune", "sad_grad_threshold", "sad_split_power", "sad_max_split_factor",
                    "sad_ks_scale_power", "sad_structure_min_opacity", "sad_opacity_cap",
                    "sad_relative_reset", "sad_reset_decay", "grad_thresh", "grow_scale3d",
                    "reset_every", "mip_splatting", "mip_kernel_size", "morton_every",
                    "cluster_cull", "flat_bucket_backward", "backward_pixel_stride",
                    "stbn_select_mode", "stbn_independent", "stbn_warmup_iters", "eval_holdout",
                    "eval_holdout_stride", "stage_timing_every", "profile_run", "log_every",
                    "preview_every", "eval_every", "no_viewer")},
                "train_wall_s": trained.wall_time_seconds,
                "train_active_s": trained.wall_time_seconds - trained.observation_seconds,
                "cpu_observation_s": trained.observation_seconds,
                "completed_loop_wall_s": completed_wall_s,
                "setup_s_excluded": setup_s, "quality_outside_timing": quality,
                "densification_ticks": application.runtime.densifier.benchmark_ticks,
                "source_fingerprint": before_sources,
                "gpu_before_setup": external_gpu_before,
                "gpu_after_training": gpu_snapshot(),
                "timing_note": "Complete training loop including scheduled densification, Morton and Mip; "
                               "setup, final quality and exports excluded. Host tick times include only "
                               "recording and existing waits, not extra GPU timing synchronization.",
            }
            Path(args.bench_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print("[training-bench-result] " + json.dumps({
                "variant": result["variant"], "scene": result["scene"],
                "train_wall_s": result["train_wall_s"], "completed_loop_wall_s": completed_wall_s,
                "n_gaussians": quality["n_gaussians"],
                "sample_psnr": quality.get("mean_sample_psnr_srgb"),
                "overbudget_ticks": sum(t["overbudget"] for t in result["densification_ticks"]),
            }), flush=True)
            return result
        finally:
            if application is not None:
                if application.runtime is not None:
                    application.runtime.artifacts.close_all()
                if application.device is not None:
                    application.device.close()


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    parser.add_argument("--bench-variant", choices=("reference", "optimized"), required=True)
    parser.add_argument("--bench-json", required=True)
    parser.add_argument("--bench-quality-views", type=int, default=3)
    args = parser.parse_args()
    if args.bench_quality_views < 0:
        parser.error("bench-quality-views must be nonnegative")
    path = Path(args.bench_json)
    if path.exists():
        parser.error("benchmark result already exists; choose a new name")
    path.parent.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    run(parse_args())