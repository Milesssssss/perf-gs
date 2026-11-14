# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Hierarchical orchestration for the v2 training application.

The file is intentionally organized by level of abstraction:

``main`` -> ``TrainingApplication`` -> ``TrainingLoop`` -> cohesive support
objects. High-level lifecycle code does not expose texture formats, cache file
keys, iteration logging schemas, or GPU setup mechanics.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from trainer.data.colmap import load_colmap, load_colmap_image
from trainer.data.parallel import default_workers_numbers, map_ordered
from trainer.common.color import srgb_to_linear
from trainer.config.config import (
    AdamLRs,
    AppearanceConfig,
    DensificationProperties,
    IterationContextualVariables,
    LossConfig,
    RasterConfig,
    RegularizationConfig,
    StbnConfig,
    StbnVariables,
    AdaptiveStbnConfig,
)
from trainer.config.config_types import (
    DensificationMode,
    ScheduleConfig,
    SplatKernel,
    validate_training_config,
)
from trainer.backend.compute.device import create_device
from trainer.backend.memory.budget import GIB, available_bytes, fit_capacity
from trainer.backend.memory.vram import query_video_memory
from trainer.backend.render import stbn as stbn_module
from trainer.common.metrics import psnr
from trainer.backend.render.pipeline import (
    TrainingPipeline,
    _module_for_reflection,
    estimate_pipeline_bytes,
)
from trainer.contracts.codegen import TILE_AREA
from trainer.frontend.cli import build_parser
from trainer.frontend.densification import DensificationController, DensifyOutcome
from trainer.frontend.evaluation.evaluation import (
    eval_test_psnr_srgb,
    render_all_cameras,
    run_eval_sweep,
)
from trainer.frontend.schedule import (
    AppearanceSchedule,
    ExtinctionHardnessSchedule,
    IterationContext,
    MeanLearningRateSchedule,
    StbnDensitySchedule,
    TrainingSchedule
)
from trainer.frontend.reporting.outputs import RunArtifacts
from trainer.frontend.reporting.error_log import ErrorLog, report_exception
from trainer.frontend.evaluation.post_training import run_post_training
from trainer.frontend.evaluation import run_report as run_report_module
from trainer.frontend.profiling.stage_timing import StageTimingRecorder
from trainer.frontend.queue_limiter import QueueLimiter
from trainer.frontend.run_control import RunStopped
from trainer.frontend.training_preflight import (
    build_densify_configs,
    compute_morton_bounds,
    resolve_holdout_split,
    resolve_loss_and_lrs,
)

# Import time, used as this process's t0 for the run report's process_wall_s. It
# trails the real process start by the import graph's cost (slangpy, cv2), so a
# suite driver that needs the authoritative total measures it from the parent —
# this is the in-process approximation, labelled as such.
_PROCESS_STARTED_AT = time.time()

from trainer.frontend.viewer.session import (
    ERROR,
    ResultInfo,
    Session,
    TrainingState,
)
from trainer.frontend.viewer.viewer_control import ViewerSession

from trainer.data.colmap import ColmapDataset

# The thin scripts/running/v2_train.py entrypoint installs scripts/running and
# scripts/debug on sys.path before importing this module.
from trainer.frontend.profiling.profiler import (
    Profiler,
    apply_profile_session_overrides,
)
from trainer.frontend.evaluation.evaluation import CameraSampler 


@dataclass(frozen=True)
class RunPaths:
    root: Path
    cameras: Path

    @classmethod
    def create(cls, run_name: str) -> "RunPaths":
        root = Path("outputs/train") / run_name
        cameras = root / "cams"
        cameras.mkdir(parents=True, exist_ok=True)
        return cls(root=root, cameras=cameras)


@dataclass(frozen=True)
class SceneContext:
    colmapScene: ColmapDataset
    width: int
    height: int
    camera_count: int
    initial_gaussian_count: int
    extent: float
    train_indices: list[int]
    test_indices: list[int]
    holdout_stride: int
    morton_min: np.ndarray
    morton_max: np.ndarray

@dataclass(frozen=True)
class TrainingResult:
    iterations: int
    wall_time_seconds: float
    psnr_curve: list[dict]
    started_at: float
    # Wall-clock spent OBSERVING the run from inside the loop (--eval-every
    # sweeps, --psnr-curve ticks, log/preview re-renders). It is contained in
    # wall_time_seconds, so subtracting it gives the active training time — the
    # only figure a sparse-vs-dense speed claim may use, since it must not
    # depend on how often quality was measured. 0.0 for an unobserved run.
    #
    # This is the IN-WINDOW subset only. wall_time_seconds is frozen right after
    # the final train_step, so the final iteration's observations fall outside it
    # already; observation_total_seconds keeps the full figure for reporting.
    observation_seconds: float = 0.0
    observation_total_seconds: float = 0.0
    # StageTimingRecorder when --stage-timing-every is set, else None.
    stage_timing: Any = None


@dataclass(frozen=True)
class ShaderSpecialization:
    """Compile-time shader policy resolved from training configuration.

    Pixel stride must remain compile-time because the backward shader uses it
    to size staged arrays and specialize loop/index arithmetic. The application
    deals in the semantic value; only this adapter knows the Slang macro name
    and unsigned-literal spelling.

    stbn_enabled turns on the backward's compacted-visit loop (STBN_ENABLED).
    The staged-array size (STBN_MAX_TILE_PIXELS) is deliberately NOT set here:
    at an 8x8 tile those arrays total under 2 KB of the ~100 KB shared memory per
    SM, so shrinking them buys nothing measurable - the speedup is the shorter
    pixel loop, which the RUNTIME budget drives.

    hardness_is_one specializes the Compositor to the pure alpha-over path when
    the run's extinction hardness is fixed at <= 1 (default): the dead pow-based
    hardened-extinction branches compile out, freeing registers in the hottest
    backward loop. Any hardness > 1 run keeps the full runtime path (define off).

    HUB EXCEPTION: defines are fixed when the DEVICE is created, and the hub creates
    its device before the user has picked the run's hyperparameters — so a hardness
    raised in the panel would hit a compositor specialized to ignore it. Hub runs
    therefore always keep the full runtime path (a register-pressure cost paid for
    the panel being able to set hardness at all). ``--backward-pixel-stride`` has no
    such escape: it sizes staged arrays, so in the hub it stays a launch-time flag.
    STBN's compacted backward is in the same position and takes the same out: a
    hub run gets the mask (the loss applies it at runtime) but not the compacted
    pixel loop, i.e. the correct sparse gradient without the speedup.

    adam_bf16 stores the Adam moments as bf16 (ADAM_MOMENTS_BF16). It changes
    buffer layouts, so in the hub it is a launch-time flag too.

    splat_kernel picks how a footprint becomes alpha (SPLAT_KERNEL) and
    splat_profile_power its radial falloff (SPLAT_PROFILE_POWER), with
    splat_profile_onset delaying opaque_core's sharpening (SPLAT_PROFILE_ONSET);
    launch-time flags for the same reason.
    """

    backward_pixel_stride: int
    # Default False = full runtime compositor (safe/back-compat). from_args sets
    # it True when the run's extinction hardness is fixed at <= 1.
    hardness_is_one: bool = False
    # Whether this build takes the compacted-visit path in the backward.
    stbn_enabled: bool = False
    # DIAGNOSTIC only (--stbn-max-fraction): shrink the backward's staged arrays
    # to this fraction of a tile. 0 = leave them at the shader default (a full
    # tile), which is what every real run wants.
    stbn_staged_fraction: float = 0.0
    adam_bf16: bool = False
    splat_kernel: SplatKernel = SplatKernel.GAUSSIAN
    splat_profile_power: float = 1.0
    splat_profile_onset: float = 0.0

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ShaderSpecialization":
        hub = bool(getattr(args, "hub", False))
        fraction = float(getattr(args, "stbn_pixel_fraction", 1.0))
        enabled = fraction < 1.0 and not hub
        staged = float(getattr(args, "stbn_max_fraction", 0.0) or 0.0)
        # A warmup ramps from full density, so a shrunken stage could not hold the
        # ramp's opening steps; ignore the diagnostic rather than truncate them.
        if int(getattr(args, "stbn_warmup_iters", 0)) > 0:
            staged = 0.0
        return cls(
            backward_pixel_stride=args.backward_pixel_stride,
            hardness_is_one=(not hub and float(args.extinction_hardness) <= 1.0),
            stbn_enabled=enabled,
            stbn_staged_fraction=(max(staged, fraction) if enabled and staged else 0.0),
            adam_bf16=getattr(args, "adam_moments", "fp32") == "bf16",
            splat_kernel=SplatKernel(getattr(args, "splat_kernel", "gaussian")),
            splat_profile_power=float(getattr(args, "splat_profile_power", 1.0)),
            splat_profile_onset=float(getattr(args, "splat_profile_onset", 0.0)),
        )

    def compiler_defines(self) -> dict[str, str] | None:
        defines: dict[str, str] = {}
        if self.backward_pixel_stride != 1:
            defines["BACKWARD_PIXEL_STRIDE"] = f"{self.backward_pixel_stride}u"
        if self.hardness_is_one:
            defines["COMPOSITOR_HARDNESS_ONE"] = "1"
        if self.stbn_enabled:
            defines["STBN_ENABLED"] = "1"
        # Otherwise STBN_MAX_TILE_PIXELS stays at its shader default (a full
        # tile) - see the class docstring for why sizing it down is not worth a
        # knob; --stbn-max-fraction exists only to A/B that claim.
        if self.stbn_staged_fraction > 0.0:
            defines["STBN_MAX_TILE_PIXELS"] = (
                f"{stbn_module.pixels_per_tile(self.stbn_staged_fraction, TILE_AREA)}u")
        if self.adam_bf16:
            defines["ADAM_MOMENTS_BF16"] = "1"
        if self.splat_kernel is not SplatKernel.GAUSSIAN:
            defines["SPLAT_KERNEL"] = str(list(SplatKernel).index(self.splat_kernel))
        if self.splat_profile_power != 1.0:
            defines["SPLAT_PROFILE_POWER"] = repr(float(self.splat_profile_power))
        if self.splat_profile_onset != 0.0:
            defines["SPLAT_PROFILE_ONSET"] = repr(float(self.splat_profile_onset))
        return defines or None


def resolve_stbn_config(args: argparse.Namespace) -> StbnConfig:
    """The run's STBN policy, agreeing with :class:`ShaderSpecialization`.

    Both are derived here from the same args so ``backward_compacted`` can never
    disagree with whether ``STBN_ENABLED`` was actually injected — a mismatch
    would bind shader parameters the compiled kernel never declared.
    """
    specialization = ShaderSpecialization.from_args(args)
    return StbnConfig(
        pixel_fraction=float(getattr(args, "stbn_pixel_fraction", 1.0)),
        warmup_iters=int(getattr(args, "stbn_warmup_iters", 0)),
        backward_compacted=specialization.stbn_enabled,
        source=str(getattr(args, "stbn_source", "nvidia")),
        file_path=str(getattr(args, "stbn_file", "")),
        seed=int(getattr(args, "stbn_seed", 0)),
        independent=bool(getattr(args, "stbn_independent", False)),
        select_mode=str(getattr(args, "stbn_select_mode", "global")),
    )


def resolve_adaptive_config(args: argparse.Namespace) -> AdaptiveStbnConfig:
    """The run's importance-guided sampling policy.

    Off unless ``--stbn-adaptive`` is passed, in which case every pre-existing
    regime is unaffected — the flag is the only thing that routes a step onto the
    adaptive path.
    """
    return AdaptiveStbnConfig(
        enabled=bool(getattr(args, "stbn_adaptive", False)),
        alpha=float(getattr(args, "stbn_importance_alpha", 0.75)),
        sigma=float(getattr(args, "stbn_importance_sigma", 8.0)),
        downsample=int(getattr(args, "stbn_importance_downsample", 4)),
        uniform=str(getattr(args, "stbn_adaptive_uniform", "blue_temporal")),
        unbiased=not bool(getattr(args, "stbn_adaptive_biased", False)),
    )


@dataclass
class TrainingRuntime:
    pipeline: TrainingPipeline
    viewer: ViewerSession
    profiler: Profiler | None
    artifacts: RunArtifacts
    queue_limiter: QueueLimiter
    densifier: DensificationController
    reporter: "TrainingReporter"


class TrainingApplication:
    """Owns the prepare -> train -> finish lifecycle of one experiment."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.paths = RunPaths.create(self.run_name)
        self.profiler: Profiler | None = None
        self.profiling_enabled = False

        self.scene_context: SceneContext | None = None
        self.targets: TargetDataset | None = None
        self.runtime: TrainingRuntime | None = None
        self.densify_mode: DensificationMode | None = None
        self.densify_config: Any = None
        self.structure_config: Any = None
        self.base_lrs: AdamLRs | None = None
        self.effective_lambda_l2 = 0.0
        self.effective_lambda_ssim = 0.0

        self.device: Any = None
        self.gpu_lock: Any = None
        self.pipeline: TrainingPipeline | None = None
        self.viewer: ViewerSession | None = None
        self.error_log: ErrorLog | None = None
        self.target_textures: dict[int, Any] = {}
        self.structure_textures: dict[int, Any] = {}
        self._stream_targets = False
        self._memory_summary: dict = {}
        self.mean_lr: MeanLearningRateSchedule | None = None
        self.extinction_hardness_schedule: ExtinctionHardnessSchedule | None = None
        self.appearance_schedule: AppearanceSchedule | None = None

    @classmethod
    def from_cli(cls) -> "TrainingApplication":
        return cls(build_parser().parse_args())

    def prepare_training_context(self) -> None:
        self.profiling_enabled = apply_profile_session_overrides(self.args)
        self._create_device()
        self._prepare_scene(self.args.scene)
        self._create_pipeline_for_scene(self._sceneContext)
        self._prepare_textures()
        self._start_viewer(self.scene_context)
        self._start_profiler()
        self._create_training_services()

    def _prepare_scene(self, scene_dir) -> None:
        """Load the dataset at ``scene_dir`` and resolve everything that depends
        on it (configs, LRs, GT targets). Shared by the CLI path and the hub
        (after a dataset is chosen). Does NOT create the device/pipeline/viewer."""
        self.args.scene = str(scene_dir)
        self.scene_context = load_scene_context(self.args)

        self.densify_mode = validate_training_config(
            self.args,
            n_cams=self._sceneContext.camera_count,
            n_train=len(self._sceneContext.train_indices),
            n_test=len(self._sceneContext.test_indices),
        )

        (
            self.densify_config,
            self.structure_config
        ) = build_densify_configs(self.args, self._sceneContext.extent)

        (
            self.effective_lambda_l2,
            self.effective_lambda_ssim,
            self.base_lrs
         ) = resolve_loss_and_lrs(self.args)

        self._write_resolved_config()
        self._prepare_target_dataset()

    def start_training(self) -> None:
        # ------ prepare for training -------------------------
        self.prepare_training_context()
        self._run_training_and_post()


    def _begin_repeat_run(self) -> None:
        """Make room for ANOTHER run in the same hub session.

        No-op for the session's first run. For every one after it: a fresh output
        directory, and the previous run's GPU resources released before the new
        ones are allocated.
        """
        if self.runtime is None:
            return
        self._rotate_run_dir()
        self._release_run_resources()

    def _rotate_run_dir(self) -> None:
        """Point ``self.paths`` at an unused ``outputs/train/<run>`` directory.

        A run owns its directory: config.json, log.txt/json, test_psnr.json,
        cams/, output.ply. A second run in the same session writing into the first
        one's would interleave two sets of numbers under one name, so it gets its
        own — ``<run_name>_2``, ``_3``, ... for an explicit ``--run-name``, or
        simply the current timestamp when the name was auto-generated.
        """
        base = self.args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        runs = self.paths.root.parent        # RunPaths owns the layout, not this
        name, index = base, 1
        while (runs / name).exists():
            index += 1
            name = f"{base}_{index}"
        self.run_name = name
        self.paths = RunPaths.create(name)
        print(f"[hub] next run writes to {self.paths.root}")

    def _release_run_resources(self) -> None:
        """Drop the finished run's GPU-heavy objects before the next one is built.

        A pipeline plus one target texture per training camera (and a structure
        texture per training camera under SAD-GS) is the bulk of what a run allocates, so
        holding the old set while the new one is created would double the peak —
        which on a large capture is the difference between fitting in VRAM and not.

        Call with rendering suspended, and only once the queue can be drained: the
        viewer's own pipeline is a SEPARATE one (it keeps showing the last snapshot
        after this), but the trainer's buffers must not be released mid-flight.
        """
        if self.pipeline is not None:
            try:
                self.pipeline.runtime.wait_for_idle()
            except Exception as error:                 # noqa: BLE001
                report_exception("[hub] draining the queue before release failed",
                                 error, self._viewer.state)
        self.runtime = None
        self.pipeline = None
        self.targets = None
        self.target_textures = {}
        self.structure_textures = {}
        self.scene_context = None
        self.densify_mode = None        # all four are re-resolved by _prepare_scene
        self.densify_config = None
        self.structure_config = None
        self.base_lrs = None
        self.profiler = None            # already closed by _run_training_and_post


    def _attach_scene_to_viewer(self) -> None:
        """Hand the chosen scene's held-out cameras to the hub viewer's dropdown."""
        scene = self._sceneContext
        images = scene.colmapScene.images
        test_cameras = [
            (images[i].image_name, images[i].camera)
            for i in scene.test_indices if 0 <= i < len(images)
        ]
        self._viewer.attach_scene(test_cameras)

    def _run_training_and_post(self, *, hold_viewer: bool = True) -> None:
        """Train, then write the run's post-training artifacts.

        ``hold_viewer`` is what the CLI path wants: block on the viewer window so
        the process outlives the run. The hub passes False — it services the panel
        itself afterwards, so parking in a thread join there would freeze every
        post-training action (see ``TrainingHub._hub_train_scene``).
        """
        # ---- launch training loop --------------------
        result = self.build_training_loop().launch_training_loop()
        self._memory_summary = self._report_memory()

        viewer = self._training_runtime.viewer
        pipeline = self._training_runtime.pipeline
        viewer.finish_training(pipeline)
        try:
            with viewer.unthrottled():
                viewer.checkpoint()
                self._publish_finalize("held-out test PSNR", 0.05)
                self._write_final_test_psnr(result)

                viewer.checkpoint()
                self._publish_finalize("final metrics", 0.15)
                run_post_training(
                    progress=self._publish_finalize,
                    checkpoint=viewer.checkpoint,
                    pipeline=pipeline,
                    scene=self._sceneContext.colmapScene,
                    args=self.args,
                    run_dir=self.paths.root,
                    cams_dir=self.paths.cameras,
                    targets_srgb=self._targets.srgb,
                    targets_linear=self._targets.linear,
                    test_indices=self._sceneContext.test_indices,
                    stride=self._sceneContext.holdout_stride,
                    n_cams=self._sceneContext.camera_count,
                    train_wall_s=result.wall_time_seconds,
                    completed_iters=result.iterations,
                    t_start=result.started_at,
                    psnr_curve=result.psnr_curve,
                )
        except RunStopped:
            viewer.state.log("Run stopped; remaining evaluation and exports skipped. "
                             "The trained model is still available.")
            print("[post-training] stopped; retaining the trained model.")
        finally:
            if self._training_runtime.profiler is not None:
                self._training_runtime.profiler.close()
            self._training_runtime.artifacts.close_all()

        if hold_viewer:
            stats = viewer.state.snapshot().stats
            viewer.state.training_completed(ResultInfo(
                iterations=result.iterations, target_iterations=self.args.iters,
                wall_s=result.wall_time_seconds, splats=stats.gaussians,
                run_dir=str(self.paths.root)))
            # Metrics must finish (or be stopped) BEFORE the CLI waits for the
            # window to close. The hub returns to its command loop instead.
            viewer.hold_after_training(pipeline)
        # The COMPLETED transition (and the result record) is published by the hub
        # once this returns — see ``TrainingHub._hub_train_scene``.

    def _report_memory(self) -> dict:
        """Print and return the run's sort-pair demand and measured VRAM."""
        pipeline = self._pipeline
        peak_pairs, overflows = (
            int(v) for v in pipeline.buffers.sort_stats.to_numpy().view(np.uint32))
        summary = {"max_splats": pipeline.max_splats,
                   "max_sort_pairs": pipeline.max_sort_pairs,
                   "peak_sort_pairs": peak_pairs, "sort_overflows": overflows,
                   "targets_streamed": self._stream_targets}
        measured = query_video_memory(self.device.info.adapter_name)
        vram = ""
        if measured is not None:
            summary["vram_budget_bytes"], summary["vram_usage_bytes"] = measured
            vram = (f"; VRAM in use {measured[1] / GIB:.2f} of "
                    f"{measured[0] / GIB:.2f} GiB budget")
        print(f"[memory] peak sort pairs {peak_pairs:,} of "
              f"{pipeline.max_sort_pairs:,}{vram}")
        if overflows:
            print(f"[memory] WARNING: {overflows} sorts dropped pairs past the "
                  f"sort capacity (those renders miss splats)")
        return summary

    def _publish_finalize(self, label: str, fraction: float) -> None:
        """Publish post-training progress (no-op without a viewer).

        Also the callback ``run_post_training`` reports through, so the panel gets
        a determinate bar for the metrics pass instead of a frozen iteration count.
        """
        if self.viewer is None:
            return
        state = self._viewer.state
        phase = state.snapshot().training_state
        if phase in (TrainingState.PAUSED, TrainingState.STOPPING):
            return
        if phase != TrainingState.FINALIZING:
            state.training_finalizing(label)
        state.set_progress(label, fraction)

    def _prepare_target_dataset(self) -> None:
        scene = self._sceneContext
        self.targets = TargetDataset(
            scene=scene.colmapScene,
            width=scene.width,
            height=scene.height,
            use_alpha=self.args.w_alpha > 0,
        )
        # The per-cam GT is the reference every cams/cam_XX/iter_*.png is read
        # against, so it is written for every run, hub or CLI — unless a sweep
        # opts out, having already cached one copy per scene elsewhere.
        if getattr(self.args, "write_ground_truths", True):
            self.targets.write_ground_truths(self.paths.cameras)

    def _create_device(self) -> None:
        """Create the GPU device + shared queue lock. Split out from pipeline
        creation so the hub can bring the device (and viewer) up BEFORE a dataset
        is chosen."""
        self.device = create_training_device(self.args)
        self.gpu_lock = create_gpu_queue_lock(viewer_enabled=not self.args.no_viewer)

    def _create_pipeline_for_scene(self, scene: SceneContext) -> None:
        """Build the training pipeline for a loaded scene (device must exist)."""
        self.pipeline = create_pipeline(
            args=self.args,
            device=self.device,
            gpu_lock=self.gpu_lock,
            scene=scene,
            max_splats=self._plan_memory(scene),
            lambda_l2=self.effective_lambda_l2,
            lambda_ssim=self.effective_lambda_ssim,
            initial_state=scene.colmapScene.initial_points,
        )
        self._pipeline.set_mipsplatting_cameras(
            [scene.colmapScene.images[i].camera for i in scene.train_indices])

    def _plan_memory(self, scene: SceneContext) -> int:
        """Fit splat capacity to the VRAM budget, loudly; returns the capacity.

        Lowering the capacity changes the trained model, so it is the last
        resort and always reported. The viewer is budgeted at full capacity: it
        grows with the live splat count up to the trainer's.
        """
        args = self.args
        self._stream_targets = args.image_residency == "stream"
        structure = self._densify_mode == DensificationMode.STRUCTURE_AWARE
        module = _module_for_reflection(self.device)
        target_bytes = 0 if self._stream_targets else (
            len(scene.train_indices) * scene.width * scene.height
            * (4 + (8 if structure else 0)))

        def cost(capacity: int) -> int:
            shape = {"max_splats": capacity, "width": scene.width,
                     "height": scene.height, "sh_degree": args.sh_degree}
            total = estimate_pipeline_bytes(module, structure_aware=structure, **shape)
            if not args.no_viewer:
                total += estimate_pipeline_bytes(module, viewer_only=True, **shape)
            return total + target_bytes

        requested = args.max_splats
        available = available_bytes(args.vram_budget_gib, self.device.info.adapter_name)
        if available is None:
            print(f"[memory] VRAM budget unknown (pass --vram-budget-gib); "
                  f"capacity {requested:,} splats unchecked")
            return requested
        capacity = fit_capacity(requested, available, cost)
        if capacity < scene.initial_gaussian_count:
            raise RuntimeError(
                f"VRAM budget of {available / GIB:.2f} GiB cannot hold the "
                f"{scene.initial_gaussian_count:,} initial points; lower --sh-degree, "
                f"--downsample more, or pass --no-viewer / --image-residency stream")
        if capacity < requested:
            print(f"[memory] WARNING: --max-splats {requested:,} does not fit the VRAM "
                  f"budget; capacity lowered to {capacity:,} (densification stops "
                  f"growing there)")
        print(f"[memory] {available / GIB:.2f} GiB available; planned "
              f"{cost(capacity) / GIB:.2f} GiB for {capacity:,} splats")
        return capacity


    def _prepare_textures(self) -> None:
        """Upload training targets, then build their structure fields.

        The per-image CPU work (alpha pad + float->uint8 quantize, ~12 MB of
        pixels per view) runs on a thread pool while the uploads stay on THIS
        thread: the device/queue is not thread-safe — it is the same one the
        viewer shares under ``gpu_lock`` — so only the numpy half is parallel.
        Results still arrive in camera order, and ``map_ordered``'s bounded
        lookahead keeps at most a few packed images alive at once.
        """
        pipeline = self._pipeline
        targets = self._targets.srgb
        indices = self._sceneContext.train_indices
        # Streamed fields are derived per step on the GPU, never stored.
        resident_fields = (self._densify_mode == DensificationMode.STRUCTURE_AWARE
                           and not self._stream_targets)
        bytes_per_pixel = 4 + (8 if resident_fields else 0)
        planned_bytes = len(indices) * self._sceneContext.width * self._sceneContext.height * bytes_per_pixel
        where = "host RAM" if self._stream_targets else "GPU"
        print(f"[memory] {len(indices)} training targets"
              f"{'/structure fields' if resident_fields else ''} on {where}: "
              f"{planned_bytes / 1024**3:.2f} GiB")
        packed_targets = map_ordered(
            pipeline.pack_target_texture, [targets[i] for i in indices])
        if self._stream_targets:
            self.target_textures = dict(zip(indices, packed_targets))
        else:
            self.target_textures = {
                i: pipeline.upload_target_texture(packed)
                for i, packed in zip(indices, packed_targets)
            }

        self.structure_textures = StructureTextureProvider(
            args=self.args,
            device=self.device,
            scene=self._sceneContext,
            targets=self._targets.srgb,
            densify_mode=self._densify_mode,
            target_textures=self.target_textures,
            pipeline=self._pipeline,
            resident=not self._stream_targets,
        ).load()

    def _start_viewer(self, scene: SceneContext) -> None:
        if not self.args.no_viewer:
            # Held-out test cameras (name, Camera), so the viewer's C/V keys can
            # snap to the exact poses eval renders from — the direct way to
            # compare in-viewer SH against the eval render at the same view.
            images = scene.colmapScene.images
            test_cameras = [
                (images[i].image_name, images[i].camera)
                for i in scene.test_indices
                if 0 <= i < len(images)
            ]
            self.viewer = ViewerSession.create(
                device=self.device,
                width=scene.width,
                height=scene.height,
                fovy=getattr(self.args, "fovy", 45.0),
                gpu_lock=self.gpu_lock,
                pipeline=self._pipeline,
                test_cameras=test_cameras,
            )
        else:
            self.viewer = ViewerSession(None)

        if self.error_log is not None:
            self.error_log.attach(self.viewer.state)
        self.viewer.start()

    def _start_profiler(self):
        if self.profiling_enabled:
            self.profiler = Profiler(self.args)
            self.profiler.open_window(
                self._pipeline, self._sceneContext.width,
                self._sceneContext.height)

    def _create_training_services(self) -> None:
        scene = self._sceneContext
        artifacts = RunArtifacts.create(self.paths.root, self.args)
        # The reporter publishes into the session state too, so the viewer's
        # training progress bar / metrics come from the same numbers as the logs.
        reporter = TrainingReporter(artifacts, self.args, scene,
                                    state=self._viewer.state)
        reporter.write_header()

        self.mean_lr = MeanLearningRateSchedule(
            lr_init=self.base_lrs.mean * scene.extent,
            lr_final=self.base_lrs.mean * 1e-2 * scene.extent,
            max_steps=_resolve_position_lr_max_steps(
                self.args.position_lr_max_steps, self.args.iters),
        )

        self.extinction_hardness_schedule = ExtinctionHardnessSchedule(
            target=self.args.extinction_hardness,
            warmup=self.args.extinction_hardness_warmup,
        )

        self.appearance_schedule = AppearanceSchedule(
            appearanceConfig=AppearanceConfig(
                sh_degree=self.args.sh_degree,
                band_interval=_resolve_sh_band_interval(
                    self.args.sh_band_interval, self.args.iters, self.args.sh_degree),
            )
        )

        queue_limiter = QueueLimiter(self._pipeline.runtime)

        densifier = DensificationController(
            pipeline=self._pipeline,
            args=self.args,
            densify_mode=self._densify_mode,
            densify_cfg=self.densify_config,
            structure_cfg=self.structure_config,
            extent=scene.extent,
            image_width=scene.width,
            image_height=scene.height,
            morton_world_min=scene.morton_min,
            morton_world_max=scene.morton_max,
            extinction_hardness_schedule=self.extinction_hardness_schedule,
            queue_limiter=queue_limiter,
        )

        self.runtime = TrainingRuntime(
            pipeline=self._pipeline,
            viewer=self._viewer,
            profiler=self.profiler,
            artifacts=artifacts,
            queue_limiter=queue_limiter,
            densifier=densifier,
            reporter=reporter,
        )

    def build_training_loop(self) -> "TrainingLoop":
        """The configured loop for the prepared context, WITHOUT running it.

        Extracted from :meth:`_run_training_and_post` so that a harness which
        needs a real, mid-training GPU state — ``scripts/running/bench_iter.py``
        for frozen-iteration timing, the paper suite's gradient probe for a
        trained scene at iteration N — can drive ``advance_to_capture`` itself
        instead of duplicating a thirteen-argument construction against private
        attributes. ``prepare_training_context`` must have run first.
        """
        return TrainingLoop(
            args=self.args,
            scene=self._sceneContext,
            targets=self._targets,
            runtime=self._training_runtime,
            densify_mode=self._densify_mode,
            densify_config=self.densify_config,
            base_lrs=self._base_lrs,
            target_textures=self.target_textures,
            structure_textures=self.structure_textures,
            mean_lr=self._mean_lr,
            paths=self.paths,
            extinction_hardness_schedule=self._extinction,
            appearance_schedule=self._appearance_schedule,
        )

    def _write_resolved_config(self) -> None:
        sceneContext = self._sceneContext
        base_lrs = self._base_lrs
        config = {
            "scene": self.args.scene,
            "iters": self.args.iters,
            "width": sceneContext.width,
            "height": sceneContext.height,
            "preview_every": self.args.preview_every,
            "log_every": self.args.log_every,
            "densify_every": self.args.densify_every,
            "densify_from": self.args.densify_from,
            "densify_to": self.args.densify_to,
            "densify_config": {
                "t_opacity": self.densify_config.t_opacity,
                "t_big_scale": self.densify_config.t_big_scale,
                "grow_scale3d": self.args.grow_scale3d,
            },
            "max_splats": self.args.max_splats,
            "z_near_cull": self.args.z_near_cull,
            "backend": self.args.backend,
            "seed": self.args.seed,
            "sample_mode": self.args.sample_mode,
            "mip_enabled": bool(self.args.mip_splatting),
            "mip_kernel_size": (
                self.args.mip_kernel_size if self.args.mip_splatting else 0.3),
            "run_name": self.run_name,
            "n_cams": sceneContext.camera_count,
            "n_init_points": sceneContext.initial_gaussian_count,
            "cli_args": dict(vars(self.args)),
            "resolved": {
                "densify_mode": self._densify_mode.value,
                "width": sceneContext.width,
                "height": sceneContext.height,
                "n_cams": sceneContext.camera_count,
                "n_init_points": sceneContext.initial_gaussian_count,
                "n_train": len(sceneContext.train_indices),
                "n_test": len(sceneContext.test_indices),
                "holdout_stride": sceneContext.holdout_stride,
                "eff_lambda_l2": self.effective_lambda_l2,
                "eff_lambda_ssim": self.effective_lambda_ssim,
                "base_lr_scale": base_lrs.scale,
                "base_lr_rot": base_lrs.rot,
                "base_lr_opacity": base_lrs.opacity,
                "base_lr_color": base_lrs.color,
            },
        }
        (self.paths.root / "config.json").write_text(json.dumps(config, indent=2))

    def _write_final_test_psnr(self, result: TrainingResult) -> None:
        # --no-evaluation vetoes this: it renders every held-out camera.
        want_report = bool(getattr(self.args, "run_report", False))
        if (self.args.profile_run or not self.args.evaluation
                or not (self.args.final_test_psnr or want_report)):
            return

        sceneContext = self._sceneContext
        # One render pass serves both files. --run-report needs every view's own
        # metrics, and the mean of those IS the paper metric, so asking
        # eval_test_psnr_srgb for the mean as well would render the held-out set
        # twice for the same number.
        per_view: list[dict] = []
        quality: dict = {}
        if want_report:
            renders_dir = (self.paths.root / "test_renders"
                           if getattr(self.args, "save_test_renders", False)
                           else None)
            per_view, quality = run_report_module.evaluate_test_views(
                self._pipeline, sceneContext.colmapScene,
                sceneContext.test_indices, self._targets.srgb,
                render_dir=renders_dir,
                with_lpips=bool(getattr(self.args, "eval_lpips", False)),
                progress=lambda done, total: self._publish_finalize(
                    f"held-out view {done}/{total}", 0.05 + 0.5 * done / max(total, 1)),
                checkpoint=self._training_runtime.viewer.checkpoint,
            )
            psnr = quality.get("psnr_srgb") or 0.0
            count = int(self._pipeline.buffers.num_gaussians_counter
                        .to_numpy().view(np.uint32)[0])
        else:
            psnr, count = eval_test_psnr_srgb(
                self._pipeline, sceneContext.colmapScene,
                sceneContext.test_indices, self._targets.srgb,
                checkpoint=self._training_runtime.viewer.checkpoint)

        self._training_runtime.viewer.checkpoint()
        active_s = max(0.0, result.wall_time_seconds - result.observation_seconds)

        if self.args.final_test_psnr:
            (self.paths.root / "test_psnr.json").write_text(json.dumps({
                "psnr_srgb": psnr,
                "n_gaussians": count,
                "n_test_cams": len(sceneContext.test_indices),
                "train_wall_s": result.wall_time_seconds,
                # Additive: train_wall_s keeps its exact old meaning, and these
                # separate the loop's own cost from the cost of watching it.
                "train_active_s": active_s,
                "observation_s": result.observation_seconds,
                "iters": result.iterations,
                "flat_bucket_backward": bool(self.args.flat_bucket_backward),
                **self._memory_summary,
            }, indent=2))

            print(f"[final-test-psnr] TEST PSNR sRGB = {psnr:.3f} dB over "
                  f"{len(sceneContext.test_indices)} held-out cams  (n_gaussians={count})")

        if want_report:
            frames = int(getattr(self.args, "render_fps_frames", 0) or 0)
            render = (run_report_module.measure_render_fps(
                          self._pipeline, sceneContext.colmapScene,
                          sceneContext.test_indices[0], frames,
                          checkpoint=self._training_runtime.viewer.checkpoint)
                      if frames > 0 and sceneContext.test_indices
                      else {"frames": 0, "fps": None, "ms_per_frame": None})
            self._training_runtime.viewer.checkpoint()
            report = run_report_module.build_report(
                pipeline=self._pipeline,
                scene_context=sceneContext,
                args=self.args,
                run_name=self.run_name,
                result=result,
                per_view=per_view,
                quality=quality,
                render=render,
                loop_timing={
                    "observation_s": result.observation_seconds,
                    "observation_total_s": result.observation_total_seconds,
                },
                process_started_at=_PROCESS_STARTED_AT,
            )
            if result.stage_timing is not None:
                report["stage_timing"] = result.stage_timing.summary()
            path = run_report_module.write_run_report(report, self.paths.root)
            fps = render.get("fps")
            print(f"[run-report] {len(per_view)} held-out views  "
                  f"psnr={psnr:.3f} dB  ssim={quality.get('ssim_srgb') or 0.0:.4f}  "
                  f"active={active_s:.1f}s (wall {result.wall_time_seconds:.1f}s, "
                  f"observation {result.observation_seconds:.1f}s)  "
                  f"n_g={count}"
                  + (f"  render={fps:.1f} fps" if fps else "")
                  + f"  -> {path}")

    @property
    def _sceneContext(self) -> SceneContext:
        assert self.scene_context is not None
        return self.scene_context

    @property
    def _targets(self) -> "TargetDataset":
        assert self.targets is not None
        return self.targets

    @property
    def _training_runtime(self) -> TrainingRuntime:
        assert self.runtime is not None
        return self.runtime

    @property
    def _densify_mode(self) -> DensificationMode:
        assert self.densify_mode is not None
        return self.densify_mode

    @property
    def _base_lrs(self) -> AdamLRs:
        assert self.base_lrs is not None
        return self.base_lrs

    @property
    def _pipeline(self) -> TrainingPipeline:
        assert self.pipeline is not None
        return self.pipeline

    @property
    def _viewer(self) -> ViewerSession:
        assert self.viewer is not None
        return self.viewer

    @property
    def _mean_lr(self) -> MeanLearningRateSchedule:
        assert self.mean_lr is not None
        return self.mean_lr

    @property
    def _extinction(self) -> ExtinctionHardnessSchedule:
        assert self.extinction_hardness_schedule is not None
        return self.extinction_hardness_schedule
    
    @property
    def _appearance_schedule(self) -> AppearanceSchedule:
        assert self.appearance_schedule is not None
        return self.appearance_schedule

class TrainingLoop:
    """Executes optimizer steps, then scheduled mutations, then observations."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        scene: SceneContext,
        targets: "TargetDataset",
        runtime: TrainingRuntime,
        densify_mode: DensificationMode,
        densify_config: Any,
        base_lrs: AdamLRs,
        target_textures: dict[int, Any],
        structure_textures: dict[int, Any],
        mean_lr: MeanLearningRateSchedule,
        paths: RunPaths,
        extinction_hardness_schedule: ExtinctionHardnessSchedule | None = None,
        appearance_schedule: AppearanceSchedule | None = None,
    ) -> None:
        self.args = args
        self.scene = scene
        self.targets = targets
        self.runtime = runtime
        self.densify_mode = densify_mode
        self.densify_config = densify_config
        self.base_lrs = base_lrs
        self.target_textures = target_textures
        self.structure_textures = structure_textures
        self.mean_lr = mean_lr
        self.extinction_hardness_schedule: ExtinctionHardnessSchedule | None = extinction_hardness_schedule
        self.appearance_schedule: AppearanceSchedule | None = appearance_schedule
        self.paths = paths

        rng = np.random.default_rng(args.seed)
        self.sampler = CameraSampler(
            args.sample_mode,
            scene.train_indices,
            scene.test_indices,
            args.seed,
            rng,
        )
        self.schedule = TrainingSchedule(ScheduleConfig.from_args(args))

        # STBN sparse supervision. The mask's temporal axis is indexed by how
        # many times a CAMERA has been drawn, not by the iteration, so repeat
        # visits of one view walk the sequence the mask was built to distribute
        # along. Each camera also gets a fixed phase, so cameras don't all march
        # through the same slices in lockstep (with a shared start every view
        # would be sampling slice 0's pixels on its first visit, slice 1's on its
        # second, and so on — correlating the whole dataset to one sequence).
        stbn_config = self.runtime.pipeline.stbn_config
        self.stbn_density = StbnDensitySchedule(
            target_fraction=stbn_config.pixel_fraction,
            warmup=stbn_config.warmup_iters,
        ) if stbn_config.enabled else None
        self.stbn_slices = self.runtime.pipeline.stbn_slices
        self.stbn_visits: dict[int, int] = {}
        self.stbn_phase: dict[int, int] = {}
        self.stbn_independent = stbn_config.independent
        self._stbn_rng = np.random.default_rng(int(getattr(self.args, "stbn_seed", 0)))
        # Separate stream for the adaptive path's per-visit IID field, so adding it
        # cannot perturb the slice sequence any pre-existing regime walks.
        self._stbn_sample_rng = np.random.default_rng(
            int(getattr(self.args, "stbn_seed", 0)) + 0x5EED)

        self.training_step = 0
        self.gaussian_count = scene.initial_gaussian_count
        self.eval_time_total = 0.0
        self.psnr_curve_records: list[dict] = []
        self.started_at = 0.0
        self.post_iteration_clean_timing = 0.0
        # Wall-clock spent inside _produce_observations — the whole observation
        self.observation_seconds = 0.0
        self.observation_in_wall_seconds = 0.0
        self._stage_timing = StageTimingRecorder(
            every=int(getattr(args, "stage_timing_every", 0) or 0),
            total_iters=int(args.iters),
        )

    def launch_training_loop(self) -> TrainingResult:
        self._begin_training_loop()

        try:
            while self._should_continue():
                self.runtime.viewer.checkpoint()
                it_context = self._next_iteration()
                self._prepare_iteration(it_context)
                self._capture_if_requested(it_context)

                self._execute_training_step(it_context)

                # Freeze training time before the final iteration's observations.
                if it_context.step_number == self.args.iters:
                    self._record_current_timing()

                self._apply_mutations(it_context)
                observed_at = time.perf_counter()
                try:
                    should_continue = self._produce_observations(it_context)
                finally:
                    spent = time.perf_counter() - observed_at
                    self.observation_seconds += spent
                    if it_context.step_number != self.args.iters:
                        self.observation_in_wall_seconds += spent
                self._present_profile_frame_if_enabled()

                if not should_continue:
                    break
        except RunStopped:
            pass

        return self._complete_training_loop()

    def advance_to_capture(self, capture_step: int) -> IterationContext:
        """Drive the normal training loop up to — but NOT executing — the
        train_step at ``capture_step``, returning that iteration's context
        (pinned camera + resolved LRs + scheduled events).

        This mirrors ``Profiler.run_capture_if_due``'s freeze point EXACTLY: the
        capture fires at the top of iter == capture_iter, after
        ``_next_iteration`` / ``_prepare_iteration`` but before
        ``_execute_training_step``. Every densify / opacity-reset / morton / mip
        mutation for iters ``1..capture_step-1`` runs as in a normal run, so the
        frozen state a benchmark then measures matches what the Nsight capture
        profiles at that iteration. Drains the queue before returning so the
        pinned state is fully realized on the GPU.
        """
        self._begin_training_loop()
        while self._should_continue():
            self.runtime.viewer.checkpoint()
            context = self._next_iteration()
            self._prepare_iteration(context)
            if context.step_number >= capture_step:
                self.runtime.pipeline.runtime.wait_for_idle()
                return context
            self._execute_training_step(context)
            self._apply_mutations(context)
            if not self._produce_observations(context):
                break
        raise RuntimeError(
            f"training ended before reaching capture step {capture_step}")

    def _begin_training_loop(self) -> None:
        if self.args.wait_for_viewer:
            self.runtime.viewer.wait_for_start()
        self.runtime.viewer.set_training_active(True)
        self.started_at = time.time()
        self.runtime.reporter.start(self.started_at)

    def _should_continue(self) -> bool:
        profiler = self.runtime.profiler
        return (
            self.training_step < self.args.iters
            and (profiler is None or not profiler.should_stop())
            # The viewer's Stop button: end the run early but finalize normally, so
            # what has been optimized so far stays a usable result.
            and not self.runtime.viewer.stop_requested()
        )

    def _stbn_variables(self, camera_index: int) -> "StbnVariables | None":
        if self.stbn_density is None:
            return None
        slices = max(self.stbn_slices, 1)
        pixels = stbn_module.pixels_per_tile(self.stbn_density.at(self.training_step), TILE_AREA)
        # From its OWN stream, not from _stbn_rng. Drawing it there would insert an
        # extra consumption into the slice sequence and silently change which
        # slices every existing --stbn-independent run visits, invalidating the
        # cached iid_white / indep_blue baselines this work is compared against.
        sample_seed = int(self._stbn_sample_rng.integers(1 << 32))
        if self.stbn_independent:
            return StbnVariables(slice_index=int(self._stbn_rng.integers(slices)),
                                 pixels_per_tile=pixels,
                                 sample_seed=sample_seed)
        if camera_index not in self.stbn_phase:
            mixed = int(camera_index) * 2654435761 + int(self.args.stbn_seed) * 40503
            self.stbn_phase[camera_index] = mixed % slices
        visit = self.stbn_visits.get(camera_index, 0)
        self.stbn_visits[camera_index] = visit + 1
        return StbnVariables(
            slice_index=(visit + self.stbn_phase[camera_index]) % slices,
            pixels_per_tile=pixels,
            sample_seed=sample_seed,
        )

    def _timed_stbn_variables(self, camera_index: int) -> "StbnVariables | None":
        started = time.perf_counter()
        try:
            return self._stbn_variables(camera_index)
        finally:
            self._stage_timing.add_host("mask_sampling", (time.perf_counter() - started) * 1e3)

    def _next_iteration(self) -> IterationContext:
        self.training_step += 1
        camera_index = self.sampler.next()
        camera_entry = self.scene.colmapScene.images[camera_index]
        learning_rates = AdamLRs(
            mean=self.mean_lr.at(self.training_step - 1),
            scale=self.base_lrs.scale,
            rot=self.base_lrs.rot,
            opacity=self.base_lrs.opacity,
            color=self.base_lrs.color,
            sh_rest=self.base_lrs.sh_rest,
        )

        return IterationContext(
            step_number=self.training_step,
            camera_index=camera_index, #the cam idx is in the camera entry but the colmap cam id is 1-indexed, kept separate to avoid conf
            camera_entry=camera_entry,
            learning_rates=learning_rates,
            events=self.schedule.events_at(self.training_step),

            iteration_contextual_variables=IterationContextualVariables(
                hardness_value=(self.extinction_hardness_schedule.at(self.training_step) if self.extinction_hardness_schedule is not None else 1.0),
                sh_degree=(self.appearance_schedule.at(self.training_step) if self.appearance_schedule is not None else int(self.args.sh_degree)),
                densification_variables=DensificationProperties(
                    structure_tensor_texture=self.structure_textures[camera_index] if self._should_accumulate_structure(self.training_step) else None,
                    #TODO: writing the threshold like this is shit, need a better way to avoid having these damn hyperparams float as magic stuff around
                    # ALSO -> the product is hacky way to hide that our magic hyperparms should all be normalized
                    structure_kwargs={"per_view_grad_thresh": (self.args.sad_freq_grad_threshold * self.scene.width * self.scene.height * 3.0) if self._should_accumulate_structure(self.training_step) else None},
                    track_blend_weight=self._tracks_blend_weight(self.training_step),
                ),
                contextual_stbn=self._timed_stbn_variables(camera_index),
            ),
        )

    def _prepare_iteration(self, context: IterationContext) -> None:
        self.runtime.pipeline.set_target_texture(self.target_textures[context.camera_index])

        if context.step_number == 1:
            print(f"[determinism] iter 1 cam_idx={context.camera_index} (should match across same-seed runs)")

        self.runtime.queue_limiter.before_submit()

    def _capture_if_requested(self, context: IterationContext) -> None:
        if self.runtime.profiler is None:
            return
        captured = self.runtime.profiler.run_capture_if_due(
            self.runtime.pipeline,
            context,
            self.runtime.queue_limiter,
        )
        if captured:
            self.runtime.artifacts.close_all()
            raise SystemExit(0)

    def _execute_training_step(self, context: IterationContext) -> None:
        pipeline = self.runtime.pipeline
        # No-op unless --stage-timing-every selected this iteration.
        self._stage_timing.before_step(context.step_number, pipeline)

        pipeline.train_step(context.camera_entry.camera, lrs=context.learning_rates,
                            contextual=context.iteration_contextual_variables)

        self._stage_timing.after_step(context.step_number, pipeline)
        self.runtime.queue_limiter.record(self.runtime.pipeline.last_submit_id)
        self._stream_viewer_snapshot_if_due(context)

    def _record_current_timing(self) -> None:
        # Lets way for all the gpu work to finish to get a fair and accurate final timing
        self.runtime.pipeline.runtime.wait_for_idle()
        self.post_iteration_clean_timing = time.time() - self.started_at


    def _apply_mutations(self, context: IterationContext) -> None:
        events = context.events
        kind = "structure" if events.structure else (
            "filler" if events.densify else "periodic")
        with self._stage_timing.mutations.measure(
            context.step_number, self.runtime.pipeline, kind,
            active=events.densify or events.morton or events.mip_periodic,
        ):
            self._execute_mutations(context)

    def _execute_mutations(self, context: IterationContext) -> None:
        events = context.events
        pipeline = self.runtime.pipeline

        if events.densify:
            densify_started = time.perf_counter()
            outcome = self.runtime.densifier.run(
                context.step_number, events, self.gaussian_count)

            self._stage_timing.add_host("densify_prune", (time.perf_counter() - densify_started) * 1e3)

            self.gaussian_count = outcome.count
            self.runtime.reporter.report_densify(outcome)

        if events.opacity_reset:
            pipeline.reset_opacity(self.densify_config)
            self.runtime.reporter.report_opacity_reset(context.step_number)

        if events.morton:
            pipeline.morton_sort(
                self.scene.morton_min,
                self.scene.morton_max,
                carry_eta=(self.densify_mode == DensificationMode.STRUCTURE_AWARE),
            )
            pipeline.clear_tracking()

            if pipeline.last_submit_id:
                self.runtime.queue_limiter.record(pipeline.last_submit_id)

            #self.runtime.reporter.report_morton_sort(context.step_number)

        if events.mip_periodic:
            pipeline.compute_3d_filter()

    def _produce_observations(self, context: IterationContext) -> bool:
        self.runtime.viewer.checkpoint()
        self.runtime.reporter.report_throughput(context)

        if context.step_number > 1 and (context.events.log or context.events.preview):
            ok, self.gaussian_count = self.runtime.reporter.report_iteration(
                context=context,
                pipeline=self.runtime.pipeline,
                target=self.targets.srgb[context.camera_index][..., :3],
                current_count=self.gaussian_count,
            )
            if not ok:
                return False

        if context.step_number > 1 and context.events.render_all:
            state = self.runtime.viewer.state
            state.begin_activity("Rendering every camera")
            with self.runtime.viewer.unthrottled():
                try:
                    render_all_cameras(
                        self.runtime.pipeline,
                        self.scene.colmapScene,
                        context.step_number,
                        self.paths.cameras,
                        checkpoint=self.runtime.viewer.checkpoint,
                    )
                finally:
                    state.end_activity()

        self._record_psnr_curve_if_due(context)
        self._run_evaluation_if_due(context)
        return True

    def _should_accumulate_structure(self, training_step) -> bool:
        return (
            self.densify_mode == DensificationMode.STRUCTURE_AWARE
            and self.args.densify_from < training_step <= self.args.densify_to
            and training_step % self.args.sad_accum_every == 0
        )

    def _tracks_blend_weight(self, training_step: int) -> bool:
        """Only the windows that end on a blend-pruning tick need blend weights."""
        first_window = self.args.prune_blend_from - self.args.densify_every
        return (self.args.prune_blend_weight > 0.0
                and first_window < training_step <= self.args.densify_to)

    def _present_profile_frame_if_enabled(self) -> None:
        if self.runtime.profiler is not None:
            self.runtime.profiler.present_frame(
                self.runtime.pipeline.last_submit_id)

    def _stream_viewer_snapshot_if_due(self, context: IterationContext) -> None:
        if context.step_number % self.args.stream_every == 0:
            self.runtime.viewer.push_snapshot(self.runtime.pipeline)

    def _record_psnr_curve_if_due(self, context: IterationContext) -> None:
        stream = self.runtime.artifacts.psnr_curve_json
        if stream is None or not context.events.psnr_curve:
            return
        started = time.time()
        psnr, count = eval_test_psnr_srgb(
            self.runtime.pipeline,
            self.scene.colmapScene,
            self.scene.test_indices,
            self.targets.srgb,
            checkpoint=self.runtime.viewer.checkpoint,
        )
        record = {
            "iter": context.step_number,
            "psnr_srgb": psnr,
            "n_gaussians": count,
        }
        self.psnr_curve_records.append(record)
        stream.write(json.dumps(record) + "\n")
        self.eval_time_total += time.time() - started
        print(f"[psnr-curve] iter {context.step_number:6d}  psnr_srgb={psnr:5.2f}  "
              f"n_g={count} (pre-densify)")

    def _run_evaluation_if_due(self, context: IterationContext) -> None:
        stream = self.runtime.artifacts.eval_curve_json
        if stream is None or not context.events.eval:
            return
        # An eval tick renders every held-out camera, which at the LAST iteration
        # is the whole visible wait after training "ends". The iteration counter
        # cannot show it (it is pinned at N/N), so the sweep says so itself.
        state = self.runtime.viewer.state
        state.begin_activity("Evaluating every camera", 0.0)
        # The sweep BLOCKS the loop, so the viewer is not competing with anything
        # for the queue while it runs - and this is exactly when someone is looking
        # at the window, because the iteration counter has stopped.
        with self.runtime.viewer.unthrottled():
            try:
                self.eval_time_total = run_eval_sweep(
                    pipeline=self.runtime.pipeline,
                    scene=self.scene.colmapScene,
                    it=context.step_number,
                    test_indices=self.scene.test_indices,
                    targets_srgb=self.targets.srgb,
                    targets_linear=self.targets.linear,
                    cams_dir=self.paths.cameras,
                    eval_curve_json=stream,
                    t_start=self.started_at,
                    eval_time_total=self.eval_time_total,
                    eval_holdout=bool(self.args.eval_holdout),
                    with_lpips=bool(getattr(self.args, "eval_lpips", False)),
                    progress=lambda done, total: state.set_progress(
                        "Evaluating every camera",
                        (done / total) if total else None,
                        detail=f"camera {done}/{total}"),
                    checkpoint=self.runtime.viewer.checkpoint,
                )
            finally:
                state.end_activity()

    def _complete_training_loop(self) -> TrainingResult:
        # The last iteration is done, so nothing after this point competes with a
        # training step: give the viewer the GPU back HERE rather than after the
        # held-out PSNR pass (which renders every test camera) has also run.
        self.runtime.viewer.set_training_active(False)
        self.runtime.pipeline.runtime.wait_for_idle()
        if not self.post_iteration_clean_timing:
            self.post_iteration_clean_timing = time.time() - self.started_at
        if self.extinction_hardness_schedule is not None:
            self.runtime.pipeline.set_extinction_hardness(self.extinction_hardness_schedule.target)
        self.runtime.reporter.report_complete(self.training_step, self.post_iteration_clean_timing)
        if self._stage_timing.enabled:
            self._stage_timing.write(self.paths.root)
            summary = self._stage_timing.summary()
            print(f"[stage-timing] {summary['sampled_iters']} sampled iters "
                  f"(1 in {summary['sample_every']}), measured overhead "
                  f"{summary['sample_overhead_s']:.2f}s of "
                  f"{self.post_iteration_clean_timing:.1f}s wall "
                  f"({100.0 * summary['sample_overhead_s'] / max(self.post_iteration_clean_timing, 1e-9):.2f}%)")
        return TrainingResult(
            iterations=self.training_step,
            wall_time_seconds=self.post_iteration_clean_timing,
            psnr_curve=self.psnr_curve_records,
            started_at=self.started_at,
            observation_seconds=self.observation_in_wall_seconds,
            observation_total_seconds=self.observation_seconds,
            stage_timing=(self._stage_timing if self._stage_timing.enabled
                          else None),
        )


class TargetDataset:
    """Owns target loading, lazy linear conversion, and ground-truth images."""

    def __init__(self, scene: Any, width: int, height: int, use_alpha: bool) -> None:
        self._scene = scene
        self._width = width
        self._height = height
        self._use_alpha = use_alpha
        # Decoding the dataset is startup's long pole and OpenCV releases the GIL
        # while it does it, so the sweep is threaded (in camera order — index i
        # must stay camera i). Bounded lookahead keeps the peak RAM below
        # "every image decoded before the first one is consumed".
        started = time.time()
        workers = default_workers_numbers(len(scene.images))
        self._srgb = list(map_ordered(
            lambda entry: load_colmap_image(
                entry, scene.image_dir, width, height, load_alpha=use_alpha),
            scene.images, workers=workers))
        print(f"[data] loaded {len(self._srgb)} targets at {width}x{height} in "
              f"{time.time() - started:.1f}s ({workers} decode threads)", flush=True)
        self._linear_cache: dict[int, np.ndarray] = {}

    @property
    def srgb(self) -> list[np.ndarray]:
        return self._srgb

    def linear(self, camera_index: int) -> np.ndarray:
        target = self._linear_cache.get(camera_index)
        if target is None:
            target = srgb_to_linear(self._srgb[camera_index][..., :3])
            self._linear_cache[camera_index] = target
        return target

    def write_ground_truths(self, cameras_dir: Path) -> None:
        """Dump one ``cam_XX/gt.png`` per camera (skipping any already written).

        Threaded like the decode sweep: PNG encoding is the other startup cost
        that spends its time inside a GIL-releasing C library.
        """
        started = time.time()
        written = sum(map_ordered(
            lambda pair: self._write_ground_truth(cameras_dir, *pair),
            enumerate(self._srgb)))
        if written:
            print(f"[data] wrote {written} ground-truth PNGs in "
                  f"{time.time() - started:.1f}s", flush=True)

    def _write_ground_truth(self, cameras_dir: Path, camera_index: int,
                            target: np.ndarray) -> int:
        camera_dir = cameras_dir / f"cam_{camera_index:02d}"
        camera_dir.mkdir(parents=True, exist_ok=True)
        path = camera_dir / "gt.png"
        if path.exists():
            return 0
        if self._use_alpha and target.shape[-1] == 4:
            pixels = (target * 255 + 0.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(pixels, mode="RGBA").save(path)
        else:
            pixels = (target[..., :3] * 255 + 0.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(pixels, mode="RGB").save(path)
        return 1


class OpacityBundleReporter:
    """Resolves and reports the SAD-GS opacity persistence policy."""

    def __init__(self, *, args: argparse.Namespace, densify_config: Any,
                 structure_config: Any) -> None:
        self._args = args
        self._densify = densify_config
        self._structure = structure_config

    def settings(self) -> dict:
        cap_enabled = bool(
            self._structure.opacity_cap and self._structure.opacity_cap > 0.0)
        return {
            "structure_min_opacity": self._structure.t_opacity,
            "opacity_cap_enabled": cap_enabled,
            "opacity_cap": self._structure.opacity_cap,
            "relative_reset_enabled": bool(self._densify.reset_relative),
            "reset_decay": self._densify.reset_decay,
            "structure_grad_thresh": self._structure.grad_threshold,
            "absgs_grad_thresh": self._args.grad_thresh,
        }

    def report(self, where: str) -> None:
        settings = self.settings()
        print(
            f"[opacity-bundle] resolved @ {where}: "
            f"structure_min_opacity={settings['structure_min_opacity']} "
            f"opacity_cap_enabled={settings['opacity_cap_enabled']} "
            f"opacity_cap={settings['opacity_cap']} "
            f"relative_reset_enabled={settings['relative_reset_enabled']} "
            f"reset_decay={settings['reset_decay']} "
            f"structure_grad_thresh={settings['structure_grad_thresh']} "
            f"absgs_grad_thresh={settings['absgs_grad_thresh']}",
            flush=True,
        )


class StructureTextureProvider:
    """Hides structure cache loading, computation, packing, and GPU upload."""

    def __init__(self, *, args: argparse.Namespace, device: Any,
                 scene: SceneContext, targets: list[np.ndarray],
                 densify_mode: DensificationMode,
                 target_textures: dict[int, Any] | None = None,
                 pipeline: Any = None, resident: bool = True) -> None:
        self._args = args
        self._device = device
        self._scene = scene
        self._targets = targets
        self._densify_mode = densify_mode
        self._resident = resident
        # Training targets: rgba8_unorm textures, or packed uint8 host images when
        # streamed. The resident GPU build reads the textures directly — an
        # all-GPU build with no readback.
        self._target_textures = target_textures
        self._pipeline = pipeline

    def load(self) -> dict[int, Any]:
        if self._densify_mode != DensificationMode.STRUCTURE_AWARE:
            return {}

        cached_fields = self._load_cache()
        if cached_fields is None and self._target_textures is not None:
            sadgs_gpu_backend = self._create_sadgs_gpu_backend()
            if sadgs_gpu_backend is not None and not self._resident:
                # A field is a pure function of its target: the pipeline derives
                # it from the streamed target on the steps that accumulate.
                return dict.fromkeys(self._scene.train_indices, sadgs_gpu_backend)
            if sadgs_gpu_backend is not None:
                return self._build_on_gpu(sadgs_gpu_backend)

        sadgs_gpu_backend = None
        if cached_fields is None:
            sadgs_gpu_backend = self._create_sadgs_gpu_backend()
        else:
            print(f"[SAD] uploading {len(self._scene.train_indices)} "
                  "structure-tensor textures to GPU...", flush=True)

        textures = {}
        started = time.time()
        for camera_index in self._scene.train_indices:
            field = self._field_for(camera_index, cached_fields, sadgs_gpu_backend)
            textures[camera_index] = self._upload(field, f"structure_tensor_{camera_index}")

        if cached_fields is None:
            method = "GPU" if sadgs_gpu_backend is not None else "CPU"
            print(f"[SAD] computed {len(self._scene.train_indices)} fields on {method} "
                  f"in {time.time() - started:.1f}s", flush=True)
        return textures

    def _build_on_gpu(self, gpu) -> dict[int, Any]:

        indices = self._scene.train_indices
        n = len(indices)
        started = time.time()
        dst = {i: self._create_structure_texture(f"structure_tensor_{i}") for i in indices}

        CHUNK = 16
        for base in range(0, n, CHUNK):
            batch = gpu.runtime.command_batch("structure_tensor_scene")
            for i in indices[base:base + CHUNK]:
                gpu.record_into(batch.encoder, self._target_textures[i], dst[i])
            gpu.runtime.submit(batch)
            gpu.runtime.wait_for_idle()
        print(f"[SAD] computed {n} fields on GPU (all-GPU, no readback, "
              f"chunk={CHUNK}) in {time.time() - started:.1f}s", flush=True)
        return dst

    def _create_structure_texture(self, label: str) -> Any:
        import slangpy as spy
        # rgba16_float (64-bit) instead of rgba32_float (128-bit): the field is
        # 3 trace/batch-normalized components (Sxx, Sxy, Syy) in ~[0,1] feeding a
        # coarse eigenvalue->eta gate (thresholds 0.1/1.0), so fp16 precision is
        # ample. Shader reads it as Texture2D<float4>.xyz unchanged. 2x smaller.
        return self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba16_float,
            width=self._scene.width, height=self._scene.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=label,
        )

    def _load_cache(self) -> np.ndarray | None:
        from trainer.backend.densify.structure_analysis import cache_path

        path = cache_path(
            self._args.scene, self._scene.width, self._scene.height)
        if not path.exists():
            return None

        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[SAD] loading cache {path.name} ({size_mb:.0f} MB)...", flush=True)
        started = time.time()
        with np.load(path) as data:
            camera_count = int(data["n_cams"])
            shape = tuple(data["fields"].shape[1:3])
            if (
                camera_count != self._scene.camera_count
                or shape != (self._scene.height, self._scene.width)
            ):
                print(f"[SAD] cache at {path} mismatches scene "
                      f"(cached n_cams={camera_count}, shape={shape}); recomputing",
                      flush=True)
                return None
            fields = np.array(data["fields"], copy=True)

        print(f"[SAD] loaded {self._scene.camera_count} cached "
              f"structure-tensor fields in {time.time() - started:.1f}s",
              flush=True)
        return fields

    def _create_sadgs_gpu_backend(self) -> Any | None:
        try:
            from trainer.backend.densify.structure_analysis_gpu import StructureTensorGPU
            gpu_backend = StructureTensorGPU(
                self._device, self._scene.width, self._scene.height)
            print(f"[SAD] no cache; structure fields for {self._scene.camera_count} "
                  "views come from the GPU...", flush=True)
            return gpu_backend
        except Exception as error:
            report_exception("[SAD] GPU structure-tensor setup failed; using CPU", error)
            print(f"[SAD] GPU structure-tensor unavailable ({error}); computing "
                  f"{len(self._scene.train_indices)} fields on CPU (slow)...", flush=True)
            return None

    def _field_for(self, camera_index: int, cached: np.ndarray | None,
                   gpu_computer: Any | None) -> np.ndarray:
        if cached is not None:
            return cached[camera_index]
        rgb = self._targets[camera_index][..., :3]
        if gpu_computer is not None:
            return gpu_computer.compute(rgb)
        from trainer.backend.densify.structure_analysis import compute_structure_tensor_field
        return compute_structure_tensor_field(rgb)

    def _upload(self, field: np.ndarray, label: str) -> Any:
        import slangpy as spy

        # rgba16_float (see _create_structure_texture): pack the 3 components as
        # fp16 with alpha=0. copy_from_numpy matches by byte size, so the array
        # dtype MUST be float16 for this format.
        packed = np.zeros(
            (self._scene.height, self._scene.width, 4), dtype=np.float16)
        packed[..., :3] = field
        if not self._resident:
            return packed       # streamed: the pipeline uploads it when used
        texture = self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba16_float,
            width=self._scene.width,
            height=self._scene.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=label,
        )
        texture.copy_from_numpy(np.ascontiguousarray(packed))
        return texture


class TrainingReporter:
    """Owns training log formatting, artifact schemas, and the live UI numbers."""

    def __init__(self, artifacts: RunArtifacts, args: argparse.Namespace,
                 scene: SceneContext, state: Session | None = None) -> None:
        self._artifacts = artifacts
        self._args = args
        self._scene = scene
        # Where the viewer reads its progress bar / metrics from. Always present
        # (ViewerSession owns one even under --no-viewer), so no null checks.
        self._state = state if state is not None else Session()
        self._started_at = 0.0
        self._last_time = 0.0
        self._last_iteration = 0
        self._rolling_loss: float | None = None

    def write_header(self) -> None:
        log = self._artifacts.log_txt
        log.write(f"# v2 training log — scene={self._args.scene}, "
                  f"backend={self._args.backend}\n")
        log.write(f"# iters={self._args.iters}, "
                  f"resolution={self._scene.width}x{self._scene.height}, "
                  f"n_cams={self._scene.camera_count}\n")
        log.write(f"# densify: every {self._args.densify_every} iters in "
                  f"[{self._args.densify_from}, {self._args.densify_to}]\n")

    def start(self, started_at: float) -> None:
        self._started_at = started_at
        self._last_time = started_at
        self._last_iteration = 0
        # Publishing the target iteration count is what makes the viewer's
        # training bar determinate (fraction = step / total). EVERY field is
        # published, not just the new ones: in the hub this reporter may follow a
        # previous run, whose loss/psnr/rate would otherwise still be on screen
        # until the new run's first log tick replaced them.
        self._state.publish_training(
            step=0, total=int(self._args.iters),
            gaussians=self._scene.initial_gaussian_count, elapsed_s=0.0,
            it_per_s=0.0, loss=None, psnr=None)
        self._state.log(f"training started: {self._args.iters} iters at "
                        f"{self._scene.width}x{self._scene.height}")

    def report_throughput(self, context: IterationContext) -> None:
        if context.step_number % 10 != 0:
            return
        now = time.time()
        rate = ((context.step_number - self._last_iteration)
                / max(now - self._last_time, 1e-9))
        elapsed = now - self._started_at
        print(f"iter {context.step_number}: train step done. It/s:{rate:.1f} "
              f"total training time:{elapsed // 60:.0f}min {elapsed % 60:.1f}sec")
        # Same cadence as the console line: advances the viewer's progress bar
        # and its it/s + ETA readouts without touching the per-iteration path.
        self._state.publish_training(step=context.step_number, it_per_s=rate,
                                     elapsed_s=elapsed)
        self._last_time = now
        self._last_iteration = context.step_number

    def report_iteration(self, *, context: IterationContext, pipeline: TrainingPipeline,
                         target: np.ndarray,
                         current_count: int) -> tuple[bool, int]:
        pipeline.render(context.camera_entry.camera)
        count = int(
            pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
        rendered = pipeline.read_output_image()[..., :3]

        if not np.all(np.isfinite(rendered)):
            nonfinite = int(np.sum(~np.isfinite(rendered)))
            message = (
                f"iter {context.step_number}: rendered image has {nonfinite} "
                f"non-finite pixels (out of {rendered.size}). Training aborted.")
            print(message)
            self._artifacts.log_txt.write(f"# ABORT: {message}\n")
            self._state.training_failed(message)  # the panel says so, not stdout
            self._state.notify("Training diverged", ERROR)
            self._state.log(message)
            return False, current_count

        clipped = np.clip(rendered, 0.0, 1.0)
        mse = float(np.mean((clipped - target) ** 2))
        self._rolling_loss = (
            mse if self._rolling_loss is None
            else 0.98 * self._rolling_loss + 0.02 * mse)
        psnr_value = psnr(clipped, target)
        elapsed = time.time() - self._started_at
        rate = context.step_number / max(elapsed, 1e-9)
        line = (
            f"iter {context.step_number:6d}  cam {context.camera_index:2d}  "
            f"loss={self._rolling_loss:.5f}  psnr={psnr_value:5.2f} dB  "
            f"lr_mean={context.learning_rates.mean:.2e}  n_g={count:6d}  "
            f"{rate:5.1f} it/s")
        print(line)
        self._state.log(line)
        self._state.publish_training(
            step=context.step_number, gaussians=count, it_per_s=rate,
            loss=self._rolling_loss, psnr=psnr_value, elapsed_s=elapsed)
        self._artifacts.log_txt.write(line + "\n")
        self._artifacts.log_json.write(json.dumps({
            "iter": context.step_number,
            "cam_idx": context.camera_index,
            "loss_rolling": self._rolling_loss,
            "loss_instant": mse,
            "psnr_instant": psnr_value,
            "lr_mean": context.learning_rates.mean,
            "n_gaussians": count,
            "it_per_s": rate,
            "elapsed_s": elapsed,
        }) + "\n")
        return True, count

    def report_densify(self, outcome: DensifyOutcome) -> None:
        """Serialize one densify tick's JSON + console diagnostics. Owns the
        densify artifact schema; the controller only produces the data."""
        self._artifacts.densify_json.write(json.dumps({
            "iter": outcome.iteration, "event": "densify",
            "kind": outcome.kind, **outcome.stats}) + "\n")

        pre_s = outcome.stats.get("pre_count")
        final_s = outcome.stats.get("final_count")
        if pre_s is not None and final_s is not None:
            line = (f"iter {outcome.iteration}: {outcome.kind:9s}| "
                    f"pre={pre_s} -> post={final_s}")
            print(line)
            self._state.log(line)

    def report_opacity_reset(self, iteration: int) -> None:
        self._artifacts.densify_json.write(json.dumps({
            "iter": iteration, "event": "opacity_reset"}) + "\n")
        print(f"iter {iteration}: opacity reset")

    def report_morton_sort(self, iteration: int) -> None:
        self._artifacts.densify_json.write(json.dumps({
            "iter": iteration, "event": "morton_sort"}) + "\n")

    def report_complete(self, iterations: int, wall_time: float) -> None:
        rate = iterations / max(wall_time, 1e-9)
        line = (f"[train] loop done: {iterations} iters in {wall_time:.1f}s "
                f"({rate:.1f} it/s)")
        print(line)
        self._state.log(line)
        self._state.publish_training(step=iterations, it_per_s=rate,
                                     elapsed_s=wall_time)
        self._artifacts.log_json.write(json.dumps({
            "event": "train_complete",
            "iters": iterations,
            "train_wall_s": wall_time,
            "it_per_s": rate,
        }) + "\n")


class PsNRPlotWriter:
    """Writes the final PSNR-curve JSON and best-effort PNG."""

    @staticmethod
    def write(*, enabled: bool, records: list[dict], args: argparse.Namespace,
              paths: RunPaths, run_name: str, scene: SceneContext) -> None:
        if not enabled:
            return
        curve = {
            "scene": args.scene,
            "run_name": run_name,
            "eval_holdout": bool(args.eval_holdout),
            "eval_holdout_stride": scene.holdout_stride,
            "n_eval_cams": len(scene.test_indices),
            "psnr_curve_every": args.psnr_curve_every,
            "iters": [record["iter"] for record in records],
            "psnr_srgb": [record["psnr_srgb"] for record in records],
            "n_gaussians": [record["n_gaussians"] for record in records],
        }
        (paths.root / "psnr_curve.json").write_text(json.dumps(curve, indent=2))
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            figure, axis = plt.subplots(figsize=(10, 6), dpi=120)
            axis.plot(
                curve["iters"], curve["psnr_srgb"], "-o",
                ms=3, lw=1.6, color="#1f77b4")
            axis.set_xlabel("iteration")
            axis.set_ylabel("test PSNR sRGB (dB)")
            scene_name = Path(args.scene).name or args.scene
            axis.set_title(f"PSNR vs iteration — {scene_name} ({run_name})")
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            figure.savefig(paths.root / "psnr_curve.png")
            plt.close(figure)
            print(f"[psnr-curve] wrote {paths.root / 'psnr_curve.json'} + "
                  f"psnr_curve.png ({len(curve['iters'])} points)")
        except Exception as error:  # noqa: BLE001 - plotting is best-effort
            report_exception("[psnr-curve] PNG creation failed", error)
            print(f"[psnr-curve] wrote psnr_curve.json but the PNG failed "
                  f"({error}); `.venv/Scripts/python.exe -m pip install "
                  "matplotlib` to enable it.")

def load_scene_context(args: argparse.Namespace) -> SceneContext:
    colmapScene : ColmapDataset = load_colmap(
        args.scene,
        width=args.width,
        height=args.height,
        downsample=args.downsample,
        init_opacity=args.init_opacity,
    )

    camera_count = len(colmapScene.images)
    initial_count = colmapScene.initial_points["means"].shape[0]
    width, height = colmapScene.images[0].camera.image_size

    print(f"loaded {camera_count} cameras, {initial_count} init points, scene_extent={colmapScene.scene_extent:.3f}")

    stride, test_indices, train_indices = resolve_holdout_split(args, camera_count)
    morton_min, morton_max = compute_morton_bounds(colmapScene.initial_points["means"])

    print(f"schedule: densify [{args.densify_from}, {args.densify_to}] every {args.densify_every}")

    return SceneContext(
        colmapScene=colmapScene,
        width=width,
        height=height,
        camera_count=camera_count,
        initial_gaussian_count=initial_count,
        extent=colmapScene.scene_extent,
        train_indices=train_indices,
        test_indices=test_indices,
        holdout_stride=stride,
        morton_min=morton_min,
        morton_max=morton_max,
    )

def create_training_device(args: argparse.Namespace) -> Any:
    specialization = ShaderSpecialization.from_args(args)
    # Vulkan only records debug object names (buffer/texture labels) when the
    # VK_EXT_debug_utils extension is enabled, and sgl enables it only when debug
    # layers are on. Default that on under external capture so the labels set in
    # buffers.allocate() / create_texture actually reach Nsight; None = auto.
    debug_names = getattr(args, "nsight_debug_names", None)
    if debug_names is None:
        debug_names = bool(getattr(args, "capture_external", False))
    device = create_device(
        backend=args.backend,
        enable_debug=debug_names,
        shader_debug_info=args.profile_run,
        defines=specialization.compiler_defines(),
    )
    if args.profile_run:
        print("[profile-run] shaders compiled with source-level debug info "
              "(SPIR-V NonSemantic.Shader.DebugInfo) for Nsight correlation.")
    if debug_names:
        print("[profile-run] Vulkan debug layers ON so resource debug names "
              "(VK_EXT_debug_utils) reach Nsight's Object Browser / Memory view.")
    return device

def create_gpu_queue_lock(*, viewer_enabled: bool) -> Any:
    return threading.RLock() if viewer_enabled else None


def _resolve_position_lr_max_steps(cli_value: int, iters: int) -> int:
    """Horizon for the exponential mean-LR decay (3DGS position_lr_max_steps).

    ``--position-lr-max-steps 0`` (the default) => FOLLOW ``--iters``: the decay
    then always lands on lr_final exactly at the end of the run. Pinning it
    independently is still possible (a positive value wins), but it is not a choice
    the hub's TRAINING SETUP offers — a horizon that disagrees with the run length
    silently either freezes the means early or never reaches the final LR.
    """
    return int(iters) if int(cli_value) <= 0 else int(cli_value)


def _resolve_sh_band_interval(cli_interval: int, iters: int, sh_degree: int) -> int:
    """Band-activation interval for the SH schedule.

    ``--sh-band-interval 0`` (the default) => AUTO: scale to the run length so all
    SH bands activate within the first ~half of training and actually get trained.
    The classic 3DGS constant (1000) assumes a 30k-iter run (deg-3 by iter 3000,
    then 27k iters to refine); on a 3k run it leaves degree 3 with ~1 iter of
    training (measured: SH3 ~= SH0). Auto puts the last band on by iters//2:
    interval = max(1, iters // (2 * sh_degree)). A positive CLI value overrides
    (explicit control / reproducing the reference constant).
    """
    if cli_interval > 0:
        return cli_interval
    if sh_degree <= 0:
        return max(1, iters)          # unused (schedule returns 0), keep non-zero
    return max(1, iters // (2 * sh_degree))

def create_pipeline(*, args: argparse.Namespace, device: Any, gpu_lock: Any,
                    scene: SceneContext, max_splats: int, lambda_l2: float,
                    lambda_ssim: float,
                    initial_state: dict | None = None) -> TrainingPipeline:
    return TrainingPipeline(
        device=device,
        width=scene.width,
        height=scene.height,
        max_splats=max_splats,
        gpu_lock=gpu_lock,
        raster_config=RasterConfig(
            z_near_cull=args.z_near_cull,
            cluster_cull=args.cluster_cull,
            mip_enabled=args.mip_splatting,
            kernel_size=(args.mip_kernel_size if args.mip_splatting else 0.3),
        ),
        loss_config=LossConfig(
            alpha_weight=args.w_alpha,
            ssim_weight=lambda_ssim,
            l2_weight=lambda_l2,
        ),
        regularization_config=RegularizationConfig(
            opacity_weight=args.opacity_reg,
            scale_weight=args.scale_reg,
        ),
        appearance_config=AppearanceConfig(
            sh_degree=args.sh_degree,
            band_interval=_resolve_sh_band_interval(
                args.sh_band_interval, args.iters, args.sh_degree),
        ),
        stbn_config=resolve_stbn_config(args),
        adaptive_config=resolve_adaptive_config(args),
        initial_state=initial_state,
        densify_mode=DensificationMode.from_args(args),
    )


def main() -> None:
    app = TrainingApplication.from_cli()
    with ErrorLog(app.paths.root / "errors.log") as errors:
        app.error_log = errors
        try:
            if app.args.hub:
                from trainer.frontend.hub import TrainingHub
                TrainingHub(app).run()
            else:
                app.start_training()
        except Exception as error:
            state = app.viewer.state if app.viewer is not None else None
            report_exception("Application failed", error, state)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
