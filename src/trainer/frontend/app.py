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
from trainer.common.color import srgb_to_linear
from trainer.config.config import (
    AdamLRs,
    CompositorConfig,
    LossConfig,
    RasterConfig,
    RegularizationConfig,
)
from trainer.config.config_types import (
    DensificationMode,
    ScheduleConfig,
    validate_training_config,
)
from trainer.backend.compute.device import create_device
from trainer.common.metrics import psnr
from trainer.backend.render.pipeline import Pipeline
from trainer.frontend.cli import build_parser
from trainer.frontend.densification import DensificationController, DensifyOutcome
from trainer.frontend.evaluation.evaluation import (
    eval_test_psnr_srgb,
    render_all_cameras,
    run_eval_sweep,
)
from trainer.frontend.lr_schedule import (
    ExtinctionHardnessSchedule,
    MeanLearningRateSchedule,
)
from trainer.frontend.reporting.outputs import RunArtifacts
from trainer.frontend.evaluation.post_training import run_post_training
from trainer.frontend.queue_limiter import QueueLimiter
from trainer.frontend.training_preflight import (
    build_densify_configs,
    compute_morton_bounds,
    resolve_holdout_split,
    resolve_loss_and_lrs,
)
from trainer.frontend.schedule import TrainingSchedule
from trainer.frontend.viewer.viewer_control import ViewerSession

# The thin scripts/running/v2_train.py entrypoint installs scripts/running and
# scripts/debug on sys.path before importing this module.
from _v2_profiling import Profiler  # pyright: ignore[reportMissingImports]
from _v2_sampling import CameraSampler  # pyright: ignore[reportMissingImports]


# ---------------------------------------------------------------------------
# Application data
# ---------------------------------------------------------------------------


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
    scene: Any
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
class IterationContext:
    number: int
    camera_index: int
    camera_entry: Any
    learning_rates: AdamLRs
    events: Any


@dataclass(frozen=True)
class TrainingResult:
    iterations: int
    wall_time_seconds: float
    psnr_curve: list[dict]
    started_at: float


@dataclass(frozen=True)
class ShaderSpecialization:
    """Compile-time shader policy resolved from training configuration.

    Pixel stride must remain compile-time because the backward shader uses it
    to size staged arrays and specialize loop/index arithmetic. The application
    deals in the semantic value; only this adapter knows the Slang macro name
    and unsigned-literal spelling.
    """

    backward_pixel_stride: int

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ShaderSpecialization":
        return cls(backward_pixel_stride=args.backward_pixel_stride)

    def compiler_defines(self) -> dict[str, str] | None:
        if self.backward_pixel_stride == 1:
            return None
        return {
            "BACKWARD_PIXEL_STRIDE": f"{self.backward_pixel_stride}u",
        }


@dataclass
class TrainingRuntime:
    pipeline: Pipeline
    viewer: ViewerSession
    profiler: Profiler
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
        self.profiler = Profiler(args)

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
        self.pipeline: Pipeline | None = None
        self.viewer: ViewerSession | None = None
        self.mip_diagnostics: MipFilterDiagnostics | None = None
        self.opacity_reporter: OpacityBundleReporter | None = None
        self.target_textures: list[Any] = []
        self.structure_textures: list[Any] = []
        self.mean_lr: MeanLearningRateSchedule | None = None
        self.extinction: ExtinctionHardnessSchedule | None = None

    @classmethod
    def from_cli(cls) -> "TrainingApplication":
        return cls(build_parser().parse_args())

    def run(self) -> None:
        self.prepare()
        result = self.train()
        self.finish(result)

    def prepare(self) -> None:
        self._prepare_profiler()
        self._load_and_validate_experiment()
        self._prepare_targets()
        self._create_rendering_system()
        self._prepare_optional_features()
        self._start_integrations()
        self._create_training_services()

    def train(self) -> TrainingResult:
        loop = TrainingLoop(
            args=self.args,
            scene=self._scene,
            targets=self._targets,
            runtime=self._runtime,
            densify_mode=self._densify_mode,
            densify_config=self.densify_config,
            base_lrs=self._base_lrs,
            target_textures=self.target_textures,
            structure_textures=self.structure_textures,
            mip_diagnostics=self._mip_diagnostics,
            mean_lr=self._mean_lr,
            extinction=self._extinction,
            paths=self.paths,
        )
        return loop.run()

    def finish(self, result: TrainingResult) -> None:
        runtime = self._runtime
        runtime.pipeline.runtime.wait_for_idle()

        self._write_final_test_psnr(result)
        runtime.profiler.close()
        runtime.viewer.hold_after_training(runtime.pipeline)
        runtime.artifacts.close_all()

        PsNRPlotWriter.write(
            enabled=self.args.psnr_curve,
            records=result.psnr_curve,
            args=self.args,
            paths=self.paths,
            run_name=self.run_name,
            scene=self._scene,
        )

        if self.args.profile_run:
            print("[profile-run] finished; skipping final metrics / PLY export.")
            return

        scene = self._scene
        run_post_training(
            pipeline=runtime.pipeline,
            scene=scene.scene,
            args=self.args,
            run_dir=self.paths.root,
            cams_dir=self.paths.cameras,
            targets_srgb=self._targets.srgb,
            targets_linear=self._targets.linear,
            test_indices=scene.test_indices,
            stride=scene.holdout_stride,
            n_cams=scene.camera_count,
            train_wall_s=result.wall_time_seconds,
            completed_iters=result.iterations,
            t_start=result.started_at,
        )

    def _prepare_profiler(self) -> None:
        # configure_session may force Vulkan, disable previews, and disable the
        # viewer; it must run before config resolution and device creation.
        self.profiler.configure_session()
        self.profiler.inject()

    def _load_and_validate_experiment(self) -> None:
        self.scene_context = load_scene_context(self.args)
        scene = self._scene

        self.densify_mode = validate_training_config(
            self.args,
            n_cams=scene.camera_count,
            n_train=len(scene.train_indices),
            n_test=len(scene.test_indices),
        )
        
        self.densify_config, self.structure_config = build_densify_configs(self.args, scene.extent)
        self.effective_lambda_l2, self.effective_lambda_ssim, self.base_lrs = resolve_loss_and_lrs(self.args)

        self.opacity_reporter = OpacityBundleReporter(
            args=self.args,
            densify_config=self.densify_config,
            structure_config=self.structure_config,
        )
        
        self.opacity_reporter.report_startup(self._densify_mode)
        self._write_resolved_config()

    def _prepare_targets(self) -> None:
        scene = self._scene
        self.targets = TargetDataset(
            scene=scene.scene,
            width=scene.width,
            height=scene.height,
            use_alpha=self.args.w_alpha > 0,
        )
        self.targets.write_ground_truths(self.paths.cameras)

    def _create_rendering_system(self) -> None:
        scene = self._scene
        self.device = create_training_device(self.args)
        self.gpu_lock = create_gpu_queue_lock(viewer_enabled=not self.args.no_viewer)
        self.pipeline = create_pipeline(
            args=self.args,
            device=self.device,
            gpu_lock=self.gpu_lock,
            scene=scene,
            lambda_l2=self.effective_lambda_l2,
            lambda_ssim=self.effective_lambda_ssim,
        )
        initialize_pipeline(
            pipeline=self._pipeline,
            initial_points=scene.scene.initial_points,
            densify_mode=self._densify_mode,
            use_flat_bucket_backward=self.args.flat_bucket_backward,
        )

        self.mip_diagnostics = MipFilterDiagnostics(
            enabled=self.args.mip_splatting,
            pipeline=self._pipeline,
        )
        train_cameras = [scene.scene.images[i].camera for i in scene.train_indices]
        self.mip_diagnostics.initialize(
            train_cameras=train_cameras,
            kernel_size=self.args.mip_kernel_size,
        )

        self.viewer = ViewerSession.create(
            enabled=not self.args.no_viewer,
            device=self.device,
            width=scene.width,
            height=scene.height,
            max_splats=self.args.max_splats,
            fovy=getattr(self.args, "fovy", 45.0),
            gpu_lock=self.gpu_lock,
            pipeline=self._pipeline,
        )

    def _prepare_optional_features(self) -> None:
        pipeline = self._pipeline
        self.target_textures = [
            pipeline.create_target_texture(target) for target in self._targets.srgb
        ]
        self.structure_textures = StructureTextureProvider(
            args=self.args,
            device=self.device,
            scene=self._scene,
            targets=self._targets.srgb,
            densify_mode=self._densify_mode,
        ).load()

    def _start_integrations(self) -> None:
        self._viewer.start()
        self.profiler.open_window(
            self._pipeline, self._scene.width, self._scene.height)

    def _create_training_services(self) -> None:
        scene = self._scene
        artifacts = RunArtifacts.create(self.paths.root, self.args)
        reporter = TrainingReporter(artifacts, self.args, scene)
        reporter.write_header()

        diag_module = None
        diag_grad_internal = None
        if artifacts.densify_diag_json is not None:
            import densify_diag as diag_module  # pyright: ignore[reportMissingImports]
            diag_grad_internal = (
                self.args.sad_grad_threshold * scene.width * scene.height * 3.0)

        self.mean_lr = MeanLearningRateSchedule(
            lr_init=1.6e-4 * scene.extent,
            lr_final=1.6e-6 * scene.extent,
            max_steps=self.args.position_lr_max_steps,
        )
        self.extinction = ExtinctionHardnessSchedule(
            target=self.args.extinction_hardness,
            warmup=self.args.extinction_hardness_warmup,
        )
        queue_limiter = QueueLimiter(self._pipeline.runtime, max_depth=1)
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
            extinction=self._extinction,
            queue_limiter=queue_limiter,
            collect_diagnostics=artifacts.densify_diag_json is not None,
            diag_module=diag_module,
            diag_grad_internal=diag_grad_internal,
            print_opacity_bundle=self._opacity_reporter.report,
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

    def _write_resolved_config(self) -> None:
        scene = self._scene
        base_lrs = self._base_lrs
        config = {
            "scene": self.args.scene,
            "iters": self.args.iters,
            "width": scene.width,
            "height": scene.height,
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
            "n_cams": scene.camera_count,
            "n_init_points": scene.initial_gaussian_count,
            "cli_args": dict(vars(self.args)),
            "resolved": {
                "densify_mode": self._densify_mode.value,
                "width": scene.width,
                "height": scene.height,
                "n_cams": scene.camera_count,
                "n_init_points": scene.initial_gaussian_count,
                "n_train": len(scene.train_indices),
                "n_test": len(scene.test_indices),
                "holdout_stride": scene.holdout_stride,
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
        if not self.args.final_test_psnr or self.args.profile_run:
            return
        scene = self._scene
        psnr, count = eval_test_psnr_srgb(
            self._pipeline, scene.scene, scene.test_indices, self._targets.srgb)
        (self.paths.root / "test_psnr.json").write_text(json.dumps({
            "psnr_srgb": psnr,
            "n_gaussians": count,
            "n_test_cams": len(scene.test_indices),
            "train_wall_s": result.wall_time_seconds,
            "iters": result.iterations,
            "flat_bucket_backward": bool(self.args.flat_bucket_backward),
        }, indent=2))
        print(f"[final-test-psnr] TEST PSNR sRGB = {psnr:.3f} dB over "
              f"{len(scene.test_indices)} held-out cams  (n_gaussians={count})")

    @property
    def _scene(self) -> SceneContext:
        assert self.scene_context is not None
        return self.scene_context

    @property
    def _targets(self) -> "TargetDataset":
        assert self.targets is not None
        return self.targets

    @property
    def _runtime(self) -> TrainingRuntime:
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
    def _pipeline(self) -> Pipeline:
        assert self.pipeline is not None
        return self.pipeline

    @property
    def _viewer(self) -> ViewerSession:
        assert self.viewer is not None
        return self.viewer

    @property
    def _mip_diagnostics(self) -> "MipFilterDiagnostics":
        assert self.mip_diagnostics is not None
        return self.mip_diagnostics

    @property
    def _opacity_reporter(self) -> "OpacityBundleReporter":
        assert self.opacity_reporter is not None
        return self.opacity_reporter

    @property
    def _mean_lr(self) -> MeanLearningRateSchedule:
        assert self.mean_lr is not None
        return self.mean_lr

    @property
    def _extinction(self) -> ExtinctionHardnessSchedule:
        assert self.extinction is not None
        return self.extinction


# ---------------------------------------------------------------------------
# Training-loop orchestration
# ---------------------------------------------------------------------------


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
        target_textures: list[Any],
        structure_textures: list[Any],
        mip_diagnostics: "MipFilterDiagnostics",
        mean_lr: MeanLearningRateSchedule,
        extinction: ExtinctionHardnessSchedule,
        paths: RunPaths,
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
        self.mip_diagnostics = mip_diagnostics
        self.mean_lr = mean_lr
        self.extinction = extinction
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
        self.iteration = 0
        self.gaussian_count = scene.initial_gaussian_count
        self.eval_time_total = 0.0
        self.psnr_curve_records: list[dict] = []
        self.started_at = 0.0

    def run(self) -> TrainingResult:
        self._begin()
        while self._should_continue():
            context = self._next_iteration()
            self._prepare_iteration(context)
            self._capture_if_requested(context)
            self._execute_training_step(context)
            self._apply_mutations(context)
            should_continue = self._produce_observations(context)
            self._present_profile_frame_if_enabled()
            if not should_continue:
                break
        return self._complete()

    def _begin(self) -> None:
        if self.args.wait_for_viewer:
            self.runtime.viewer.wait_for_start()
        self.runtime.viewer.set_training_active(True)
        self.started_at = time.time()
        self.runtime.reporter.start(self.started_at)

    def _should_continue(self) -> bool:
        return (
            self.iteration < self.args.iters
            and not self.runtime.profiler.should_stop()
        )

    def _next_iteration(self) -> IterationContext:
        self.iteration += 1
        camera_index = self.sampler.next()
        camera_entry = self.scene.scene.images[camera_index]
        learning_rates = AdamLRs(
            mean=self.mean_lr.at(self.iteration - 1),
            scale=self.base_lrs.scale,
            rot=self.base_lrs.rot,
            opacity=self.base_lrs.opacity,
            color=self.base_lrs.color,
        )
        return IterationContext(
            number=self.iteration,
            camera_index=camera_index,
            camera_entry=camera_entry,
            learning_rates=learning_rates,
            events=self.schedule.events_at(self.iteration),
        )

    def _prepare_iteration(self, context: IterationContext) -> None:
        self.runtime.pipeline.set_target_texture(
            self.target_textures[context.camera_index])
        if context.number == 1:
            print(f"[determinism] iter 1 cam_idx={context.camera_index} "
                  "(should match across same-seed runs)")
        self.runtime.queue_limiter.before_submit()

    def _capture_if_requested(self, context: IterationContext) -> None:
        captured = self.runtime.profiler.run_capture_if_due(
            self.runtime.pipeline,
            context.number,
            context.camera_entry,
            context.camera_index,
            self.runtime.queue_limiter,
        )
        if captured:
            self.runtime.artifacts.close_all()
            raise SystemExit(0)

    def _execute_training_step(self, context: IterationContext) -> None:
        structure_texture = None
        structure_kwargs = None
        if self._should_accumulate_structure(context):
            structure_texture = self.structure_textures[context.camera_index]
            structure_kwargs = {
                "per_view_grad_thresh": (
                    self.args.sad_freq_grad_threshold
                    * self.scene.width * self.scene.height * 3.0),
            }

        pipeline = self.runtime.pipeline
        pipeline.set_extinction_hardness(self.extinction.at(context.number))
        pipeline.train_step(
            context.camera_entry.camera,
            lrs=context.learning_rates,
            structure_tensor=structure_texture,
            structure_kwargs=structure_kwargs,
        )
        self.runtime.queue_limiter.record(pipeline.last_submit_id)
        self._stream_viewer_snapshot_if_due(context)

    def _apply_mutations(self, context: IterationContext) -> None:
        events = context.events
        pipeline = self.runtime.pipeline

        if events.densify:
            outcome = self.runtime.densifier.run(
                context.number, events, self.gaussian_count)
            self.gaussian_count = outcome.count
            self.runtime.reporter.report_densify(outcome)

        if events.opacity_reset:
            pipeline.reset_opacity(self.densify_config)
            self.runtime.reporter.report_opacity_reset(context.number)

        if events.morton:
            pipeline.morton_sort(
                self.scene.morton_min,
                self.scene.morton_max,
                carry_eta=(self.densify_mode == DensificationMode.STRUCTURE_AWARE),
            )
            pipeline.clear_tracking()
            if pipeline.last_submit_id:
                self.runtime.queue_limiter.record(pipeline.last_submit_id)
            self.runtime.reporter.report_morton_sort(context.number)

        if events.mip_periodic:
            self.mip_diagnostics.refresh("periodic", context.number)

    def _produce_observations(self, context: IterationContext) -> bool:
        self.runtime.reporter.report_throughput(context)

        if context.number > 1 and (context.events.log or context.events.preview):
            ok, self.gaussian_count = self.runtime.reporter.report_iteration(
                context=context,
                pipeline=self.runtime.pipeline,
                target=self.targets.srgb[context.camera_index][..., :3],
                current_count=self.gaussian_count,
            )
            if not ok:
                return False

        if context.number > 1 and context.events.render_all:
            render_all_cameras(
                self.runtime.pipeline,
                self.scene.scene,
                context.number,
                self.paths.cameras,
            )

        self._record_psnr_curve_if_due(context)
        self._run_evaluation_if_due(context)
        return True

    def _should_accumulate_structure(self, context: IterationContext) -> bool:
        return (
            self.densify_mode == DensificationMode.STRUCTURE_AWARE
            and self.args.densify_from < context.number <= self.args.densify_to
            and context.number % self.args.sad_accum_every == 0
        )

    def _present_profile_frame_if_enabled(self) -> None:
        if self.args.profile_run:
            self.runtime.profiler.present_frame(
                self.runtime.pipeline.last_submit_id)

    def _stream_viewer_snapshot_if_due(self, context: IterationContext) -> None:
        if context.number % self.args.stream_every == 0:
            self.runtime.viewer.push_snapshot(self.runtime.pipeline)

    def _record_psnr_curve_if_due(self, context: IterationContext) -> None:
        stream = self.runtime.artifacts.psnr_curve_json
        if stream is None or not context.events.psnr_curve:
            return
        started = time.time()
        psnr, count = eval_test_psnr_srgb(
            self.runtime.pipeline,
            self.scene.scene,
            self.scene.test_indices,
            self.targets.srgb,
        )
        record = {
            "iter": context.number,
            "psnr_srgb": psnr,
            "n_gaussians": count,
        }
        self.psnr_curve_records.append(record)
        stream.write(json.dumps(record) + "\n")
        self.eval_time_total += time.time() - started
        print(f"[psnr-curve] iter {context.number:6d}  psnr_srgb={psnr:5.2f}  "
              f"n_g={count} (pre-densify)")

    def _run_evaluation_if_due(self, context: IterationContext) -> None:
        stream = self.runtime.artifacts.eval_curve_json
        if stream is None or not context.events.eval:
            return
        self.eval_time_total = run_eval_sweep(
            pipeline=self.runtime.pipeline,
            scene=self.scene.scene,
            it=context.number,
            test_indices=self.scene.test_indices,
            targets_srgb=self.targets.srgb,
            targets_linear=self.targets.linear,
            cams_dir=self.paths.cameras,
            eval_curve_json=stream,
            bin_viz_json=self.runtime.artifacts.bin_viz_json,
            t_start=self.started_at,
            eval_time_total=self.eval_time_total,
            eval_holdout=bool(self.args.eval_holdout),
        )

    def _complete(self) -> TrainingResult:
        self.runtime.pipeline.runtime.wait_for_idle()
        wall_time = time.time() - self.started_at
        self.runtime.pipeline.set_extinction_hardness(self.extinction.target)
        self.runtime.reporter.report_complete(self.iteration, wall_time)
        return TrainingResult(
            iterations=self.iteration,
            wall_time_seconds=wall_time,
            psnr_curve=self.psnr_curve_records,
            started_at=self.started_at,
        )



class TargetDataset:
    """Owns target loading, lazy linear conversion, and ground-truth images."""

    def __init__(self, scene: Any, width: int, height: int, use_alpha: bool) -> None:
        self._scene = scene
        self._width = width
        self._height = height
        self._use_alpha = use_alpha
        self._srgb = [
            load_colmap_image(
                entry,
                scene.image_dir,
                width,
                height,
                load_alpha=use_alpha,
            )
            for entry in scene.images
        ]
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
        for camera_index, target in enumerate(self._srgb):
            camera_dir = cameras_dir / f"cam_{camera_index:02d}"
            camera_dir.mkdir(parents=True, exist_ok=True)
            path = camera_dir / "gt.png"
            if path.exists():
                continue
            if self._use_alpha and target.shape[-1] == 4:
                pixels = (target * 255 + 0.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(pixels, mode="RGBA").save(path)
            else:
                pixels = (target[..., :3] * 255 + 0.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(pixels, mode="RGB").save(path)


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

    def report_startup(self, mode: DensificationMode) -> None:
        if mode != DensificationMode.STRUCTURE_AWARE:
            return
        self.report("startup")
        settings = self.settings()
        if (
            not settings["opacity_cap_enabled"]
            or settings["structure_min_opacity"] < 0.1
        ):
            print(
                "[opacity-bundle] WARNING: bundle is INACTIVE at reference "
                "strength — the 0.8 cap and/or 0.1 floor will NOT run. To match "
                "SADGS pass: --sad-structure-min-opacity 0.1 --sad-opacity-cap "
                "0.8 --sad-relative-reset. Running as configured.",
                flush=True,
            )


class MipFilterDiagnostics:
    """Owns Mip-Splatting camera setup, refreshes, and buffer diagnostics."""

    def __init__(self, *, enabled: bool, pipeline: Pipeline) -> None:
        self._enabled = enabled
        self._pipeline = pipeline

    def initialize(self, *, train_cameras: list[Any], kernel_size: float) -> None:
        if not self._enabled:
            return
        self._pipeline.set_cameras(train_cameras)
        self._pipeline.compute_3d_filter()
        print(f"[mip] enabled: kernel_size={kernel_size}, 3D filter over "
              f"{len(train_cameras)} train cameras", flush=True)
        self.log("init")

    def refresh(self, tag: str, iteration: int | None = None) -> None:
        if not self._enabled:
            return
        self._pipeline.compute_3d_filter()
        self.log(tag, iteration)

    def log(self, tag: str, iteration: int | None = None) -> None:
        if not self._enabled:
            return
        buffers = self._pipeline.buffers
        count = int(
            buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
        if count == 0:
            print(f"[filter3d] {tag}: no active gaussians", flush=True)
            return
        values = np.frombuffer(
            buffers.filter_3D.to_numpy().tobytes(), dtype=np.float32
        )[:count].astype(np.float64)
        iteration_text = f"it={iteration} " if iteration is not None else ""
        print(
            f"[filter3d] {tag} {iteration_text}n={count} "
            f"min={values.min():.6g} mean={values.mean():.6g} "
            f"max={values.max():.6g} "
            f"nonzero={np.count_nonzero(values) / count:.3f} "
            f"sum={values.sum():.6g}",
            flush=True,
        )


class StructureTextureProvider:
    """Hides structure cache loading, computation, packing, and GPU upload."""

    def __init__(self, *, args: argparse.Namespace, device: Any,
                 scene: SceneContext, targets: list[np.ndarray],
                 densify_mode: DensificationMode) -> None:
        self._args = args
        self._device = device
        self._scene = scene
        self._targets = targets
        self._densify_mode = densify_mode

    def load(self) -> list[Any]:
        if self._densify_mode != DensificationMode.STRUCTURE_AWARE:
            return []

        cached_fields = self._load_cache()
        gpu_computer = None
        if cached_fields is None:
            gpu_computer = self._create_gpu_computer()
        else:
            print(f"[SAD] uploading {self._scene.camera_count} "
                  "structure-tensor textures to GPU...", flush=True)

        textures = []
        started = time.time()
        log_stride = max(1, self._scene.camera_count // 6)
        for camera_index in range(self._scene.camera_count):
            field = self._field_for(camera_index, cached_fields, gpu_computer)
            textures.append(self._upload(field, f"structure_tensor_{camera_index}"))
            if (
                (camera_index + 1) % log_stride == 0
                or camera_index == self._scene.camera_count - 1
            ):
                print(f"[SAD]   {camera_index + 1:3d}/{self._scene.camera_count} "
                      f"({time.time() - started:.1f}s elapsed)", flush=True)

        if cached_fields is None:
            method = "GPU" if gpu_computer is not None else "CPU"
            print(f"[SAD] computed {self._scene.camera_count} fields on {method} "
                  f"in {time.time() - started:.1f}s", flush=True)
        return textures

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

    def _create_gpu_computer(self) -> Any | None:
        try:
            from trainer.backend.densify.structure_analysis_gpu import StructureTensorGPU
            computer = StructureTensorGPU(
                self._device, self._scene.width, self._scene.height)
            print(f"[SAD] no cache; computing {self._scene.camera_count} fields "
                  "on GPU...", flush=True)
            return computer
        except Exception as error:  # noqa: BLE001 - GPU failure selects CPU fallback
            print(f"[SAD] GPU structure-tensor unavailable ({error}); computing "
                  f"{self._scene.camera_count} fields on CPU (slow)...", flush=True)
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

        packed = np.zeros(
            (self._scene.height, self._scene.width, 4), dtype=np.float32)
        packed[..., :3] = field
        texture = self._device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=self._scene.width,
            height=self._scene.height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=label,
        )
        texture.copy_from_numpy(np.ascontiguousarray(packed))
        return texture


class TrainingReporter:
    """Owns training log formatting and artifact schemas."""

    def __init__(self, artifacts: RunArtifacts, args: argparse.Namespace,
                 scene: SceneContext) -> None:
        self._artifacts = artifacts
        self._args = args
        self._scene = scene
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

    def report_throughput(self, context: IterationContext) -> None:
        if context.number % 10 != 0:
            return
        now = time.time()
        rate = ((context.number - self._last_iteration)
                / max(now - self._last_time, 1e-9))
        elapsed = now - self._started_at
        print(f"iter {context.number}: train step done. It/s:{rate:.1f} "
              f"total training time:{elapsed // 60:.0f}min {elapsed % 60:.1f}sec")
        self._last_time = now
        self._last_iteration = context.number

    def report_iteration(self, *, context: IterationContext, pipeline: Pipeline,
                         target: np.ndarray,
                         current_count: int) -> tuple[bool, int]:
        pipeline.render(context.camera_entry.camera)
        count = int(
            pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
        rendered = pipeline.read_output_image()[..., :3]

        if not np.all(np.isfinite(rendered)):
            nonfinite = int(np.sum(~np.isfinite(rendered)))
            message = (
                f"iter {context.number}: rendered image has {nonfinite} "
                f"non-finite pixels (out of {rendered.size}). Training aborted.")
            print(message)
            self._artifacts.log_txt.write(f"# ABORT: {message}\n")
            return False, current_count

        clipped = np.clip(rendered, 0.0, 1.0)
        mse = float(np.mean((clipped - target) ** 2))
        self._rolling_loss = (
            mse if self._rolling_loss is None
            else 0.98 * self._rolling_loss + 0.02 * mse)
        psnr_value = psnr(clipped, target)
        elapsed = time.time() - self._started_at
        rate = context.number / max(elapsed, 1e-9)
        line = (
            f"iter {context.number:6d}  cam {context.camera_index:2d}  "
            f"loss={self._rolling_loss:.5f}  psnr={psnr_value:5.2f} dB  "
            f"lr_mean={context.learning_rates.mean:.2e}  n_g={count:6d}  "
            f"{rate:5.1f} it/s")
        print(line)
        self._artifacts.log_txt.write(line + "\n")
        self._artifacts.log_json.write(json.dumps({
            "iter": context.number,
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

        rec = outcome.diag_record
        if rec is not None:
            self._artifacts.densify_diag_json.write(json.dumps(rec) + "\n")
            fo = rec.get("fanout", {})
            gate = rec.get("eta_gate", {})
            sp = rec.get("spatial_post", {})
            print(f"  [diag] {outcome.kind:9s} {rec['pre_count']}->{rec['post_count']} "
                  f"(x{rec['growth_factor']}) | "
                  f"trig={gate.get('frac_triggered_of_seen')} "
                  f"prune={gate.get('frac_prune_of_seen')} "
                  f"prod_p90={fo.get('prod_p90')} prod_max={fo.get('prod_max')} "
                  f"| gini={sp.get('gini')} top1%={sp.get('top1pct_cell_share')}")
            if outcome.opacity_assert is not None:
                amin, amax, msgs = outcome.opacity_assert
                print(f"  [opacity-bundle assert] iter {outcome.iteration}: "
                      f"min={amin:.4f} max={amax:.4f} | " + " | ".join(msgs))

        pre_s = outcome.stats.get("pre_count")
        final_s = outcome.stats.get("final_count")
        if pre_s is not None and final_s is not None:
            print(f"iter {outcome.iteration}: {outcome.kind:9s}| "
                  f"pre={pre_s} -> post={final_s}")

    def report_opacity_reset(self, iteration: int) -> None:
        self._artifacts.densify_json.write(json.dumps({
            "iter": iteration, "event": "opacity_reset"}) + "\n")
        print(f"iter {iteration}: opacity reset")

    def report_morton_sort(self, iteration: int) -> None:
        self._artifacts.densify_json.write(json.dumps({
            "iter": iteration, "event": "morton_sort"}) + "\n")

    def report_complete(self, iterations: int, wall_time: float) -> None:
        rate = iterations / max(wall_time, 1e-9)
        print(f"[train] loop done: {iterations} iters in {wall_time:.1f}s "
              f"({rate:.1f} it/s)")
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
            print(f"[psnr-curve] wrote psnr_curve.json but the PNG failed "
                  f"({error}); `.venv/Scripts/python.exe -m pip install "
                  "matplotlib` to enable it.")



def load_scene_context(args: argparse.Namespace) -> SceneContext:
    scene = load_colmap(
        args.scene,
        width=args.width,
        height=args.height,
        downsample=args.downsample,
        init_opacity=args.init_opacity,
    )
    camera_count = len(scene.images)
    initial_count = scene.initial_points["means"].shape[0]
    width, height = scene.images[0].camera.image_size
    print(f"loaded {camera_count} cameras, {initial_count} init points, "
          f"scene_extent={scene.scene_extent:.3f}")

    stride, test_indices, train_indices = resolve_holdout_split(
        args, camera_count)
    morton_min, morton_max = compute_morton_bounds(
        scene.initial_points["means"])
    print(f"schedule: densify [{args.densify_from}, {args.densify_to}] "
          f"every {args.densify_every}")
    return SceneContext(
        scene=scene,
        width=width,
        height=height,
        camera_count=camera_count,
        initial_gaussian_count=initial_count,
        extent=scene.scene_extent,
        train_indices=train_indices,
        test_indices=test_indices,
        holdout_stride=stride,
        morton_min=morton_min,
        morton_max=morton_max,
    )


def create_training_device(args: argparse.Namespace) -> Any:
    specialization = ShaderSpecialization.from_args(args)
    device = create_device(
        backend=args.backend,
        shader_debug_info=args.profile_run,
        defines=specialization.compiler_defines(),
    )
    if args.profile_run:
        print("[profile-run] shaders compiled with source-level debug info "
              "(SPIR-V NonSemantic.Shader.DebugInfo) for Nsight correlation.")
    return device


def create_gpu_queue_lock(*, viewer_enabled: bool) -> Any:
    """Return the reentrant queue lock shared by training and the viewer.

    A reentrant lock is required because coarse training/viewer operations can
    enter runtime code that acquires the same queue lock again.
    """
    return threading.RLock() if viewer_enabled else None


def create_pipeline(*, args: argparse.Namespace, device: Any, gpu_lock: Any,
                    scene: SceneContext, lambda_l2: float,
                    lambda_ssim: float) -> Pipeline:
    return Pipeline(
        device=device,
        width=scene.width,
        height=scene.height,
        max_splats=args.max_splats,
        gpu_lock=gpu_lock,
        raster_config=RasterConfig(
            z_near_cull=args.z_near_cull,
            cluster_cull=args.cluster_cull,
            mip_enabled=args.mip_splatting,
            kernel_size=(args.mip_kernel_size if args.mip_splatting else 0.3),
        ),
        compositor_config=CompositorConfig(
            extinction_hardness=args.extinction_hardness,
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
    )


def initialize_pipeline(*, pipeline: Pipeline, initial_points: dict,
                        densify_mode: DensificationMode,
                        use_flat_bucket_backward: bool) -> None:
    pipeline.set_gaussians_from_numpy(**initial_points)
    pipeline.clear_tracking()
    pipeline.use_flat_bucket_backward = use_flat_bucket_backward
    pipeline._clear_grads()
    if densify_mode == DensificationMode.STRUCTURE_AWARE:
        pipeline.clear_structure_scores()


def main() -> None:
    TrainingApplication.from_cli().run()


if __name__ == "__main__":
    main()
