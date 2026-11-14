# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Training logs, densify/opacity events, and the numbers the viewer shows."""

from __future__ import annotations

import argparse
import json
import time
from typing import TYPE_CHECKING

import numpy as np

from trainer.common.metrics import psnr
from trainer.frontend.reporting.outputs import RunArtifacts
from trainer.frontend.viewer.session import ERROR, Session

if TYPE_CHECKING:
    from trainer.backend.render.pipeline import TrainingPipeline
    from trainer.frontend.densification import DensifyOutcome
    from trainer.frontend.scene import SceneContext
    from trainer.frontend.schedule import IterationContext


class TrainingReporter:
    """Owns training log formatting, artifact schemas, and the live UI numbers."""

    def __init__(self, artifacts: RunArtifacts, args: argparse.Namespace,
                 scene: SceneContext, state: Session | None = None) -> None:
        self._artifacts = artifacts
        self._args = args
        self._scene = scene
        # The viewer's progress source; ViewerSession owns one even under --no-viewer.
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
        # Publish every field: a hub run may follow one whose numbers are still shown.
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
        self._state.publish_training(step=context.step_number, it_per_s=rate,
                                     elapsed_s=elapsed)
        self._last_time = now
        self._last_iteration = context.step_number

    def report_iteration(self, *, context: IterationContext, pipeline: TrainingPipeline,
                         target: np.ndarray,
                         current_count: int) -> tuple[bool, int]:
        pipeline.render(context.camera_entry.camera)
        count = pipeline.primitives.count()
        rendered = pipeline.read_output_image()[..., :3]

        if not np.all(np.isfinite(rendered)):
            nonfinite = int(np.sum(~np.isfinite(rendered)))
            message = (
                f"iter {context.step_number}: rendered image has {nonfinite} "
                f"non-finite pixels (out of {rendered.size}). Training aborted.")
            print(message)
            self._artifacts.log_txt.write(f"# ABORT: {message}\n")
            self._state.training_failed(message)
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
