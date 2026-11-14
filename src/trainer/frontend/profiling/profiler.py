# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Nsight Graphics profiling, constructed only on a profile run: the present window Nsight uses
as a frame boundary, and the deterministic GPU-Trace capture loop."""

from __future__ import annotations

import dataclasses
import time


from trainer.config.config import (AdamLRs, DensificationProperties)
from trainer.frontend.schedule import IterationContext


def apply_profile_session_overrides(args) -> bool:
    enabled = bool(args.profile_run or args.capture_external)
    if enabled:
        args.profile_run = True
        args.no_viewer = True
        args.backend = "vulkan"
        args.preview_every = 0
    return enabled


class Profiler:
    """Encapsulates Nsight Graphics profiling for a single training run."""

    def __init__(self, args) -> None:
        self.args = args
        self.window = None         # spy.Window presented one-frame-per-step
        self.surface = None        # its swapchain surface
        self.pipeline = None       # bound in open_window()

    def open_window(self, pipeline, width: int, height: int) -> None:
        """Open the present window Nsight delimits work by.
        Each regular frame is one completed loop iteration; capture frames hold one train_step."""
        self.pipeline = pipeline
        import slangpy as spy
        self.window = spy.Window(
            width=width,
            height=height,
            title="3DGS profile (Nsight Graphics)",
        )
        self.surface = pipeline.device.create_surface(self.window)
        self.surface.configure(width=width, height=height)
        print(
            "[profile-run] presenting one fence-completed iteration per "
            "Nsight frame. Frame boundaries cannot run ahead of GPU passes. "
            "Densify + observations remain inside their owning iteration; "
            "close the window (or reach --iters) to quit."
        )

    def should_stop(self) -> bool:
        """True once the user closes the profile window, so Nsight can capture and quit cleanly."""
        return self.window is not None and self.window.should_close()

    def present_completed_frame(self, submit_id: int) -> None:
        """Present one completed GPU iteration as an Nsight frame, fenced on both sides so
        frame boundaries cannot run ahead of the GPU. acquire_next_image can raise on
        minimize/resize; a dropped frame reconfigures instead of killing training."""
        if self.surface is None:
            return
        if submit_id:
            self.pipeline.runtime.wait_for_submit(submit_id)
        surf_tex = None
        try:
            surf_tex = self.surface.acquire_next_image()
        except Exception:
            surf_tex = None
        if surf_tex is None:
            try:
                w, h = max(self.window.width, 1), max(self.window.height, 1)
                self.pipeline.runtime.wait_for_idle()
                self.surface.unconfigure()
                self.surface.configure(width=w, height=h)
            except Exception:
                pass
            return
        batch = self.pipeline.runtime.command_batch("profile_present")
        batch.encoder.blit(surf_tex, self.pipeline.buffers.raster.output_image)
        present_submit = self.pipeline.runtime.submit(
            batch, after_submit=self.surface.present)
        self.pipeline.runtime.wait_for_submit(present_submit)

    def present_frame(self, submit_id: int) -> None:
        """Pump events and delimit one completed iteration. No-op normally."""
        if self.window is None:
            return
        self.window.process_events()
        self.present_completed_frame(submit_id)

    def close(self) -> None:
        """Close the profile window if one is open (idempotent)."""
        if self.window is None:
            return
        try:
            self.window.close()
        except Exception:
            pass

    def run_capture_if_due(self, pipeline, context: IterationContext,
                           queue_limiter) -> bool:
        """At ``capture_iter``, present ``--capture-frames`` frozen, bit-identical train_steps
        (pinned camera, zero LRs, no densify) for the external GPU-Trace capture, which samples
        a continuous frame stream. Returns True when it ran; the caller then ends the run."""
        args = self.args
        it = context.step_number
        cam_idx = context.camera_index
        if not (args.capture_external and it == args.capture_iter):
            return False

        print(f"[capture] draining queue at iter {it}...")
        pipeline.runtime.wait_for_idle()
        queue_limiter.clear()
        try:
            n_g_capture = pipeline.primitives.count()
        except Exception:
            n_g_capture = -1
        print(f"[capture] state @ iter {it}: cam_idx={cam_idx}  "
              f"n_gaussians={n_g_capture}")

        # Zero LRs run Adam's full workload without moving params; no densification accumulation.
        frozen_ctx = dataclasses.replace(
            context,
            learning_rates=AdamLRs(mean=0.0, scale=0.0, rot=0.0, opacity=0.0,
                                   color=0.0, sh_rest=0.0),
            iteration_contextual_variables=dataclasses.replace(
                context.iteration_contextual_variables,
                densification_variables=DensificationProperties(),
            ),
        )

        # Warm up so the trace window sees a steady state (no first-frame compile/alloc spikes).
        print("[capture] warming up pinned/frozen frames before arm...")
        for _ in range(8):
            if self.window is not None:
                self.window.process_events()
            pipeline.train_step(frozen_ctx.camera_entry.camera, lrs=frozen_ctx.learning_rates,
                                contextual=frozen_ctx.iteration_contextual_variables)
            if self.window is not None:
                self.present_frame(pipeline.last_submit_id)
        pipeline.runtime.wait_for_idle()

        # Bounded frame count: the trace window must fall inside frozen frames before exiting.
        print(f"[capture] presenting {args.capture_frames} frozen frames for "
              "ngfx.exe, then exiting.")
        frozen_i = 0
        while frozen_i < args.capture_frames:
            if self.window is not None:
                if self.window.should_close():
                    break
                self.window.process_events()
            pipeline.train_step(frozen_ctx.camera_entry.camera, lrs=frozen_ctx.learning_rates,
                                contextual=frozen_ctx.iteration_contextual_variables)
            if self.window is not None:
                self.present_frame(pipeline.last_submit_id)
            else:
                time.sleep(0.01)
            frozen_i += 1

        # Drain before teardown, else ngfx.exe may see an in-flight device at exit.
        pipeline.runtime.wait_for_idle()
        self.close()
        print(f"[capture] presented {args.capture_frames} frozen frames; "
              "exiting so ngfx.exe can finalize + auto-save the trace.")
        return True
