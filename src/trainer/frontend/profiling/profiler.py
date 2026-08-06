"""Nsight Graphics profiling for the v2 trainer.

Isolated from the training path: a ``Profiler`` is constructed ONLY on a profile
run (see :func:`apply_profile_session_overrides`), so a normal run never
instantiates it or touches its hooks. It owns the present window Nsight uses as a
frame boundary and the deterministic GPU-Trace capture loop, driven externally by
``scripts/running/capture_gpu_trace.py`` (which invokes ngfx.exe).

Profile frames are intentionally serialized: each Present is issued only after
the iteration's latest GPU submit completes, so Nsight cannot place a frame
boundary halfway through an asynchronous command buffer.
"""

from __future__ import annotations

import time

import numpy as np

from trainer.config.config import AdamLRs


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

        Nsight Graphics hooks the Vulkan device and delimits work by Present, so
        profiling needs a real windowed, presenting Vulkan app. Each regular
        frame is one completed loop iteration: train_step, scheduled mutations,
        and any scheduled observations. The deterministic capture loop emits a
        narrower frozen frame containing only one train_step.
        """
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

    # -- per-iteration hooks ----------------------------------------------

    def should_stop(self) -> bool:
        """True once the user closes the profile window (stop training so Nsight
        can capture and quit cleanly)."""
        return self.window is not None and self.window.should_close()

    def present_completed_frame(self, submit_id: int) -> None:
        """Present exactly one completed GPU iteration as an Nsight frame.

        Nsight counts ``Present`` calls on the CPU while GPU submissions are
        asynchronous. Without the pre-present fence, the CPU can announce
        several frame boundaries while the GPU is still executing an earlier
        pass, making a captured context begin/end halfway through iterations.
        We therefore complete the iteration submit first, then submit the blit
        and Present atomically through ``GpuRuntime``, and fence that blit before
        allowing the next iteration to start.

        acquire_next_image can RAISE (not just return None) when the window is
        minimized/resized/loses the swapchain, so we catch it and reconfigure —
        a dropped frame must never kill the training loop we're profiling.
        """
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
        batch.encoder.blit(surf_tex, self.pipeline.buffers.output_image)
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

    # -- deterministic GPU-Trace capture ----------------------------------

    def run_capture_if_due(self, pipeline, it: int, entry, cam_idx: int,
                           queue_limiter) -> bool:
        """Run the deterministic GPU-Trace capture when ``it == capture_iter``.

        GPU Trace is NOT a single-frame debugger: it is a continuous
        hardware-counter profiler that, after arm, samples counters across a
        collection window (~1-2s) of CONTINUOUSLY-PRESENTED frames, replaying
        the recurring per-frame workload. It needs a steady stream of
        representative frames — freezing on a static blit gives it nothing to
        sample. So at ``capture_iter`` we enter a dedicated capture loop that
        keeps presenting representative frames, made deterministic + stable by:
          - PIN the camera to capture_iter's camera (every frame renders the
            same view, so whichever frame Nsight samples matches across runs);
          - lr = 0 so the Adam step still runs (full backward workload IS
            profiled) but gaussians never move -> every frame is bit-identical;
          - no densify / opacity-reset / readback in the loop, so the workload
            shape is a pure, stable train_step.

        External capture (``scripts/running/capture_gpu_trace.py`` -> ngfx.exe)
        owns injection + start/stop + auto-save; the trainer just presents a
        bounded ``--capture-frames`` window of frozen frames and exits.

        Returns True when a capture ran — the caller should then finalize its
        own resources (log files, etc.) and exit the process; False otherwise.
        """
        args = self.args
        if not (args.capture_external and it == args.capture_iter):
            return False

        print(f"[capture] draining queue at iter {it}...")
        pipeline.runtime.wait_for_idle()
        queue_limiter.clear()
        try:
            n_g_capture = int(
                pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
        except Exception:
            n_g_capture = -1
        print(f"[capture] state @ iter {it}: cam_idx={cam_idx}  "
              f"n_gaussians={n_g_capture}")

        pinned_camera = entry.camera
        # Zero ALL learning rates so the Adam step runs its full workload but no
        # parameter moves -> every frame in the collection window is identical.
        frozen_lrs = AdamLRs(mean=0.0, scale=0.0, rot=0.0, opacity=0.0, color=0.0)

        # Warm up a few frozen frames BEFORE the trace window so Nsight sees a
        # steady state (caches warm, no first-frame compile/alloc spikes).
        print("[capture] warming up pinned/frozen frames before arm...")
        for _ in range(8):
            if self.window is not None:
                self.window.process_events()
            pipeline.train_step(pinned_camera, lrs=frozen_lrs)
            if self.window is not None:
                self.present_frame(pipeline.last_submit_id)
        pipeline.runtime.wait_for_idle()

        # Present a BOUNDED --capture-frames (chosen > start-after + limit-to +
        # margin) so the whole trace window falls inside identical frozen frames,
        # then exit so ngfx.exe can finalize + auto-save the .ngfx-gputrace.
        print(f"[capture] presenting {args.capture_frames} frozen frames for "
              "ngfx.exe, then exiting.")
        frozen_i = 0
        while frozen_i < args.capture_frames:
            if self.window is not None:
                if self.window.should_close():
                    break
                self.window.process_events()
            pipeline.train_step(pinned_camera, lrs=frozen_lrs)
            if self.window is not None:
                self.present_frame(pipeline.last_submit_id)
            else:
                time.sleep(0.01)
            frozen_i += 1

        # Drain so the last presented frames complete before teardown —
        # otherwise ngfx.exe may see an in-flight device at exit.
        pipeline.runtime.wait_for_idle()
        self.close()
        print(f"[capture] presented {args.capture_frames} frozen frames; "
              "exiting so ngfx.exe can finalize + auto-save the trace.")
        return True
