"""Nsight Graphics profiling for the v2 trainer.

Keeps every GPU-profiling concern out of the training script so the trainer
reads as pure training. A ``Profiler`` owns: the profile-run session tweaks,
self-injection of the Nsight capture, the present window Nsight needs as a
frame boundary, and the deterministic GPU-Trace capture loop.

It is a no-op unless ``--profile-run`` (or a ``--capture-*`` flag) is set.
Profile frames are intentionally serialized: each Present is issued only after
the iteration's latest GPU submit completes, so Nsight cannot place a frame
boundary halfway through an asynchronous command buffer.
"""

from __future__ import annotations

import time

import numpy as np

from trainer.config import AdamLRs


class Profiler:
    """Encapsulates Nsight Graphics profiling for a single training run."""

    def __init__(self, args) -> None:
        self.args = args
        self.ngfx_cap = None       # NgfxCapture self-injection handle, or None
        self.window = None         # spy.Window presented one-frame-per-step
        self.surface = None        # its swapchain surface
        self.pipeline = None       # bound in open_window()

    # -- session setup -----------------------------------------------------

    def configure_session(self) -> None:
        """Force the session into a clean, self-contained presenting Vulkan app.

        A profile run keeps a normal training loop (densify + logs still run)
        but disables the viewer stream (a second device/window pollutes the
        capture) and previews (frame-boundary I/O breaks Nsight's
        step-per-frame cadence). Vulkan is the only backend Nsight can hook.
        Headless CLI-driven capture (``--capture-external``) implies a profile
        run and lets ngfx.exe own injection + start/stop/save.
        """
        args = self.args
        if args.capture_external:
            args.profile_run = True
        if args.profile_run:
            args.no_viewer = True
            args.backend = "vulkan"
            args.preview_every = 0

    def inject(self) -> None:
        """Self-injection path (``--capture-iter``): load NGFX_Injection BEFORE
        any Vulkan driver load.

        This process loads NGFX_Injection.dll and triggers the capture itself;
        the trace lives in the in-process HUD and must be persisted via the
        HUD's Save button (there is NO save function in the NGFX_Injection API).
        The external path (``--capture-external``) is skipped here — ngfx.exe
        injects and controls start/stop + auto-save. If the SDK isn't installed
        or injection fails we log and continue without captures.
        """
        args = self.args
        if not (args.capture_iter > 0 and not args.capture_external):
            return
        if not args.profile_run:
            print("[capture] --capture-iter requires --profile-run. ignoring.")
            return
        from trainer.ngfx_capture import NgfxCapture
        cap = NgfxCapture()
        if not cap.inject(activity=args.capture_activity,
                          project_path=args.capture_project):
            return
        self.ngfx_cap = cap
        proj_msg = (f" (project={args.capture_project})"
                    if args.capture_project else "")
        print(f"[capture] will trigger Nsight {args.capture_activity!r} "
              f"capture at iter {args.capture_iter}{proj_msg}")

    def open_window(self, pipeline, width: int, height: int) -> None:
        """Open the present window Nsight delimits work by (``--profile-run``).

        Nsight Graphics hooks the Vulkan device and delimits work by Present, so
        profiling needs a real windowed, presenting Vulkan app. Each regular
        frame is one completed loop iteration: train_step, scheduled mutations,
        and any scheduled observations. The deterministic capture loop emits a
        narrower frozen frame containing only one train_step.
        """
        self.pipeline = pipeline
        if not self.args.profile_run:
            return
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

        Returns True when a capture ran — the caller should then finalize its
        own resources (log files, etc.) and exit the process; False otherwise.
        """
        args = self.args
        if not ((self.ngfx_cap is not None or args.capture_external)
                and it == args.capture_iter):
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

        # Warm up a few frozen frames BEFORE arming so Nsight sees a steady
        # state (caches warm, no first-frame compile/alloc spikes) at sampling.
        print("[capture] warming up pinned/frozen frames before arm...")
        for _ in range(8):
            if self.window is not None:
                self.window.process_events()
            pipeline.train_step(pinned_camera, lrs=frozen_lrs)
            if self.window is not None:
                self.present_frame(pipeline.last_submit_id)
        pipeline.runtime.wait_for_idle()

        if self.ngfx_cap is not None:
            print(f"[capture] triggering Nsight capture at iter {it}")
            self.ngfx_cap.trigger()

        # Continuous capture loop of the identical, representative frame.
        #   self-injection: run until the profile window is closed (user saves
        #     from the HUD, then closes the window).
        #   external (ngfx.exe): present a BOUNDED --capture-frames (chosen >
        #     start-after + limit-to + margin) so the whole trace window falls
        #     inside identical frozen frames, then exit so ngfx.exe can finalize
        #     + auto-save the .ngfx-gputrace.
        if args.capture_external:
            print(f"[capture] external mode: presenting {args.capture_frames} "
                  f"frozen frames for ngfx.exe, then exiting.")
        else:
            print("[capture] presenting pinned/frozen frames continuously for "
                  "GPU Trace; close the profile window when Nsight is done.")
        frozen_i = 0
        while True:
            if args.capture_external and frozen_i >= args.capture_frames:
                break
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
        print("[capture] window closed, exiting." if not args.capture_external
              else f"[capture] presented {args.capture_frames} frozen frames; "
                   "exiting so ngfx.exe can finalize + auto-save the trace.")
        return True
