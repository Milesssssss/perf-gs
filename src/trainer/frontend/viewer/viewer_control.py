# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Lifecycle wrapper for the optional in-process live viewer.

Live viewing is IN-PROCESS: a background-thread :class:`~trainer.viewer.Viewer`
shares the trainer's GPU device (serialized by a shared ``gpu_lock``). Its
lifecycle used to be scattered across ``app.main()`` as five separate
``if viewer is not None`` blocks that also reached into the viewer's private
``_thread`` attribute.

:class:`ViewerSession` owns that lifecycle so the training frontend never has to
null-check the viewer or touch its thread state:

  * :meth:`create`             — build + push the initial snapshot (render
                                 thread deferred until setup submits are done)
  * :meth:`start`              — start the render thread
  * :meth:`wait_for_start`     — the ``--wait-for-viewer`` gate
  * :meth:`set_training_active`
  * :meth:`push_snapshot`      — per-iteration zero-copy GPU->GPU snapshot
  * :meth:`finish_training`    — publish the end of a run (non-blocking)
  * :meth:`hold_after_training`— keep the window open until closed

When live viewing is disabled (``--no-viewer``) the session wraps ``None`` and
every method is a no-op. :meth:`is_alive` replaces direct ``viewer._thread``
access.

The session ALWAYS owns a :class:`~trainer.frontend.viewer.session.Session`,
even under ``--no-viewer``: producers (the reporter, the hub orchestrator) publish
progress/log/stats into ``session.state`` unconditionally and never null-check a
viewer. With no viewer attached nothing reads it, which costs a lock and a dataclass
copy per publish.
"""
from __future__ import annotations

import contextlib
import threading
import time
from typing import TYPE_CHECKING

from trainer.frontend.viewer.session import MODE_HUB, MODE_TRAIN, OK, Session

if TYPE_CHECKING:
    import slangpy as spy

    from trainer.backend.render.pipeline import TrainingPipeline
    from trainer.data.dataset import Camera
    from trainer.frontend.settings import TrainingSetup
    from trainer.frontend.viewer.viewer import Viewer


class ViewerSession:
    """Owns the optional live viewer, its full lifecycle, and the session state."""

    def __init__(self, viewer: Viewer | None,
                 state: Session | None = None) -> None:
        self._viewer = viewer
        # One state object shared with the viewer's panel (see module docstring).
        if state is None:
            state = viewer.state if viewer is not None else Session(MODE_TRAIN)
        self.state = state

    @property
    def enabled(self) -> bool:
        """Whether a live viewer is attached (``False`` under ``--no-viewer``)."""
        return self._viewer is not None

    @classmethod
    def create(
        cls,
        *,
        device: spy.Device,
        width: int,
        height: int,
        max_splats: int,
        fovy: float,
        gpu_lock,
        pipeline: TrainingPipeline,
        test_cameras: list[tuple[str, Camera]] | None = None,
    ) -> ViewerSession:
        """Build the viewer and push the initialized gaussians, but DO NOT start
        the render thread yet.

        The remaining setup submits ~200 texture uploads on the trainer thread
        WITHOUT ``gpu_lock``; a running render thread would race those on the
        shared VkQueue. The caller must invoke :meth:`start` once setup is done.
        """
        from trainer.frontend.viewer.viewer import Viewer

        state = Session(MODE_TRAIN)
        viewer = Viewer(
            device=device, width=width, height=height,
            max_splats=max_splats, fovy=fovy, gpu_lock=gpu_lock,
            sh_degree=pipeline.appearance_config.sh_degree,
            mip_enabled=pipeline.raster_config.mip_enabled,
            mip_kernel_size=pipeline.raster_config.kernel_size,
            test_cameras=test_cameras,
            state=state,
        )
        viewer.update_gaussians(pipeline)  # show the initialized gaussians
        print("[viewer] created (render thread starts after setup completes).")
        return cls(viewer, state)

    @classmethod
    def create_hub(
        cls,
        *,
        device: spy.Device,
        width: int,
        height: int,
        max_splats: int,
        fovy: float,
        gpu_lock,
        sh_degree: int,
        mip_enabled: bool,
        mip_kernel_size: float,
        setup: TrainingSetup | None = None,
    ) -> ViewerSession:
        """Build a SCENE-LESS hub viewer (no pipeline/scene yet).

        The user picks a dataset via drag-drop and drives reconstruct/train from
        the panel; the orchestrator reads those requests off ``session.state``,
        publishes progress back into it, and later feeds gaussians via
        :meth:`push_snapshot` and cameras via :meth:`attach_scene`.

        Model hyperparameters start from the CLI args at launch and stay EDITABLE
        until Train: ``setup`` is the shared store the panel's TRAINING SETUP
        section writes into (see :mod:`trainer.frontend.settings`), which the
        orchestrator applies to its args when it builds the run.
        """
        from trainer.frontend.viewer.viewer import Viewer

        state = Session(MODE_HUB, setup=setup)
        viewer = Viewer(
            device=device, width=width, height=height,
            max_splats=max_splats, fovy=fovy, gpu_lock=gpu_lock,
            sh_degree=sh_degree, mip_enabled=mip_enabled,
            mip_kernel_size=mip_kernel_size, test_cameras=None,
            state=state,
        )
        print("[viewer] hub created (no dataset yet).")
        return cls(viewer, state)

    def attach_scene(self, test_cameras, snap: bool = False) -> None:
        """Populate the camera dropdown once a dataset/scene is available.
        ``snap`` also parks on the first camera (see Viewer.attach_scene)."""
        if self._viewer is not None:
            self._viewer.attach_scene(test_cameras, snap=snap)

    def preview_points(self, initial_points: dict) -> int:
        """Display a COLMAP point cloud before training starts. Returns the number
        of points shown (0 with no viewer)."""
        if self._viewer is None:
            return 0
        return self._viewer.preview_points(initial_points)

    def reconfigure_model(self, *, max_splats: int, sh_degree: int,
                          mip_enabled: bool, mip_kernel_size: float) -> bool:
        """Re-size/re-shape the viewer's own pipeline after the hub's TRAINING
        SETUP changed the model (no-op when unchanged or with no viewer)."""
        if self._viewer is None:
            return False
        return self._viewer.reconfigure_model(
            max_splats=max_splats, sh_degree=sh_degree,
            mip_enabled=mip_enabled, mip_kernel_size=mip_kernel_size)

    def suspend_rendering(self) -> None:
        """Park the render thread (no GPU work, no frames) until :meth:`resume_rendering`.

        Needed for two different reasons, and the second one is easy to miss:
          * GPU — uploads/pipeline swaps must not race the render thread on the
            shared queue;
          * CPU — the render thread holds the GIL for most of every frame, so any
            heavy pure-Python work on another thread (e.g. the COLMAP binary
            parsers) runs 50-100x slower next to a live viewer.
        """
        if self._viewer is not None:
            self._viewer.suspend_rendering()

    def resume_rendering(self) -> None:
        if self._viewer is not None:
            self._viewer.resume_rendering()

    def start(self) -> None:
        """Start the background render thread (call once setup submits are done)."""
        if self._viewer is None:
            return
        self._viewer.start()
        print("[viewer] render thread started.", flush=True)

    def is_alive(self) -> bool:
        """Whether the render thread has been started and is still running."""
        if self._viewer is None:
            return False
        thread = self._viewer._thread
        return thread is not None and thread.is_alive()

    def wait_for_start(self) -> None:
        """Hold before training so the viewer can inspect the initialized
        gaussians, then block until the viewer's 'Start Training' button, an
        ``Enter`` in this terminal (headless escape hatch), or the viewer window
        closing.
        """
        if self._viewer is None:
            return
        print("[wait-for-viewer] showing initialized gaussians. Waiting for the "
              "viewer's 'Start Training' button (or press Enter here)...")
        enter = threading.Event()

        def _wait_enter() -> None:
            try:
                input()
            except EOFError:
                pass
            enter.set()

        threading.Thread(target=_wait_enter, daemon=True).start()
        while not self._viewer.start_requested() and not enter.is_set():
            if not self.is_alive():
                print("[wait-for-viewer] viewer window closed; starting training.")
                break
            time.sleep(0.05)
        print("[wait-for-viewer] starting training.")

    def set_training_active(self, active: bool = True) -> None:
        """Flag that the trainer is inside its loop.

        The viewer needs it for the Pause button; the session needs the RUNNING
        transition either way, so with no viewer attached it is published here and
        the state stays truthful under ``--no-viewer``."""
        if self._viewer is not None:
            self._viewer.set_training_active(active)
        elif active:
            self.state.training_running()

    @contextlib.contextmanager
    def unthrottled(self):
        """Render at full rate for the duration: the trainer is not stepping.

        The viewer rations scene renders while training steps compete for the queue
        (see ``Viewer._scene_due``). An eval sweep or a render-all dump is NOT that
        situation — the loop is blocked on it — so the ration is pure loss there,
        and those phases are exactly when someone is watching the window because
        nothing else is happening. No-op with ``--no-viewer``.
        """
        if self._viewer is None:
            yield
            return
        self._viewer.suspend_frame_budget()
        try:
            yield
        finally:
            self._viewer.resume_frame_budget()

    def is_paused(self) -> bool:
        """Whether the viewer's Pause button currently has training suspended."""
        return self._viewer is not None and self._viewer.is_paused()

    def stop_requested(self) -> bool:
        """Whether the user asked to end the run early (the Stop button)."""
        return self._viewer is not None and self._viewer.stop_requested()

    def clear_stop_request(self) -> None:
        """Before a new run: a stop belongs to the run that was stopped."""
        if self._viewer is not None:
            self._viewer.clear_stop_request()

    def wait_while_paused(self) -> None:
        """Block the trainer while the viewer's Pause button is engaged (no-op
        when live viewing is disabled)."""
        if self._viewer is not None:
            self._viewer.wait_while_paused()

    def push_snapshot(self, pipeline: TrainingPipeline) -> None:
        """Push a gaussian snapshot to the viewer (GPU->GPU copy of the ACTIVE
        splat range, plus a 4-byte count readback — see ``update_gaussians``)."""
        if self._viewer is not None:
            self._viewer.update_gaussians(pipeline)

    def clear_view(self) -> None:
        """Blank the displayed gaussians and the camera list (the hub's Clear)."""
        if self._viewer is not None:
            self._viewer.clear_view()

    def finish_training(self, pipeline: TrainingPipeline) -> None:
        """Publish the end of the training LOOP and push its final gaussians —
        WITHOUT blocking.

        Split out of :meth:`hold_after_training` for the hub, where the end of a run
        is not the end of the session: the orchestrator goes back to serving the
        panel (export, re-train, another dataset), so it must not be parked in a
        thread join. Retires the "running" state of the TRAINING section, which
        otherwise reads as if the trainer were still going.

        The PHASE is deliberately NOT set here: the loop finishing is not the run
        finishing — the metrics pass and the .ply export still have to run, and the
        orchestrator publishes ``PHASE_FINALIZE`` for that and ``PHASE_DONE`` when
        it is genuinely over (see ``TrainingApplication._run_training_and_post``).
        """
        if self._viewer is not None:
            self._viewer.set_training_active(False)
            try:
                self._viewer.update_gaussians(pipeline)
            except Exception:
                pass

    def hold_after_training(self, pipeline: TrainingPipeline) -> None:
        """Push the final state and leave the window live until the user closes
        it (or Ctrl-C, which stops the viewer)."""
        self.finish_training(pipeline)
        self.state.notify("Training complete", OK)
        self.state.log("training complete — close the viewer window to exit")
        if self._viewer is None:
            return
        print("[viewer] training done — viewer still live; close its window to exit.")
        try:
            thread = self._viewer._thread
            if thread is not None:
                # We just wait until the user closes/kills the window 
                thread.join()
        except KeyboardInterrupt:
            self._viewer.stop()
