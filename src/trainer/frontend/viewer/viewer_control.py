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
  * :meth:`hold_after_training`— keep the window open until closed

When live viewing is disabled (``--no-viewer``) the session wraps ``None`` and
every method is a no-op. :meth:`is_alive` replaces direct ``viewer._thread``
access.
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import slangpy as spy

    from trainer.backend.render.pipeline import Pipeline
    from trainer.frontend.viewer.viewer import Viewer


class ViewerSession:
    """Owns the optional live viewer and its full lifecycle."""

    def __init__(self, viewer: "Viewer | None") -> None:
        self._viewer = viewer

    @property
    def enabled(self) -> bool:
        """Whether a live viewer is attached (``False`` under ``--no-viewer``)."""
        return self._viewer is not None

    @classmethod
    def create(
        cls,
        *,
        device: "spy.Device",
        width: int,
        height: int,
        max_splats: int,
        fovy: float,
        gpu_lock,
        pipeline: "Pipeline",
    ) -> "ViewerSession":
        """Build the viewer and push the initialized gaussians, but DO NOT start
        the render thread yet.

        The remaining setup submits ~200 texture uploads on the trainer thread
        WITHOUT ``gpu_lock``; a running render thread would race those on the
        shared VkQueue. The caller must invoke :meth:`start` once setup is done.
        """
        from trainer.frontend.viewer.viewer import Viewer

        viewer = Viewer(
            device=device, width=width, height=height,
            max_splats=max_splats, fovy=fovy, gpu_lock=gpu_lock,
            sh_degree=pipeline.appearance_config.sh_degree,
        )
        viewer.update_gaussians(pipeline)  # show the initialized gaussians
        print("[viewer] created (render thread starts after setup completes).")
        return cls(viewer)

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
        """Flag the viewer that training is now running."""
        if self._viewer is not None:
            self._viewer.set_training_active(active)

    def push_snapshot(self, pipeline: "Pipeline") -> None:
        """Push a gaussian snapshot to the viewer (zero-copy GPU->GPU blit)."""
        if self._viewer is not None:
            self._viewer.update_gaussians(pipeline)

    def hold_after_training(self, pipeline: "Pipeline") -> None:
        """Push the final state and leave the window live until the user closes
        it (or Ctrl-C, which stops the viewer)."""
        if self._viewer is None:
            return
        try:
            self._viewer.update_gaussians(pipeline)
        except Exception:
            pass
        print("[viewer] training done — viewer still live; close its window to exit.")
        try:
            thread = self._viewer._thread
            if thread is not None:
                thread.join()
        except KeyboardInterrupt:
            self._viewer.stop()
