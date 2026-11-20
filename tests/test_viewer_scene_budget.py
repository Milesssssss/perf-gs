# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The viewer's scene-render budget while training.

A viewer frame does the same work as a training step — bin_count, the 8-pass
radix sort, the blend over the whole model — and unthrottled it runs ~60 times a
second. Measured on bicycle with the splat count pinned, that took throughput
from ~155 it/s headless to ~35 it/s with the viewer attached, and the penalty is
worst early in a run when a training step is cheapest.

While training competes for the queue, the viewer paces frame presentation
(including UI-only frames) and can reuse the previous scene texture:

  * parked on a test camera -> only when new gaussians arrive (the view cannot
    move: navigating calls ``_CameraRig.leave``),
  * free-fly              -> capped at TRAINING_SCENE_FPS,
  * not training          -> uncapped.

These tests drive the policy directly rather than through a window, so they run
without a GPU: the method reads only the training flag, the rig's parked index,
the dirty Event and a clock it is handed.
"""

from __future__ import annotations

import threading

from trainer.frontend.viewer.viewer import TRAINING_SCENE_FPS, Viewer


class _Rig:
    def __init__(self, index: int = -1) -> None:
        self.index = index          # -1 == free-fly, >= 0 == parked


def _viewer(*, training: bool, parked: bool, dirty: bool = False) -> Viewer:
    """A Viewer shell carrying only what ``_scene_due`` reads. Constructing a real
    one needs a device, a window and a font atlas; the policy needs none of it."""
    viewer = object.__new__(Viewer)
    viewer._training_active = training
    viewer._rig = _Rig(0 if parked else -1)
    viewer._scene_dirty = threading.Event()
    if dirty:
        viewer._scene_dirty.set()
    viewer._last_scene_s = 0.0
    viewer._frame_budget_off = threading.Event()
    viewer._pause_event = threading.Event()
    return viewer


FRAME = 1.0 / TRAINING_SCENE_FPS


def test_frame_pacing_only_while_training_shares_the_gpu() -> None:
    viewer = _viewer(training=False, parked=True)
    assert viewer._training_frame_interval() == 0
    viewer._training_active = True
    assert viewer._training_frame_interval() == FRAME
    viewer._pause_event.set()
    assert viewer._training_frame_interval() == 0
    viewer._pause_event.clear()
    viewer.suspend_frame_budget()
    assert viewer._training_frame_interval() == 0
    viewer.resume_frame_budget()
    assert viewer._training_frame_interval() == FRAME


def test_not_training_is_never_throttled() -> None:
    """Once the run is over the viewer owns the GPU, so a finished result must be
    as smooth to inspect as any other 3D viewer."""
    viewer = _viewer(training=False, parked=False)
    assert viewer._scene_due(0.0)
    assert viewer._scene_due(0.001)       # back to back, still allowed
    parked = _viewer(training=False, parked=True)
    assert parked._scene_due(0.001)       # parked does not gate it either


def test_free_fly_while_training_is_capped_at_the_frame_budget() -> None:
    viewer = _viewer(training=True, parked=False)
    assert viewer._scene_due(FRAME)       # first frame of the run
    viewer._last_scene_s = FRAME

    assert not viewer._scene_due(FRAME + FRAME * 0.5)   # too soon
    assert viewer._scene_due(FRAME + FRAME)             # budget elapsed
    assert viewer._scene_due(FRAME + FRAME * 10)        # long overdue


def test_parked_on_a_test_camera_renders_only_on_new_data() -> None:
    """The view cannot move while parked, so there is nothing to animate: a
    held-out view should cost ONE render per snapshot, not 60 per second."""
    viewer = _viewer(training=True, parked=True)
    viewer._last_scene_s = 0.0

    # No new data: never due, no matter how much time passes.
    assert not viewer._scene_due(1.0)
    assert not viewer._scene_due(60.0)

    viewer.invalidate_scene()             # a snapshot landed
    assert viewer._scene_due(60.1)


def test_a_pending_change_overrides_the_budget() -> None:
    """New gaussians (or a camera snap, or a resize) must not wait out the frame
    budget — otherwise the picture lags the thing that changed it."""
    viewer = _viewer(training=True, parked=False, dirty=True)
    viewer._last_scene_s = 100.0
    assert viewer._scene_due(100.0)       # zero elapsed, but dirty


def test_a_paused_run_is_never_throttled() -> None:
    """Paused, the trainer submits nothing, so flying around the scene must be as
    smooth as after the run instead of crawling at the training budget."""
    viewer = _viewer(training=True, parked=False)
    viewer._last_scene_s = 0.0
    assert not viewer._scene_due(0.001)

    viewer._pause_event.set()
    assert viewer._scene_due(0.001)
    assert viewer._scene_due(0.002)

    viewer._pause_event.clear()
    viewer._last_scene_s = 0.002
    assert not viewer._scene_due(0.003)   # resumed: back on the ration


def test_leaving_a_parked_camera_restores_the_free_fly_cap() -> None:
    """Navigating away from a test camera drops to free-fly, which must start
    animating again rather than staying frozen on the parked policy."""
    viewer = _viewer(training=True, parked=True)
    viewer._last_scene_s = 0.0
    assert not viewer._scene_due(1.0)

    viewer._rig.index = -1                # _CameraRig.leave()
    assert viewer._scene_due(1.0)


def test_the_camera_rig_announces_a_snap() -> None:
    """``_scene_due`` can only see a snap if the rig reports it; while parked that
    callback is the ONLY thing that gets the new view on screen."""
    from trainer.frontend.viewer.viewer import _CameraRig
    from trainer.data.dataset import simple_camera
    from trainer.frontend.scene import CameraSet

    changes = []
    camera = simple_camera(width=8, height=8, eye=(0.0, 0.0, 0.0),
                           target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                           fovy_degrees=60.0)
    rig = _CameraRig(fovy=60.0, render_size=(8, 8),
                     camera_sets=[CameraSet("eval", (("a", camera), ("b", camera)))],
                     on_change=lambda: changes.append(1))

    rig.snap_to(1)
    assert len(changes) == 1, "parking on a camera did not announce the change"
    rig.leave()
    assert len(changes) == 2, "dropping to free-fly did not announce the change"


def test_rig_without_a_callback_still_works() -> None:
    """The default keeps ``_CameraRig`` usable on its own (the panel tests build
    one without a viewer)."""
    from trainer.frontend.viewer.viewer import _CameraRig

    rig = _CameraRig(fovy=60.0, render_size=(8, 8))
    rig.snap_to(-1)          # must not raise
    assert rig.index == -1


def test_suspending_the_budget_uncaps_a_still_active_run() -> None:
    """The eval sweep and the render-all dump BLOCK the training loop, so nothing
    is competing for the queue while they run - and they are exactly when someone
    is watching the window, because the iteration counter has stopped moving. The
    ration has to lift for them even though the run is nominally active."""
    viewer = _viewer(training=True, parked=False)
    viewer._last_scene_s = 0.0
    assert not viewer._scene_due(0.001)          # capped: mid-run, too soon

    Viewer.suspend_frame_budget(viewer)
    assert viewer._scene_due(0.001)              # uncapped for the duration
    viewer._scene_dirty.clear()
    assert viewer._scene_due(0.002)              # ...and it STAYS uncapped

    Viewer.resume_frame_budget(viewer)
    viewer._scene_dirty.clear()
    viewer._last_scene_s = 0.002
    assert not viewer._scene_due(0.003)          # back on the ration


def test_suspending_the_budget_also_forces_a_repaint() -> None:
    """Whatever suspends the budget has usually just changed the model (a densify,
    an eval render), so the frame on screen is stale; lifting the ration without
    marking it dirty would leave a parked view showing the old scene."""
    viewer = _viewer(training=True, parked=True)
    viewer._scene_dirty.clear()
    assert not viewer._scene_due(10.0)
    Viewer.suspend_frame_budget(viewer)
    assert viewer._scene_dirty.is_set()
