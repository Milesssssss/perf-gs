# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""FlyCamera mouse-look must be screen-relative at ANY roll.

The bug: rotate() fed the mouse delta into world-referenced yaw/pitch with a
roll compensation that rotated by the wrong sign, so look-around was correct
at roll=0 but skewed/tumbled as roll grew (worse with the r/t roll keys).

The invariant (convention-free): a given mouse drag (dx, dy), expressed in the
camera's CURRENT screen basis (rolled right/up), must produce the same forward
motion regardless of roll. We measure it at roll=0 (the known-good reference)
and require every rolled camera to reproduce the same screen-space response.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from trainer.data.dataset import Camera
from trainer.frontend.viewer.viewer import FlyCamera, _CameraRig


def _screen_response(roll_deg: float, dx: float, dy: float) -> tuple[float, float]:
    """Apply rotate(dx, dy) to a camera at the given roll and return the change
    in the forward vector projected onto that camera's (right, up) screen
    basis. Small deltas keep the forward(yaw,pitch) map ~linear."""
    cam = FlyCamera(yaw=20.0, pitch=-10.0, roll=roll_deg)
    r = cam.right.copy()
    u = cam.up.copy()
    f0 = cam.forward.copy()
    cam.rotate(dx, dy)
    df = cam.forward - f0
    return float(np.dot(df, r)), float(np.dot(df, u))


@pytest.mark.parametrize("roll", [0.0, 30.0, 45.0, 90.0, -60.0, 150.0])
@pytest.mark.parametrize("drag", [(1.0, 0.0), (0.0, 1.0), (0.7, -0.5)])
def test_look_is_screen_relative_under_roll(roll: float, drag: tuple) -> None:
    dx, dy = drag
    # Reference: the response in the roll=0 screen frame (the behavior the user
    # confirms is correct for level cameras).
    ref = _screen_response(0.0, dx, dy)
    got = _screen_response(roll, dx, dy)
    assert got[0] == pytest.approx(ref[0], abs=2e-3), (
        f"roll={roll} drag={drag}: screen-right response {got[0]:.4f} "
        f"!= reference {ref[0]:.4f}")
    assert got[1] == pytest.approx(ref[1], abs=2e-3), (
        f"roll={roll} drag={drag}: screen-up response {got[1]:.4f} "
        f"!= reference {ref[1]:.4f}")


def test_basis_orthonormal_and_rolls() -> None:
    """Sanity: right/up/forward stay orthonormal as roll varies."""
    for roll in [0.0, 37.0, 90.0, -120.0]:
        cam = FlyCamera(yaw=15.0, pitch=25.0, roll=roll)
        r, u, f = cam.right, cam.up, cam.forward
        for v in (r, u, f):
            assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)
        assert np.dot(r, u) == pytest.approx(0.0, abs=1e-6)
        assert np.dot(r, f) == pytest.approx(0.0, abs=1e-6)
        assert np.dot(u, f) == pytest.approx(0.0, abs=1e-6)


def _fake_camera(index: int):
    """A Camera whose view matrix is a pure translation along +x."""
    view = np.eye(4, dtype=np.float32)
    view[0, 3] = float(index)
    proj = np.eye(4, dtype=np.float32)
    return Camera(view_matrix=view, proj_matrix=proj, image_size=(100, 80))


def test_attaching_a_scene_stays_in_free_fly_by_default() -> None:
    rig = _CameraRig(fovy=45.0, render_size=(960, 540))

    rig.set_test_cameras([("a", _fake_camera(1)), ("b", _fake_camera(2))])

    assert rig.index == -1
    assert rig.selection() == 0
    assert rig.exact is None


def test_the_point_cloud_preview_parks_on_the_first_camera() -> None:
    """Without this the preview is invisible: the fly camera sits at the origin
    and a scene reconstructed anywhere else renders off-screen, which reads as
    "the dataset failed to load"."""
    rig = _CameraRig(fovy=45.0, render_size=(960, 540))

    rig.set_test_cameras([("a", _fake_camera(7)), ("b", _fake_camera(9))], snap=True)

    assert rig.index == 0
    assert rig.selection() == 1
    assert "a" in rig.label()


def test_snapping_an_empty_scene_falls_back_to_free_fly() -> None:
    rig = _CameraRig(fovy=45.0, render_size=(960, 540))

    rig.set_test_cameras([], snap=True)

    assert rig.index == -1
