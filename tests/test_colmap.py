"""COLMAP camera-alignment tests.

Verify that our Python-side parsing of ``cameras.txt``/``images.txt``/
``points3D.txt`` into :class:`Camera` view+projection matrices is
consistent with the ground-truth PNGs: projecting the initial point
cloud with each camera should yield a pixel-space bounding box that
matches the non-black bounding box in the corresponding GT image.

Catches regressions in:
- view-matrix quaternion/translation encoding
- projection-matrix y-axis convention (COLMAP is y-down, matches our
  shader's ``uv * H`` pixel mapping — no flip)
- principal-point offset handling for off-center principal points
- image-name → camera-pose index mapping
- scaling of fx/fy/cx/cy when rendering at a resolution other than the
  COLMAP native size

Skipped if the Suzanne scene isn't present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


SCENE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "synthetic" / "suzanne_blender"


def _load_scene_or_skip(width: int = 960, height: int = 540):
    if not SCENE_DIR.exists():
        pytest.skip(f"suzanne_blender scene not present at {SCENE_DIR}")
    from trainer.data.colmap import load_colmap
    return load_colmap(SCENE_DIR, width=width, height=height)


def _project_points(pts: np.ndarray, V: np.ndarray, P: np.ndarray,
                    width: int, height: int) -> np.ndarray:
    """Return (M, 2) pixel coordinates of the points that project in front
    of the camera AND fall inside the frame."""
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), np.float32)], axis=1)
    v = pts_h @ V.T
    c = v @ P.T
    ok = (c[:, 3] > 1e-6) & (v[:, 2] > 0)
    ndc = c[ok, :2] / c[ok, 3:4]
    uv = ndc * 0.5 + 0.5
    px = uv * np.array([width, height], np.float32)
    in_frame = (px[:, 0] >= 0) & (px[:, 0] < width) & (px[:, 1] >= 0) & (px[:, 1] < height)
    return px[in_frame]


def _gt_bbox(image_path: Path, target_w: int, target_h: int) -> tuple[int, int, int, int]:
    """Load a GT PNG, downsample to the target resolution, return the
    bounding box ``(x0, y0, x1, y1)`` of non-black foreground pixels.
    """
    img = Image.open(image_path).convert("RGB")
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.LANCZOS)
    arr = np.asarray(img, np.float32) / 255.0
    mask = arr.sum(-1) > 0.05
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def test_colmap_scene_loads() -> None:
    """Basic smoke: scene parses, produces the expected cam + point counts."""
    scene = _load_scene_or_skip()
    assert len(scene.images) == 20
    n_init = scene.initial_points["means"].shape[0]
    assert n_init > 0
    assert scene.initial_points["means"].shape == (n_init, 3)
    assert scene.initial_points["rot_quats"].shape == (n_init, 4)


def test_camera_projection_matches_gt_bbox() -> None:
    """For every camera: project the initial point cloud and assert the
    pixel-space bounding box matches the GT PNG's non-black bounding box
    within a small tolerance.

    Tolerance is set generously (12 px on each side) to absorb:
    - GT LANCZOS downsampling halo
    - the alpha-rendered edge of a Blender render being a few pixels
      softer than the point-cloud bbox
    """
    tol = 12
    width, height = 960, 540
    scene = _load_scene_or_skip(width, height)
    pts = scene.initial_points["means"]

    failures: list[str] = []
    for i, entry in enumerate(scene.images):
        V = entry.camera.view_matrix
        P = entry.camera.proj_matrix
        px = _project_points(pts, V, P, width, height)
        if px.shape[0] == 0:
            failures.append(f"cam {i}: no points project in frame")
            continue
        proj_bbox = (
            int(px[:, 0].min()), int(px[:, 1].min()),
            int(px[:, 0].max()), int(px[:, 1].max()),
        )
        gt_bbox = _gt_bbox(scene.image_dir / entry.image_name, width, height)

        dx0 = abs(proj_bbox[0] - gt_bbox[0])
        dy0 = abs(proj_bbox[1] - gt_bbox[1])
        dx1 = abs(proj_bbox[2] - gt_bbox[2])
        dy1 = abs(proj_bbox[3] - gt_bbox[3])
        worst = max(dx0, dy0, dx1, dy1)
        if worst > tol:
            failures.append(
                f"cam {i} ({entry.image_name}): proj_bbox={proj_bbox} "
                f"gt_bbox={gt_bbox}  max corner diff {worst} px > {tol}"
            )

    if failures:
        pytest.fail("\n".join(failures))


def test_camera_resolution_scaling() -> None:
    """Loading at a smaller resolution must scale fx/fy/cx/cy linearly.
    We verify by asserting the projected bbox at half-res is within 1 px
    (on each edge) of the full-res bbox, scaled down by 0.5.
    """
    scene_full = _load_scene_or_skip(1920, 1080)
    scene_half = _load_scene_or_skip(960, 540)
    pts_full = scene_full.initial_points["means"]
    pts_half = scene_half.initial_points["means"]
    # Same point cloud regardless of render resolution.
    np.testing.assert_array_equal(pts_full, pts_half)

    for (ef, eh) in list(zip(scene_full.images, scene_half.images))[:5]:
        px_full = _project_points(pts_full, ef.camera.view_matrix, ef.camera.proj_matrix, 1920, 1080)
        px_half = _project_points(pts_half, eh.camera.view_matrix, eh.camera.proj_matrix, 960, 540)
        bbox_full = np.array([px_full[:, 0].min(), px_full[:, 1].min(),
                              px_full[:, 0].max(), px_full[:, 1].max()]) * 0.5
        bbox_half = np.array([px_half[:, 0].min(), px_half[:, 1].min(),
                              px_half[:, 0].max(), px_half[:, 1].max()])
        np.testing.assert_allclose(bbox_full, bbox_half, atol=1.0)


def test_view_matrix_inverse_is_camera_position() -> None:
    """COLMAP convention: ``x_cam = R x_world + t`` so camera position in
    world is ``-R^T t``. Verify that all 20 cameras are positioned roughly
    on a sphere around the scene origin (consistent with Blender's Suzanne
    turntable rig)."""
    scene = _load_scene_or_skip()
    cam_positions: list[np.ndarray] = []
    for entry in scene.images:
        V = entry.camera.view_matrix
        R = V[:3, :3]
        t = V[:3, 3]
        cam_world = -R.T @ t
        cam_positions.append(cam_world)

    cam_positions = np.stack(cam_positions, axis=0)
    distances = np.linalg.norm(cam_positions, axis=-1)
    # Suzanne turntable rig: all 20 cams at roughly the same radius from origin.
    assert distances.std() / distances.mean() < 0.05, (
        f"cam distances vary too much (mean={distances.mean():.2f}, std={distances.std():.2f}); "
        "suggests the view-matrix interpretation is broken"
    )
