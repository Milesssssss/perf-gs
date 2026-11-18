# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

import cv2
import numpy as np
import pytest

from trainer.data.colmap import ColmapDataset, ColmapImageEntry, _projection_matrix
from trainer.data.dataset import Camera
from trainer.data.stereo import seed_from_stereo


@pytest.fixture
def plane_scene(tmp_path):
    """Three different image sizes observing one textured plane four world units away."""
    rng = np.random.default_rng(22)
    pattern = rng.integers(0, 256, (640, 1024), dtype=np.uint8)
    texture_bgr = np.dstack([pattern // 4, pattern // 2, pattern])
    entries = []
    for i, (width, height, eye_x) in enumerate([(640, 480, -0.25), (608, 460, 0), (672, 494, 0.25)]):
        focal = width * 1.1
        y, x = np.mgrid[:height, :width].astype(np.float32)
        # The plane texture has 150 texels per world unit in both directions.
        texture_x = 500 + 150 * (eye_x + 4 * (x + 0.5 - width / 2) / focal)
        texture_y = 300 + 150 * 4 * (y + 0.5 - height / 2) / focal
        pixels = cv2.remap(texture_bgr, texture_x, texture_y, cv2.INTER_LINEAR)
        filename = f"train{i}.png"
        assert cv2.imwrite(str(tmp_path / filename), pixels)
        view = np.eye(4, dtype=np.float32)
        view[0, 3] = -eye_x
        projection = _projection_matrix(focal, focal, width / 2, height / 2, width, height)
        entries.append(ColmapImageEntry(i, filename, Camera(view, projection, (512, 384))))
    # No pose, image name, or file exists for the held-out camera.
    entries.insert(1, object())
    points = {
        "means": np.array([[0, 0, 4]], dtype=np.float32),
        "log_scales": np.zeros((1, 3), dtype=np.float32),
        "rot_quats": np.array([[1, 0, 0, 0]], dtype=np.float32),
        "opacities": np.array([0.1], dtype=np.float32),
        "color_logits": np.zeros((1, 3), dtype=np.float32),
    }
    return ColmapDataset(entries, tmp_path, points)


def test_stereo_triangulates_training_views(plane_scene):
    points = plane_scene.initial_points
    original = {key: value.copy() for key, value in points.items()}
    assert seed_from_stereo(plane_scene, [0, 2, 3], limit=512) == 512
    for key, value in points.items():
        np.testing.assert_array_equal(value[:1], original[key])
        assert len(value) == 513 and np.isfinite(value).all()
    assert np.median(abs(points["means"][1:, 2] - 4.0)) < 0.05
    assert np.median(points["color_logits"][1:, 0] - points["color_logits"][1:, 2]) > 0.5


def test_coincident_cameras_preserve_original_cloud(plane_scene):
    for index in (2, 3):
        plane_scene.images[index].camera.view_matrix[:] = plane_scene.images[0].camera.view_matrix
    original = {key: value.copy() for key, value in plane_scene.initial_points.items()}
    assert seed_from_stereo(plane_scene, [0, 2, 3]) == 0
    for key, value in plane_scene.initial_points.items():
        np.testing.assert_array_equal(value, original[key])


@pytest.mark.parametrize(("capacity", "expected_limit"), [(1, None), (2, None), (17, 15), (5_600_000, 256_000)])
def test_scene_stereo_budget_respects_requested_capacity(plane_scene, capacity, expected_limit, monkeypatch):
    from trainer.data import stereo
    from trainer.frontend import scene
    from trainer.frontend.cli import parse_args

    del plane_scene.images[1]  # Use all three real views in this capacity test.
    points = plane_scene.initial_points
    points.update({k: np.repeat(v, 2, axis=0) for k, v in points.items()})
    original = points["means"].copy()
    monkeypatch.setattr(scene, "load_colmap", lambda *args, **kwargs: plane_scene)
    limits = []

    def observe(dataset, train_indices, limit):
        limits.append(limit)
        return seed_from_stereo(dataset, train_indices, limit)

    monkeypatch.setattr(stereo, "seed_from_stereo", observe)
    args = parse_args(["--stereo-init", "--max-splats", str(capacity), "--eval", "0"])
    context = scene.load_scene_context(args)
    np.testing.assert_array_equal(points["means"][:2], original)
    assert limits == ([] if expected_limit is None else [expected_limit])
    assert context.initial_gaussian_count == len(points["means"])
    if expected_limit is None:
        assert context.initial_gaussian_count == 2
    elif capacity == 17:
        assert context.initial_gaussian_count == capacity
    else:
        from dataclasses import replace
        baseline = replace(plane_scene, initial_points={key: value[:2].copy() for key, value in points.items()})
        assert seed_from_stereo(baseline, context.train_indices) > 0
        for key, actual in points.items():
            np.testing.assert_array_equal(actual, baseline.initial_points[key])
