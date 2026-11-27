# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Training resolution defaults must preserve captures without changing benchmarks."""

from __future__ import annotations

import numpy as np
import pytest
from _colmap_testkit import write_colmap_scene

from trainer.data.colmap import load_colmap, load_colmap_image

cv2 = pytest.importorskip("cv2")


def _scene(path, width=100, height=80):
    return write_colmap_scene(path, [("a.png", width, height, 120.0)])


def _pyramid(path, factor, size=None):
    width, height = size if size is not None else (100 // factor, 80 // factor)
    directory = path / f"images_{factor}"
    directory.mkdir(exist_ok=True)
    assert cv2.imwrite(str(directory / "a.png"),
                       np.full((height, width, 3), factor * 32, np.uint8))


def _release_marker(path):
    np.save(path / "poses_bounds.npy", np.zeros((1, 17)))


@pytest.mark.parametrize("scene_name", ["warehouse", "room", "bicycle"])
def test_custom_scene_auto_resolution_is_native(tmp_path, scene_name):
    path = _scene(tmp_path / "captures" / scene_name)

    dataset = load_colmap(path, downsample=None)

    assert dataset.downsample == 1
    assert dataset.image_dir == path / "images"
    assert dataset.images[0].camera.image_size == (100, 80)


@pytest.mark.parametrize("relative_path, marker, pyramids, siblings", [
    ("captures/room", False, (2, 4), False),
    ("captures/room", True, (), False),
    ("captures/room", True, (4,), False),
    ("captures/bicycle", True, (2,), False),
    ("360_v2/warehouse", True, (2, 4), True),
    ("360_extra_scenes/room", False, (), False),
])
def test_incomplete_or_unrelated_benchmark_evidence_keeps_native_resolution(
        tmp_path, relative_path, marker, pyramids, siblings):
    path = _scene(tmp_path / relative_path)
    if marker:
        _release_marker(path)
    for factor in pyramids:
        _pyramid(path, factor)
    if siblings:
        for name in ("bicycle", "garden", "stump"):
            (path.parent / name).mkdir(exist_ok=True)

    dataset = load_colmap(path, downsample=None)

    assert dataset.downsample == 1
    assert dataset.image_dir == path / "images"
    assert dataset.images[0].camera.image_size == (100, 80)


@pytest.mark.parametrize("root, scene_name, factor", [
    ("360_v2", "bicycle", 4),
    ("360_v2", "bonsai", 2),
    ("360_v2", "counter", 2),
    ("360_v2", "garden", 4),
    ("360_v2", "kitchen", 2),
    ("360_v2", "room", 2),
    ("360_v2", "stump", 4),
    ("360_extra_scenes", "flowers", 4),
    ("360_extra_scenes", "treehill", 4),
    ("mipnerf360", "room", 2),
    ("mipnerf360_v2", "garden", 4),
])
def test_known_benchmark_layout_keeps_protocol_without_precomputed_images(
        tmp_path, root, scene_name, factor):
    path = _scene(tmp_path / root / scene_name)

    dataset = load_colmap(path, downsample=None)

    assert dataset.downsample == factor
    assert dataset.image_dir == path / "images"
    assert dataset.images[0].camera.image_size == (100 // factor, 80 // factor)


@pytest.mark.parametrize("scene_name, factor", [("room", 2), ("bicycle", 4), ("flowers", 4)])
def test_relocated_release_uses_its_matching_pyramid(tmp_path, scene_name, factor):
    path = _scene(tmp_path / "relocated" / scene_name)
    _release_marker(path)
    _pyramid(path, factor)

    dataset = load_colmap(path, downsample=None)

    assert dataset.downsample == factor
    assert dataset.image_dir == path / f"images_{factor}"
    assert dataset.images[0].camera.image_size == (100 // factor, 80 // factor)


@pytest.mark.parametrize("scene_name, factor", [("room", 2), ("garden", 4)])
def test_relocated_benchmark_siblings_keep_protocol(tmp_path, scene_name, factor):
    path = _scene(tmp_path / "relocated" / scene_name)
    for name in ("bicycle", "garden", "stump"):
        (path.parent / name).mkdir(exist_ok=True)

    dataset = load_colmap(path, downsample=None)

    assert dataset.downsample == factor
    assert dataset.images[0].camera.image_size == (100 // factor, 80 // factor)


@pytest.mark.parametrize("relative_path", ["captures/warehouse", "360_v2/room"])
@pytest.mark.parametrize("factor", [1, 2, 4])
def test_explicit_downsample_wins_over_auto_policy(tmp_path, relative_path, factor):
    path = _scene(tmp_path / relative_path)
    _pyramid(path, 2)
    _pyramid(path, 4)

    dataset = load_colmap(path, downsample=factor)

    assert dataset.downsample == factor
    assert dataset.image_dir == path / ("images" if factor == 1 else f"images_{factor}")
    assert dataset.images[0].camera.image_size == (100 // factor, 80 // factor)


def test_library_call_keeps_its_explicit_full_resolution_default(tmp_path):
    path = _scene(tmp_path / "360_v2" / "room")
    _pyramid(path, 2)

    dataset = load_colmap(path)

    assert dataset.downsample == 1
    assert dataset.image_dir == path / "images"
    assert dataset.images[0].camera.image_size == (100, 80)


@pytest.mark.parametrize("relative_path, factor, width, height, expected_size", [
    ("captures/warehouse", None, 40, None, (40, 80)),
    ("captures/warehouse", 2, None, 30, (50, 30)),
    ("360_v2/room", None, 160, 90, (160, 90)),
    ("360_v2/bicycle", 4, 60, 45, (60, 45)),
])
def test_explicit_dimensions_override_only_the_requested_target_axes(
        tmp_path, relative_path, factor, width, height, expected_size):
    path = _scene(tmp_path / relative_path)
    _pyramid(path, 2)
    _pyramid(path, 4)

    dataset = load_colmap(path, downsample=factor, width=width, height=height)

    camera = dataset.images[0].camera
    assert camera.image_size == expected_size
    assert camera.proj_matrix[0, 0] == pytest.approx(2 * 120.0 / 100)
    assert camera.proj_matrix[1, 1] == pytest.approx(2 * 120.0 / 80)


def test_odd_sized_pyramid_uses_actual_pixels_and_scaled_intrinsics(tmp_path):
    path = _scene(tmp_path / "captures" / "warehouse", width=101, height=81)
    # A real downsampler may round upward, unlike integer division in the fallback.
    _pyramid(path, 2, size=(51, 41))
    (path / "sparse" / "0" / "cameras.txt").write_text(
        "1 PINHOLE 101 81 121 117 48.25 36.5\n")

    dataset = load_colmap(path, downsample=2)

    entry = dataset.images[0]
    assert dataset.image_dir == path / "images_2"
    assert entry.camera.image_size == (51, 41)
    projection = entry.camera.proj_matrix
    assert projection[0, 0] == pytest.approx(2 * 121 / 101)
    assert projection[1, 1] == pytest.approx(2 * 117 / 81)
    assert projection[0, 2] == pytest.approx(2 * 48.25 / 101 - 1)
    assert projection[1, 2] == pytest.approx(2 * 36.5 / 81 - 1)
    loaded = load_colmap_image(entry, dataset.image_dir, 51, 41)
    assert loaded.shape == (41, 51, 3)
    np.testing.assert_allclose(loaded, 64 / 255, rtol=1e-6)
