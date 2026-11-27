# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""A COLMAP model with per-view image sizes must load.

``colmap image_undistorter`` emits one PINHOLE camera per image, each cropped to
that view's own valid region, so sizes differ by a few pixels across one scene.
The loader used to reject that outright ("scene has mixed native camera sizes"),
which made every undistorted reconstruction — including everything our own
reconstruction pipeline produces — unloadable.

The renderer has one target resolution, so the contract is: render at the most
common native size, and scale each camera's intrinsics by ITS OWN size so the
projection still matches the pixels it is compared against.
"""

from __future__ import annotations

import numpy as np
import pytest

from trainer.data.colmap import load_colmap, load_colmap_image

from _colmap_testkit import MIXED_VIEWS as VIEWS, write_colmap_scene

cv2 = pytest.importorskip("cv2")


def _write_scene(root, views=VIEWS):
    return write_colmap_scene(root, views)


@pytest.mark.parametrize("root, scene, requested, expected", [
    ("360_v2", "room", None, 2),
    ("360_v2", "bicycle", None, 4),
    ("360_v2", "room", 1, 1),
    ("my_capture", "room", None, 1),
    ("relocated_dataset", "room", None, 2),
])
def test_training_resolution_uses_benchmark_pyramid(tmp_path, root, scene, requested, expected):
    path = _write_scene(tmp_path / root / scene, [("a.png", 100, 80, 120.0)])
    if root == "relocated_dataset":
        np.save(path / "poses_bounds.npy", np.zeros((1, 17)))
    for factor in (2, 4):
        directory = path / f"images_{factor}"
        directory.mkdir()
        cv2.imwrite(str(directory / "a.png"), np.zeros((80 // factor, 100 // factor, 3), np.uint8))
    dataset = load_colmap(path, downsample=requested)
    assert dataset.downsample == expected
    assert dataset.image_dir.name == ("images" if expected == 1 else f"images_{expected}")
    assert dataset.images[0].camera.image_size == (100 // expected, 80 // expected)


def test_mixed_camera_sizes_load_at_the_most_common_size(tmp_path) -> None:
    scene = _write_scene(tmp_path / "recon")

    dataset = load_colmap(scene)

    assert len(dataset.images) == len(VIEWS)
    # 3 of 4 views are 100x80, so that is the render target — those three are then
    # read off disk with no resampling at all.
    assert {tuple(e.camera.image_size) for e in dataset.images} == {(100, 80)}


def test_each_camera_keeps_its_own_intrinsics(tmp_path) -> None:
    scene = _write_scene(tmp_path / "recon")

    dataset = load_colmap(scene)

    # P[0][0] == 2 * fx * (target_w / native_w) / target_w == 2 * fx / native_w.
    # The odd-sized view is scaled by ITS 98px width, not the target's 100.
    by_name = {e.image_name: e.camera for e in dataset.images}
    for name, native_w, _native_h, fx in VIEWS:
        assert by_name[name].proj_matrix[0][0] == pytest.approx(
            2.0 * fx / native_w, rel=1e-6)


def test_a_uniform_model_is_unaffected(tmp_path) -> None:
    """The fix must not perturb the ordinary same-size case."""
    uniform = [(name, 100, 80, fx) for name, _w, _h, fx in VIEWS]
    scene = _write_scene(tmp_path / "recon", views=uniform)

    dataset = load_colmap(scene)

    assert {tuple(e.camera.image_size) for e in dataset.images} == {(100, 80)}
    by_name = {e.image_name: e.camera for e in dataset.images}
    for name, _w, _h, fx in uniform:
        assert by_name[name].proj_matrix[0][0] == pytest.approx(2.0 * fx / 100)


def test_off_size_views_are_resampled_to_the_render_target(tmp_path) -> None:
    scene = _write_scene(tmp_path / "recon")
    dataset = load_colmap(scene)
    entries = {e.image_name: e for e in dataset.images}

    for name, _w, _h, _fx in VIEWS:
        image = load_colmap_image(entries[name], dataset.image_dir, 100, 80)
        assert image.shape == (80, 100, 3), name       # every target is uniform
        assert image.dtype == np.float32


def test_enlarging_interpolates_instead_of_duplicating_pixels(tmp_path) -> None:
    """The undistorted path can ask for a target a pixel or two LARGER than the
    source, where INTER_AREA degenerates to nearest-neighbour."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    ramp = np.tile(np.arange(0, 64, 8, dtype=np.uint8), (4, 1))   # 8 wide, 4 tall
    cv2.imwrite(str(images_dir / "ramp.png"), np.dstack([ramp] * 3))
    entry = type("E", (), {"image_name": "ramp.png"})()

    grown = load_colmap_image(entry, images_dir, 15, 4)

    assert grown.shape == (4, 15, 3)
    source = set(np.unique(ramp / 255.0).round(6))
    produced = set(np.unique(grown).round(6))
    assert produced - source, "enlarging only reproduced source values (nearest)"
