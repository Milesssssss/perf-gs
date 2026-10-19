# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Builds a minimal on-disk COLMAP text model for tests.

Shared so the loader tests and the hub-preview tests describe the same kind of
scene — in particular an UNDISTORTED one, where every view has its own camera
record and sizes differ by a few pixels.
"""

from __future__ import annotations

import numpy as np

# (name, width, height, fx) — 3 views at 100x80 (the majority), 1 at 98x79.
MIXED_VIEWS = [
    ("a.png", 100, 80, 120.0),
    ("b.png", 100, 80, 121.0),
    ("c.png", 100, 80, 122.0),
    ("d.png", 98, 79, 118.0),
]


def write_colmap_scene(root, views=MIXED_VIEWS, points: int = 8):
    """Write ``sparse/0/{cameras,images,points3D}.txt`` plus matching images."""
    import cv2

    sparse = root / "sparse" / "0"
    sparse.mkdir(parents=True)
    images_dir = root / "images"
    images_dir.mkdir()

    cameras = ["# camera list"]
    images = ["# image list"]
    for i, (name, w, h, fx) in enumerate(views, start=1):
        cameras.append(f"{i} PINHOLE {w} {h} {fx} {fx} {w / 2} {h / 2}")
        # Identity rotation, cameras marching along +x so the extent is nonzero.
        images.append(f"{i} 1 0 0 0 {i} 0 0 {i} {name}")
        # The parser reads metadata/feature lines strictly alternating, so the
        # feature line has to be present even though its content is ignored.
        images.append("0.0 0.0 -1")
        cv2.imwrite(str(images_dir / name), np.full((h, w, 3), i * 20, np.uint8))
    (sparse / "cameras.txt").write_text("\n".join(cameras) + "\n")
    (sparse / "images.txt").write_text("\n".join(images) + "\n")

    rows = ["# 3D points"]
    for p in range(points):
        rows.append(f"{p} {p * 0.1} {p * 0.2} {p * 0.3} 200 100 50 0.5 0")
    (sparse / "points3D.txt").write_text("\n".join(rows) + "\n")
    return root
