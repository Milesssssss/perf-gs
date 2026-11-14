# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The training targets of a scene: decoded images, linear copies, ground-truth PNGs."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from trainer.common.color import srgb_to_linear
from trainer.data.colmap import ColmapDataset, load_colmap_image
from trainer.data.parallel import default_workers_numbers, map_ordered


class TargetDataset:
    """Owns target loading, lazy linear conversion, and ground-truth images."""

    def __init__(self, scene: ColmapDataset, width: int, height: int, use_alpha: bool) -> None:
        self._scene = scene
        self._width = width
        self._height = height
        self._use_alpha = use_alpha
        # Decoding is startup's long pole and OpenCV releases the GIL, so decode on threads.
        started = time.time()
        workers = default_workers_numbers(len(scene.images))
        self._srgb = list(map_ordered(
            lambda entry: load_colmap_image(
                entry, scene.image_dir, width, height, load_alpha=use_alpha),
            scene.images, workers=workers))
        print(f"[data] loaded {len(self._srgb)} targets at {width}x{height} in "
              f"{time.time() - started:.1f}s ({workers} decode threads)", flush=True)
        self._linear_cache: dict[int, np.ndarray] = {}

    @property
    def srgb(self) -> list[np.ndarray]:
        return self._srgb

    def linear(self, camera_index: int) -> np.ndarray:
        target = self._linear_cache.get(camera_index)
        if target is None:
            target = srgb_to_linear(self._srgb[camera_index][..., :3])
            self._linear_cache[camera_index] = target
        return target

    def write_ground_truths(self, cameras_dir: Path) -> None:
        """One ``cam_XX/gt.png`` per camera, skipping those already written."""
        started = time.time()
        written = sum(map_ordered(
            lambda pair: self._write_ground_truth(cameras_dir, *pair),
            enumerate(self._srgb)))
        if written:
            print(f"[data] wrote {written} ground-truth PNGs in "
                  f"{time.time() - started:.1f}s", flush=True)

    def _write_ground_truth(self, cameras_dir: Path, camera_index: int,
                            target: np.ndarray) -> int:
        camera_dir = cameras_dir / f"cam_{camera_index:02d}"
        camera_dir.mkdir(parents=True, exist_ok=True)
        path = camera_dir / "gt.png"
        if path.exists():
            return 0
        if self._use_alpha and target.shape[-1] == 4:
            pixels = (target * 255 + 0.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(pixels, mode="RGBA").save(path)
        else:
            pixels = (target[..., :3] * 255 + 0.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(pixels, mode="RGB").save(path)
        return 1
