# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Output helpers: write rendered images to PNG, save gaussians as .ply."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from trainer.backend.render.pipeline import TrainingPipeline
from trainer.data.dataset import Camera
from trainer.data.ply import write_gaussian_ply


def render_to_png(pipeline: TrainingPipeline, camera: Camera, path: str | Path) -> None:
    """Render once and save the result as an 8-bit PNG; the output is already sRGB (no gamma)."""
    pipeline.render(camera)
    img = pipeline.read_output_image()

    assert img.ndim == 3 and img.shape[2] == 4, f"unexpected image shape {img.shape}"

    rgb = np.clip(img[..., :3], 0.0, 1.0)
    rgba = np.concatenate([rgb, np.ones_like(rgb[..., :1])], axis=-1)
    png_uint8 = (rgba * 255.0 + 0.5).astype(np.uint8)

    Image.fromarray(png_uint8, mode="RGBA").save(path)


def save_ply(pipeline: TrainingPipeline, path: str | Path) -> None:
    """Export the live gaussians as a 3DGS-format .ply (inverse of ``load_ply``)."""
    gaussians = pipeline.primitives.read()
    if len(gaussians) == 0:
        raise ValueError("pipeline has 0 gaussians; nothing to save")
    features_rest = (pipeline.primitives.read_features_rest(len(gaussians))
                     if pipeline.buffers.sh_enabled else None)
    write_gaussian_ply(gaussians, path, features_rest=features_rest,
                       natural_vignetting=pipeline.rasterizer.natural_vignetting)
