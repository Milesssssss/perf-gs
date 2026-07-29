"""Output helpers: write rendered images to PNG, save gaussians as .ply."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from trainer.backend.render.pipeline import Pipeline
from trainer.data.dataset import Camera
from trainer.data.ply import write_gaussian_ply
from trainer.contracts.codegen import gaussian_numpy_dtype


def render_to_png(pipeline: Pipeline, camera: Camera, path: str | Path) -> None:
    """Render once and save the result as an 8-bit sRGB PNG.

    COLORSPACE: the renderer composites in sRGB (matches INRIA 3DGS / gsplat /
    SADGS), so the output is already display-encoded — write it directly, no
    gamma. (Previously the renderer's output was treated as linear and this
    applied a 1/2.2 gamma; that was the pre-migration semantics.)
    """
    pipeline.render(camera)
    img = pipeline.read_output_image()

    assert img.ndim == 3 and img.shape[2] == 4, f"unexpected image shape {img.shape}"

    rgb = np.clip(img[..., :3], 0.0, 1.0)
    rgba = np.concatenate([rgb, np.ones_like(rgb[..., :1])], axis=-1)
    png_uint8 = (rgba * 255.0 + 0.5).astype(np.uint8)

    Image.fromarray(png_uint8, mode="RGBA").save(path)


def save_ply(pipeline: Pipeline, path: str | Path) -> None:
    """Export the pipeline's current gaussians as a 3DGS-format .ply.

    Reads the GPU buffers, then delegates the format serialization to
    :func:`trainer.data.ply.write_gaussian_ply` (inverse of
    :func:`trainer.data.ply.load_ply`).
    """
    n = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    if n == 0:
        raise ValueError("pipeline has 0 gaussians; nothing to save")

    raw = pipeline.buffers.gaussians.to_numpy()
    gaussians = np.frombuffer(raw.tobytes(), dtype=gaussian_numpy_dtype())[:n]
    write_gaussian_ply(gaussians, path)
