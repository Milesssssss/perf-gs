"""Output helpers: write rendered images to PNG, save gaussians as .ply."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from PIL import Image

from trainer.backend.render.pipeline import Pipeline
from trainer.data.dataset import Camera, _SH_C0
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

    Inverse of :func:`trainer.dataset.load_ply`:
    - The GPU stores opacity as a pre-sigmoid logit; the .ply format
      also stores pre-sigmoid opacity, so we write it directly.
    - ``sh0`` is the SH band-0 DC coeff: color = 0.5 + C0*sh0. With the sRGB
      migration this color IS sRGB (matches the 3DGS PLY convention: viewers
      render SH_C0*f_dc + 0.5 as sRGB), so f_dc == sh0 directly — no round-trip.

    Only the 14 fields M1/M2 produce are written; higher-order SH bands
    land in a later milestone when they matter.
    """
    n = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    if n == 0:
        raise ValueError("pipeline has 0 gaussians; nothing to save")

    raw = pipeline.buffers.gaussians.to_numpy()
    g = np.frombuffer(raw.tobytes(), dtype=gaussian_numpy_dtype())[:n]

    mean = g["mean"]
    log_scale = g["log_scale"]
    rot_quat = g["rot_quat"]
    opacity_logit = g["opacity"].astype(np.float32)   # stored as logit already
    sh0 = g["sh0"]

    # Our renderer now composites in sRGB: color = 0.5 + SH_C0 * sh0 IS sRGB,
    # matching the 3DGS PLY convention (viewers render SH_C0*f_dc + 0.5 as sRGB),
    # so f_dc = sh0 directly — no linear<->sRGB round-trip.
    f_dc = sh0.astype(np.float32)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    records = np.empty(n, dtype=dtype)
    records["x"] = mean[:, 0]
    records["y"] = mean[:, 1]
    records["z"] = mean[:, 2]
    records["nx"] = 0.0
    records["ny"] = 0.0
    records["nz"] = 0.0
    records["f_dc_0"] = f_dc[:, 0]
    records["f_dc_1"] = f_dc[:, 1]
    records["f_dc_2"] = f_dc[:, 2]
    records["opacity"] = opacity_logit
    records["scale_0"] = log_scale[:, 0]
    records["scale_1"] = log_scale[:, 1]
    records["scale_2"] = log_scale[:, 2]
    records["rot_0"] = rot_quat[:, 0]
    records["rot_1"] = rot_quat[:, 1]
    records["rot_2"] = rot_quat[:, 2]
    records["rot_3"] = rot_quat[:, 3]

    elem = PlyElement.describe(records, "vertex")
    PlyData([elem], text=False).write(str(path))
