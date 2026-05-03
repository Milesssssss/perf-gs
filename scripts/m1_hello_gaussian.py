"""M1 entry — render a single hand-authored 3D gaussian to a PNG.

Usage:
    .venv/Scripts/python.exe scripts/m1_hello_gaussian.py

Produces ``outputs/m1.png`` — a fuzzy red blob near the image center.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trainer.dataset import simple_camera
from trainer.device import create_device
from trainer.export import render_to_png
from trainer.pipeline import Pipeline


def main() -> None:
    width, height = 512, 512

    device = create_device(backend="vulkan")
    pipeline = Pipeline(device=device, width=width, height=height, max_splats=1024)

    # One gaussian at +Z=2, isotropic scale 0.3 world-units, identity rotation,
    # high opacity, red (logit -> sigmoid stays roughly the same for moderate
    # magnitudes; 2.0 ≈ 0.88, -2.0 ≈ 0.12, giving a mostly-red color).
    pipeline.set_gaussians_from_numpy(
        means=np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
        log_scales=np.log(np.array([[0.3, 0.3, 0.3]], dtype=np.float32)),
        rot_quats=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([0.9], dtype=np.float32),
        color_logits=np.array([[2.0, -2.0, -2.0]], dtype=np.float32),
    )

    camera = simple_camera(
        width=width,
        height=height,
        eye=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        fovy_degrees=60.0,
    )

    out_path = Path("outputs/m1.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_to_png(pipeline, camera, out_path)

    rendered = pipeline.read_output_image()
    nonzero = (rendered[..., :3].sum(axis=-1) > 1e-3).sum()
    print(f"wrote {out_path}")
    print(f"image stats: min={rendered.min():.4f} max={rendered.max():.4f} "
          f"non-black pixels={nonzero}")


if __name__ == "__main__":
    main()
