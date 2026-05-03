"""Verify tile-binned forward sorts correctly.

Scene: a red gaussian at z=3 (far), a green gaussian at z=2 (near) covering
the same screen region. With front-to-back blending the near green should
dominate. If we accidentally walked the tile in reverse buffer order the
pixel would be red instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from trainer.dataset import simple_camera
from trainer.device import create_device
from trainer.export import render_to_png
from trainer.pipeline import Pipeline


def main() -> None:
    W, H = 256, 256
    device = create_device(backend="vulkan")
    pipeline = Pipeline(device=device, width=W, height=H, max_splats=16)

    # Far red then near green. Deliberately ordered "wrong" in the buffer
    # so any unsorted path would blend red first.
    pipeline.set_gaussians_from_numpy(
        means=np.array([
            [0.0, 0.0, 3.0],  # FAR red
            [0.0, 0.0, 2.0],  # NEAR green
        ], dtype=np.float32),
        log_scales=np.log(np.array([
            [0.2, 0.2, 0.2],
            [0.2, 0.2, 0.2],
        ], dtype=np.float32)),
        rot_quats=np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ], dtype=np.float32),
        opacities=np.array([0.95, 0.95], dtype=np.float32),
        color_logits=np.array([
            [5.0, -5.0, -5.0],  # red
            [-5.0, 5.0, -5.0],  # green
        ], dtype=np.float32),
    )

    camera = simple_camera(
        width=W, height=H,
        eye=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        fovy_degrees=60.0,
    )

    out_path = Path("outputs/m3_occlusion.png")
    render_to_png(pipeline, camera, out_path)

    img = pipeline.read_output_image()
    center = img[H // 2, W // 2, :3]
    print(f"center pixel RGB (linear): {center}")
    print(f"expected: green-dominant (near green occludes far red)")
    r, g, b = center
    if g > r * 1.5:
        print("PASS: near green dominates")
    else:
        print("FAIL: near green did NOT dominate")


if __name__ == "__main__":
    main()
