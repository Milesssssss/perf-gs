"""M2 entry — load a 3DGS .ply (or synthesize a test scene) and render it.

Usage:
    .venv/Scripts/python.exe scripts/m2_ply.py [--input path/to/scene.ply]

With no --input, generates 1000 gaussians arranged on a sphere of radius 1.

Produces:
  outputs/m2.png             — rendered PNG
  outputs/m2_roundtrip.ply   — gaussians read back from GPU, written out

If --input was given, loading outputs/m2_roundtrip.ply in a third-party
viewer (e.g. SuperSplat) should match the input visually.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from trainer.dataset import Camera, load_ply, simple_camera
from trainer.device import create_device
from trainer.export import render_to_png, save_ply
from trainer.pipeline import Pipeline


def _synthetic_sphere(n: int = 1000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Uniform points on a unit sphere.
    theta = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
    phi = np.arccos(1 - 2 * rng.uniform(0, 1, n)).astype(np.float32)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    means = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Place the sphere at +Z=4 so the forward-facing camera at origin sees it.
    means[:, 2] += 4.0

    log_scales = np.full((n, 3), np.log(0.03), dtype=np.float32)
    rot_quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    opacities = np.full((n,), 0.8, dtype=np.float32)
    # Color each gaussian by its position — a cheap visual signal so you
    # can tell orientation/placement is right.
    color_rgb = (means - means.min(0)) / (means.max(0) - means.min(0) + 1e-6)
    color_rgb = np.clip(color_rgb, 1e-3, 1.0 - 1e-3)
    color_logits = np.log(color_rgb / (1.0 - color_rgb)).astype(np.float32)

    return {
        "means": means,
        "log_scales": log_scales,
        "rot_quats": rot_quats,
        "opacities": opacities,
        "color_logits": color_logits,
    }


def _frame_camera(means: np.ndarray, width: int, height: int) -> Camera:
    """Point a camera at the centroid of the gaussians, offset along -Z so
    the whole cloud is in view with forward=+Z."""
    centroid = means.mean(axis=0)
    extent = np.linalg.norm(means - centroid, axis=-1).max() + 1e-6
    # Place eye in front of the cloud along -Z relative to centroid.
    eye = centroid - np.array([0.0, 0.0, 2.5 * extent], dtype=np.float32)
    return simple_camera(
        width=width, height=height,
        eye=tuple(eye.tolist()),
        target=tuple(centroid.tolist()),
        up=(0.0, 1.0, 0.0),
        fovy_degrees=60.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="M2: load or synthesize gaussians, render + roundtrip .ply")
    parser.add_argument("--input", type=str, default=None, help="Optional .ply to load")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()

    if args.input:
        print(f"loading {args.input}")
        data = load_ply(args.input)
        n = data["means"].shape[0]
        print(f"  {n} gaussians")
    else:
        print("no --input given, synthesizing sphere scene")
        data = _synthetic_sphere(n=1000)
        n = data["means"].shape[0]

    device = create_device(backend="vulkan")
    pipeline = Pipeline(
        device=device, width=args.width, height=args.height,
        max_splats=max(n, 1024),
    )
    pipeline.set_gaussians_from_numpy(**data)

    camera = _frame_camera(data["means"], args.width, args.height)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    render_to_png(pipeline, camera, out_dir / "m2.png")
    save_ply(pipeline, out_dir / "m2_roundtrip.ply")

    rendered = pipeline.read_output_image()
    nonzero = (rendered[..., :3].sum(axis=-1) > 1e-3).sum()
    print(f"rendered {n} gaussians -> outputs/m2.png ({nonzero} lit pixels)")
    print(f"round-trip .ply -> outputs/m2_roundtrip.ply")


if __name__ == "__main__":
    main()
