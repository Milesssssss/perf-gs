"""Profile a single v2 training step, per phase.

Prints per-phase GPU time (from timestamp queries), wall-clock total,
and relative cost. Use to find the bottleneck before reaching for
Nsight/PIX — which struggle to hook a headless compute-only Python
process anyway.

Usage:
    .venv/Scripts/python.exe scripts/profile_step.py
        [--scene datasets/scans/cactus] [--width 532] [--height 947]
        [--z-near-cull 0.2] [--backend vulkan]
"""
from __future__ import annotations

import argparse

from trainer.colmap import load_colmap, load_image
from trainer.device import create_device
from trainer.pipeline import AdamLRs, Pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="datasets/scans/cactus")
    ap.add_argument("--width", type=int, default=532)
    ap.add_argument("--height", type=int, default=947)
    ap.add_argument("--z-near-cull", type=float, default=0.2)
    ap.add_argument("--backend", type=str, default="vulkan",
                    choices=["vulkan", "d3d12"])
    ap.add_argument("--max-splats", type=int, default=500_000)
    ap.add_argument("--max-sort-pairs", type=int, default=4_000_000)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    scene = load_colmap(args.scene, width=args.width, height=args.height)
    print(f"loaded {len(scene.images)} cameras, "
          f"{scene.initial_points['means'].shape[0]} init points")

    device = create_device(backend=args.backend)
    pipeline = Pipeline(
        device=device, width=args.width, height=args.height,
        max_splats=args.max_splats, max_sort_pairs=args.max_sort_pairs,
    )
    pipeline.z_near_cull = args.z_near_cull
    pipeline.set_gaussians_from_numpy(**scene.initial_points)
    pipeline.clear_tracking()

    target_srgb = load_image(scene.images[0], scene.image_dir,
                              args.width, args.height)
    tex = pipeline.create_target_texture(target_srgb)
    pipeline.set_target_texture(tex)

    pipeline.profile_one_step(scene.images[0].camera, warmup_steps=args.warmup)


if __name__ == "__main__":
    main()
