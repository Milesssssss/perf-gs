# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""v1.1 training entry: overfit all 20 Suzanne views at half-res.

No densification, pruning or SH; loss is L2 in linear space.
Usage: python scripts/running/v1_train.py --iters 3000 --scene datasets/synthetic/suzanne_blender
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from trainer.common.color import srgb_to_linear
from trainer.data.colmap import load_colmap, load_colmap_image
from trainer.backend.compute.device import create_device
from trainer.backend.export import render_to_png
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.config.config import AdamLRs


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return srgb_to_linear(c)


def _psnr_linear(linear_rendered: np.ndarray, linear_target: np.ndarray) -> float:
    mse = float(np.mean((linear_rendered - linear_target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def _save_srgb_png(rgb_linear: np.ndarray, path: Path) -> None:
    """Save linear RGB as an 8-bit PNG with the same display gamma as ``render_to_png``."""
    rgb = np.clip(rgb_linear, 0.0, 1.0)
    rgb_srgb = np.power(rgb, 1.0 / 2.2)
    rgba = np.concatenate([rgb_srgb, np.ones_like(rgb[..., :1])], axis=-1)
    png = (rgba * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(png, mode="RGBA").save(path)


def _render_all_cameras(pipeline: TrainingPipeline, scene, iter_idx: int, cams_dir: Path) -> None:
    for cam_i, entry in enumerate(scene.images):
        pipeline.render(entry.camera)
        img = pipeline.read_output_image()[..., :3]
        out_path = cams_dir / f"cam_{cam_i:02d}" / f"iter_{iter_idx:06d}.png"
        _save_srgb_png(img, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="datasets/synthetic/suzanne_blender")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--preview-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--backend", type=str, default="vulkan", choices=["vulkan", "d3d12"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", type=str, default=None,
                    help="name of output subfolder; defaults to timestamp")
    args = ap.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs/train") / run_name
    cams_dir = run_dir / "cams"
    run_dir.mkdir(parents=True, exist_ok=True)
    cams_dir.mkdir(parents=True, exist_ok=True)

    scene = load_colmap(args.scene, width=args.width, height=args.height)
    n_cams = len(scene.images)
    n_init = scene.initial_points["means"].shape[0]
    print(f"loaded {n_cams} cameras, {n_init} init points")

    # Write config up front so even a crashed run leaves breadcrumbs.
    (run_dir / "config.json").write_text(json.dumps({
        "scene": args.scene,
        "iters": args.iters,
        "width": args.width,
        "height": args.height,
        "preview_every": args.preview_every,
        "log_every": args.log_every,
        "backend": args.backend,
        "seed": args.seed,
        "run_name": run_name,
        "n_cams": n_cams,
        "n_init_points": n_init,
    }, indent=2))

    targets_srgb = [load_colmap_image(e, scene.image_dir, args.width, args.height)
                    for e in scene.images]
    targets_linear = [_srgb_to_linear(t) for t in targets_srgb]
    for cam_i, (entry, tgt) in enumerate(zip(scene.images, targets_srgb)):
        cam_dir = cams_dir / f"cam_{cam_i:02d}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        # gt.png is already-sRGB (loaded from PNG as sRGB and handed straight back).
        Image.fromarray((tgt * 255 + 0.5).clip(0, 255).astype(np.uint8),
                        mode="RGB").save(cam_dir / "gt.png")

    device = create_device(backend=args.backend)
    pipeline = TrainingPipeline(
        device=device,
        width=args.width, height=args.height,
        max_splats=max(n_init, 1024),
        initial_state=scene.initial_points,
    )

    _render_all_cameras(pipeline, scene, 0, cams_dir)

    log_txt = (run_dir / "log.txt").open("w", buffering=1)
    log_json = (run_dir / "log.jsonl").open("w", buffering=1)
    log_txt.write(f"# v1.1 training log — scene={args.scene}, backend={args.backend}\n")
    log_txt.write(f"# iters={args.iters}, resolution={args.width}x{args.height}, n_cams={n_cams}\n")

    rng = np.random.default_rng(args.seed)

    # LR decay on lr_mean only.
    lr_mean_init = 1.6e-4
    lr_mean_final = lr_mean_init * 1e-2
    decay = (lr_mean_final / lr_mean_init) ** (1.0 / max(args.iters, 1))

    t_start = time.time()
    rolling_loss = None

    for it in range(1, args.iters + 1):
        cam_idx = int(rng.integers(0, n_cams))
        entry = scene.images[cam_idx]
        pipeline.targets.set_image(targets_srgb[cam_idx])

        lrs = AdamLRs(mean=lr_mean_init * (decay ** (it - 1)))
        pipeline.train_step(entry.camera, lrs=lrs)

        rendered = pipeline.read_output_image()[..., :3]
        mse = float(np.mean((rendered - targets_linear[cam_idx]) ** 2))
        rolling_loss = mse if rolling_loss is None else 0.98 * rolling_loss + 0.02 * mse
        psnr = _psnr_linear(rendered, targets_linear[cam_idx])

        if it % args.log_every == 0 or it == 1:
            elapsed = time.time() - t_start
            it_per_s = it / elapsed
            line = (f"iter {it:6d}  cam {cam_idx:2d}  "
                    f"loss={rolling_loss:.5f}  psnr={psnr:5.2f} dB  "
                    f"lr_mean={lrs.mean:.2e}  {it_per_s:5.1f} it/s")
            print(line)
            log_txt.write(line + "\n")
            log_json.write(json.dumps({
                "iter": it,
                "cam_idx": cam_idx,
                "loss_rolling": rolling_loss,
                "loss_instant": mse,
                "psnr_instant": psnr,
                "lr_mean": lrs.mean,
                "it_per_s": it_per_s,
                "elapsed_s": elapsed,
            }) + "\n")

        if it % args.preview_every == 0 or it == args.iters:
            _render_all_cameras(pipeline, scene, it, cams_dir)

    log_txt.close()
    log_json.close()

    print("computing final per-camera PSNR...")
    per_cam: list[dict] = []
    total_mse = 0.0
    for cam_i, entry in enumerate(scene.images):
        pipeline.render(entry.camera)
        rendered = pipeline.read_output_image()[..., :3]
        mse = float(np.mean((rendered - targets_linear[cam_i]) ** 2))
        psnr = _psnr_linear(rendered, targets_linear[cam_i])
        total_mse += mse
        per_cam.append({"cam_idx": cam_i, "mse": mse, "psnr": psnr,
                        "image_name": entry.image_name})
    avg_mse = total_mse / n_cams
    avg_psnr = -10.0 * math.log10(avg_mse) if avg_mse > 1e-12 else 99.0
    summary = {
        "avg_mse": avg_mse,
        "avg_psnr": avg_psnr,
        "per_cam": per_cam,
        "n_cams": n_cams,
        "iters": args.iters,
    }
    (run_dir / "final_psnr.json").write_text(json.dumps(summary, indent=2))

    print(f"done in {time.time() - t_start:.1f}s")
    print(f"avg PSNR across all {n_cams} cams: {avg_psnr:.2f} dB")
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
