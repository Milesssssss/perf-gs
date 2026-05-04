"""Convergence diag variant C: densify starts late.

Runs 5000 iters. Densify every 100 iters in [densify_from, densify_from+1500].
No opacity reset. Compares densify_from=500 (our old default) vs 1500 vs 2500.

If the late-start trials converge and 500 doesn't, densify-too-early is
the primary failure mode for cactus.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from trainer.colmap import load_colmap, load_image
from trainer.device import create_device
from trainer.pipeline import AdamLRs, DensifyConfig, Pipeline


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    low = c / 12.92
    high = np.power(np.clip((c + 0.055) / 1.055, 1e-7, None), 2.4)
    return np.where(c <= 0.04045, low, high).astype(np.float32)


def _psnr(mse: float) -> float:
    return 99.0 if mse <= 1e-12 else -10.0 * math.log10(mse)


def _save_preview(img_lin: np.ndarray, path: Path) -> None:
    rgb = np.clip(img_lin, 0.0, 1.0)
    rgb_srgb = np.power(rgb, 1.0 / 2.2)
    rgba = np.concatenate([rgb_srgb, np.ones_like(rgb[..., :1])], axis=-1)
    Image.fromarray((rgba * 255.0 + 0.5).astype(np.uint8), mode="RGBA").save(path)


def run(densify_from: int, densify_window: int, out_tag: str):
    W, H = 532, 947
    scene = load_colmap("datasets/scans/cactus", width=W, height=H)
    n_cams = len(scene.images)
    n_init = scene.initial_points["means"].shape[0]
    print(f"\n=== densify_from={densify_from}, window={densify_window} ===")
    print(f"loaded {n_cams} cameras, {n_init} init points")

    device = create_device(backend="vulkan")
    pipeline = Pipeline(
        device=device, width=W, height=H,
        max_splats=500_000, max_sort_pairs=8_000_000,
    )
    pipeline.z_near_cull = 0.2
    pipeline.set_gaussians_from_numpy(**scene.initial_points)
    pipeline.clear_tracking()

    cfg = DensifyConfig()

    targets_srgb = [load_image(e, scene.image_dir, W, H) for e in scene.images]
    targets_linear = [_srgb_to_linear(t) for t in targets_srgb]

    out_dir = Path(f"outputs/train/convergence_diag_v3_{out_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    n_iters = 5000
    densify_every = 100
    densify_to = densify_from + densify_window

    lr_mean_init = 1.6e-4
    lr_mean_final = lr_mean_init * 1e-2
    decay = (lr_mean_final / lr_mean_init) ** (1.0 / n_iters)

    rolling = None
    t_start = time.time()

    print(f"{'iter':>5} {'rolling':>10} {'psnr':>7} {'n_g':>7}")
    for it in range(1, n_iters + 1):
        cam_idx = int(rng.integers(0, n_cams))
        entry = scene.images[cam_idx]
        pipeline.set_target_image(targets_srgb[cam_idx])

        lrs = AdamLRs(mean=lr_mean_init * (decay ** (it - 1)))
        pipeline.train_step(entry.camera, lrs=lrs)

        rendered = pipeline.read_output_image()[..., :3]
        if not np.all(np.isfinite(rendered)):
            print(f"iter {it}: ABORT NaN")
            return
        mse = float(np.mean((rendered - targets_linear[cam_idx]) ** 2))
        rolling = mse if rolling is None else 0.98 * rolling + 0.02 * mse

        if densify_from <= it <= densify_to and (it - densify_from) % densify_every == 0:
            pipeline.densify(cfg, window_iters=densify_every)
            pipeline.clear_tracking()

        if it % 250 == 0 or it == 1:
            n_g = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
            print(f"{it:5d} {rolling:10.5f} {_psnr(rolling):7.2f} {n_g:7d}")

        if it in {500, 1500, 3000, 5000}:
            pipeline.render(scene.images[0].camera)
            _save_preview(
                pipeline.read_output_image()[..., :3],
                out_dir / f"cam00_iter_{it:06d}.png",
            )

    elapsed = time.time() - t_start
    n_g = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    print(f"done in {elapsed:.1f}s  final n_g={n_g}  final rolling={rolling:.5f}  PSNR(rolling)={_psnr(rolling):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", type=str, default="all",
                    choices=["500", "1500", "2500", "all"])
    args = ap.parse_args()

    trials = []
    if args.which in ("500", "all"):
        trials.append((500, 1500, "from500"))
    if args.which in ("1500", "all"):
        trials.append((1500, 1500, "from1500"))
    if args.which in ("2500", "all"):
        trials.append((2500, 1500, "from2500"))

    for df, dw, tag in trials:
        run(df, dw, tag)


if __name__ == "__main__":
    main()
