"""Fixed-count convergence diagnostic.

Trains cactus with densification + opacity reset DISABLED for 3000 iters
on the initial point cloud. Reports loss every 50 iters so we can see
the convergence curve clean. Also saves preview images at a few
checkpoints.

If loss drops steadily and the preview gradually sharpens: Adam / forward
/ backward are fine, and densification (or something about its
interaction with the pipeline) is what breaks cactus.

If loss plateaus early or bounces: the bug is in the core training math
independent of densification.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from trainer.colmap import load_colmap, load_image
from trainer.device import create_device
from trainer.pipeline import AdamLRs, Pipeline


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


def main() -> None:
    W, H = 532, 947
    scene = load_colmap("datasets/scans/cactus", width=W, height=H)
    n_cams = len(scene.images)
    n_init = scene.initial_points["means"].shape[0]
    print(f"loaded {n_cams} cameras, {n_init} init points")

    device = create_device(backend="vulkan")
    pipeline = Pipeline(
        device=device, width=W, height=H,
        max_splats=max(n_init * 2, 32_000),  # tight cap: no densification, small budget
        max_sort_pairs=2_000_000,
    )
    pipeline.z_near_cull = 0.2
    pipeline.set_gaussians_from_numpy(**scene.initial_points)
    pipeline.clear_tracking()

    # Pre-cache targets.
    targets_srgb = [load_image(e, scene.image_dir, W, H) for e in scene.images]
    targets_linear = [_srgb_to_linear(t) for t in targets_srgb]

    out_dir = Path("outputs/train/convergence_diag")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    n_iters = 3000
    preview_iters = {100, 500, 1000, 2000, 3000}

    lr_mean_init = 1.6e-4
    lr_mean_final = lr_mean_init * 1e-2
    decay = (lr_mean_final / lr_mean_init) ** (1.0 / n_iters)

    rolling = None
    t_start = time.time()

    # Save a baseline render of cam 0 at iter 0.
    pipeline.render(scene.images[0].camera)
    _save_preview(pipeline.read_output_image()[..., :3], out_dir / "cam00_iter_000000.png")

    print()
    print(f"{'iter':>5} {'loss':>10} {'rolling':>10} {'psnr_lin':>9} {'mean_C':>7} {'std_C':>7}")

    for it in range(1, n_iters + 1):
        cam_idx = int(rng.integers(0, n_cams))
        entry = scene.images[cam_idx]
        pipeline.set_target_image(targets_srgb[cam_idx])

        lrs = AdamLRs(mean=lr_mean_init * (decay ** (it - 1)))
        pipeline.train_step(entry.camera, lrs=lrs)

        rendered = pipeline.read_output_image()[..., :3]
        if not np.all(np.isfinite(rendered)):
            n_nan = int(np.sum(~np.isfinite(rendered)))
            print(f"iter {it}: {n_nan} NaN pixels. ABORT.")
            break
        mse = float(np.mean((rendered - targets_linear[cam_idx]) ** 2))
        rolling = mse if rolling is None else 0.98 * rolling + 0.02 * mse

        if it % 50 == 0 or it == 1:
            print(f"{it:5d} {mse:10.5f} {rolling:10.5f} {_psnr(mse):9.2f} "
                  f"{rendered.mean():7.3f} {rendered.std():7.3f}")

        if it in preview_iters:
            pipeline.render(scene.images[0].camera)
            _save_preview(
                pipeline.read_output_image()[..., :3],
                out_dir / f"cam00_iter_{it:06d}.png",
            )

    elapsed = time.time() - t_start
    print(f"\ndone in {elapsed:.1f}s")

    # Final per-camera sweep on a fixed subset.
    print("\nper-camera instant loss (subset):")
    for cam_idx in [0, 20, 73, 100, 130]:
        entry = scene.images[cam_idx]
        pipeline.render(entry.camera)
        rendered = pipeline.read_output_image()[..., :3]
        mse = float(np.mean((rendered - targets_linear[cam_idx]) ** 2))
        print(f"  cam {cam_idx:3d}: mse={mse:.5f}  psnr={_psnr(mse):.2f} dB")


if __name__ == "__main__":
    main()
