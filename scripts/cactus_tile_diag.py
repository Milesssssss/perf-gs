"""Diagnose tile occupancy under the v2.x uncapped-binning pipeline.

Reports per-tile gaussian counts from tile_start / tile_end after a few
thousand training iterations. There is no cap anymore — any tile count
is legal — but the *distribution* still tells us how badly the scene
concentrates gaussians.
"""
from __future__ import annotations

import numpy as np

from trainer.colmap import load_colmap, load_image
from trainer.device import create_device
from trainer.pipeline import AdamLRs, DensifyConfig, Pipeline


def main() -> None:
    W, H = 532, 947
    scene = load_colmap("datasets/scans/cactus", width=W, height=H)

    device = create_device(backend="vulkan")
    pipeline = Pipeline(
        device=device, width=W, height=H, max_splats=500_000,
        max_sort_pairs=8_000_000,
    )
    pipeline.z_near_cull = 0.2
    pipeline.set_gaussians_from_numpy(**scene.initial_points)
    pipeline.clear_tracking()

    cfg = DensifyConfig()
    rng = np.random.default_rng(0)

    lr_mean_init = 1.6e-4
    lr_mean_final = lr_mean_init * 1e-2
    total_iters = 10000
    decay = (lr_mean_final / lr_mean_init) ** (1.0 / total_iters)

    target_iter = 3500
    densify_every = 100
    densify_from = 500
    densify_to = 7000
    opacity_reset_iters = {2000}
    n_cams = len(scene.images)

    print(f"training to iter {target_iter} ...")
    for it in range(1, target_iter + 1):
        cam_idx = int(rng.integers(0, n_cams))
        entry = scene.images[cam_idx]
        pipeline.set_target_image(load_image(entry, scene.image_dir, W, H))
        lrs = AdamLRs(mean=lr_mean_init * (decay ** (it - 1)))
        pipeline.train_step(entry.camera, lrs=lrs)
        #if it in opacity_reset_iters:
        #    pipeline.reset_opacity(cfg)
        #if densify_from <= it <= densify_to and (it - densify_from) % densify_every == 0:
        #    pipeline.densify(cfg, window_iters=densify_every)
        #    pipeline.clear_tracking()
        if it % 500 == 0:
            n_g = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
            print(f"  iter {it}  n_g={n_g}")

    num_tiles_x = pipeline.buffers.num_tiles_x
    num_tiles_y = pipeline.buffers.num_tiles_y
    total_tiles = num_tiles_x * num_tiles_y
    print(f"\ntile grid: {num_tiles_x} x {num_tiles_y} = {total_tiles} tiles\n")

    sampled = [0, 20, 40, 60, 73, 80, 100, 120]
    for cam_idx in sampled:
        entry = scene.images[cam_idx]
        pipeline.render(entry.camera)
        ts_raw = pipeline.buffers.tile_start.to_numpy()
        te_raw = pipeline.buffers.tile_end.to_numpy()
        ts = np.frombuffer(ts_raw.tobytes(), dtype=np.uint32)[:total_tiles]
        te = np.frombuffer(te_raw.tobytes(), dtype=np.uint32)[:total_tiles]
        tc = (te.astype(np.int64) - ts.astype(np.int64)).clip(min=0)
        total_pairs = int(tc.sum())

        img = pipeline.read_output_image()[..., :3]
        target = load_image(entry, scene.image_dir, W, H)
        def s2l(c):
            low = c / 12.92
            high = np.power(np.clip((c + 0.055) / 1.055, 1e-7, None), 2.4)
            return np.where(c <= 0.04045, low, high).astype(np.float32)
        tgt_lin = s2l(target)

        p99 = int(np.percentile(tc, 99))
        p90 = int(np.percentile(tc, 90))
        print(f"cam {cam_idx:3d}:")
        print(f"  per-tile count  min={tc.min()} med={int(np.median(tc))} p90={p90} p99={p99} max={tc.max()}")
        print(f"  total tile entries: {total_pairs}")
        print(f"  render brightness: {img.mean(axis=(0,1)).round(3)}  target: {tgt_lin.mean(axis=(0,1)).round(3)}")


if __name__ == "__main__":
    main()
