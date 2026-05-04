"""Diagnose whether tile overflow is degrading cactus training.

Strategy: train for N iters with current defaults, then for a sample of
cameras dispatch just forward (bin) and read back screen_tiles_count.
Report: max count, count of saturated tiles (>=1024), histogram.
Also: foreground render brightness to see if we reproduce the 'too
white' artifact.
"""
from __future__ import annotations

import numpy as np

from trainer.buffers import MAX_GAUSSIANS_PER_TILE
from trainer.colmap import load_colmap, load_image
from trainer.device import create_device
from trainer.pipeline import AdamLRs, DensifyConfig, Pipeline


def main() -> None:
    W, H = 532, 947
    scene = load_colmap("datasets/scans/cactus", width=W, height=H)

    device = create_device(backend="vulkan")
    pipeline = Pipeline(device=device, width=W, height=H, max_splats=500_000)
    pipeline.z_near_cull = 0.2
    pipeline.set_gaussians_from_numpy(**scene.initial_points)
    pipeline.clear_tracking()

    cfg = DensifyConfig()
    rng = np.random.default_rng(0)

    # Match the running config.
    lr_mean_init = 1.6e-4
    lr_mean_final = lr_mean_init * 1e-2
    total_iters = 10000
    decay = (lr_mean_final / lr_mean_init) ** (1.0 / total_iters)

    target_iter = 3500
    densify_every = 100
    densify_from = 500
    densify_to = 7000
    opacity_reset_iters = {2000}  # only first one, since we stop at 3500
    n_cams = len(scene.images)

    print(f"training to iter {target_iter} ...")
    for it in range(1, target_iter + 1):
        cam_idx = int(rng.integers(0, n_cams))
        entry = scene.images[cam_idx]
        pipeline.set_target_image(load_image(entry, scene.image_dir, W, H))
        lrs = AdamLRs(mean=lr_mean_init * (decay ** (it - 1)))
        pipeline.train_step(entry.camera, lrs=lrs)
        if it in opacity_reset_iters:
            pipeline.reset_opacity(cfg)
        if densify_from <= it <= densify_to and (it - densify_from) % densify_every == 0:
            pipeline.densify(cfg, window_iters=densify_every)
            pipeline.clear_tracking()
        if it % 500 == 0:
            n_g = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
            print(f"  iter {it}  n_g={n_g}")

    # Now: for a sample of cameras, dispatch just the forward chain and
    # read tile counts.
    import slangpy as spy
    num_tiles_x = pipeline.buffers.num_tiles_x
    num_tiles_y = pipeline.buffers.num_tiles_y
    total_tiles = num_tiles_x * num_tiles_y
    print(f"\ntile grid: {num_tiles_x} x {num_tiles_y} = {total_tiles} tiles")
    print(f"MAX_GAUSSIANS_PER_TILE = {MAX_GAUSSIANS_PER_TILE}\n")

    sampled = [0, 20, 40, 60, 73, 80, 100, 120]
    for cam_idx in sampled:
        entry = scene.images[cam_idx]
        pipeline.render(entry.camera)  # uses the full forward chain
        tc_raw = pipeline.buffers.screen_tiles_count.to_numpy()
        tc = np.frombuffer(tc_raw.tobytes(), dtype=np.uint32)[:total_tiles]
        # Note: sort kernel CLAMPS the stored value to MAX_GAUSSIANS_PER_TILE
        # (see sort.slang). But bin.slang does InterlockedAdd WITHOUT clamping
        # the counter; it only checks bounds for the slot write. So we
        # read the POST-sort value here (== min(attempted_count, 1024)).
        saturated = int((tc >= MAX_GAUSSIANS_PER_TILE).sum())
        # Also render image brightness
        img = pipeline.read_output_image()[..., :3]
        target = load_image(entry, scene.image_dir, W, H)
        # Linear target
        def s2l(c):
            low = c / 12.92
            high = np.power(np.clip((c + 0.055) / 1.055, 1e-7, None), 2.4)
            return np.where(c <= 0.04045, low, high).astype(np.float32)
        tgt_lin = s2l(target)
        print(f"cam {cam_idx:3d}:")
        print(f"  tile count stats: min={tc.min()} med={int(np.median(tc))} max={tc.max()}")
        print(f"  saturated tiles (>=MAX): {saturated} / {total_tiles}  ({100*saturated/total_tiles:.1f}%)")
        print(f"  render brightness mean: {img.mean(axis=(0,1)).round(3)}  target: {tgt_lin.mean(axis=(0,1)).round(3)}")
        # Also dump how many gaussian entries total were pushed
        print(f"  tile entries total: {tc.sum()}  (if we had no cap, could be higher)")


if __name__ == "__main__":
    main()
