"""Inspect what decideActions will see at iter 500.

Runs 500 iters of cactus training with NO densify + NO tracking clear
(so the 500-iter accumulator is what decideActions would see). Then:
- Reads tracking buffers and gaussian state.
- Computes the normalized per-iter grad magnitude the shader compares.
- Prints distributions of: grad mag, max_scale, opacity, coverage.
- Predicts SPLIT/CLONE/PRUNE/KEEP counts at various thresholds.
"""
from __future__ import annotations

import numpy as np

from trainer.colmap import load_colmap, load_image
from trainer.device import create_device
from trainer.pipeline import AdamLRs, Pipeline


def main() -> None:
    W, H = 532, 947
    scene = load_colmap("datasets/scans/cactus", width=W, height=H)
    n_cams = len(scene.images)
    n_init = scene.initial_points["means"].shape[0]

    device = create_device(backend="vulkan")
    pipeline = Pipeline(
        device=device, width=W, height=H,
        max_splats=500_000, max_sort_pairs=8_000_000,
    )
    pipeline.z_near_cull = 0.2
    pipeline.set_gaussians_from_numpy(**scene.initial_points)
    pipeline.clear_tracking()

    targets_srgb = [load_image(e, scene.image_dir, W, H) for e in scene.images]

    rng = np.random.default_rng(0)
    n_iters = 500

    lr_mean_init = 1.6e-4
    lr_mean_final = lr_mean_init * 1e-2
    # Schedule as though we were going to train 10k iters.
    decay = (lr_mean_final / lr_mean_init) ** (1.0 / 10000)

    print(f"training {n_iters} iters to populate the accumulators...")
    for it in range(1, n_iters + 1):
        cam_idx = int(rng.integers(0, n_cams))
        entry = scene.images[cam_idx]
        pipeline.set_target_image(targets_srgb[cam_idx])
        lrs = AdamLRs(mean=lr_mean_init * (decay ** (it - 1)))
        pipeline.train_step(entry.camera, lrs=lrs)
        if it % 100 == 0:
            n_g = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
            print(f"  iter {it}  n_g={n_g}")

    # --- Dump gaussian + tracking state ---
    n = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    assert n == n_init, f"expected fixed count ({n_init}), got {n}"

    # Gaussian3D layout: 14 floats per element.
    raw_g = pipeline.buffers.gaussians.to_numpy()
    g = np.frombuffer(raw_g.tobytes(), dtype=np.float32).reshape(-1, 14)[:n]
    log_scales = g[:, 3:6]   # (N, 3)
    opacity_logits = g[:, 10]
    # Opacity is stored as a logit; decide_actions applies sigmoid before
    # comparing to t_opacity. Do the same here for the prediction.
    opacities = 1.0 / (1.0 + np.exp(-opacity_logits))
    max_scale = np.exp(log_scales).max(axis=-1)
    min_scale = np.exp(log_scales).min(axis=-1)

    tracking = pipeline.read_tracking()
    mag_total = tracking["grad_mean_2d_mag"][:n]
    visibility = tracking["visibility_count"].astype(np.int64)[:n]

    # Replicate decideActions' normalization: mag_avg = mag_total / vis.
    mag_per_iter = np.where(
        visibility > 0, mag_total / np.maximum(visibility, 1), 0.0
    )

    # --- Print distributions ---
    def pct(arr, qs=(1, 10, 50, 90, 95, 99, 100)):
        return {q: float(np.percentile(arr, q)) for q in qs}

    print()
    print("=== gaussian state at iter", n_iters, "(n =", n, ") ===")
    print(f"opacity_logit pct: {pct(opacity_logits)}")
    print(f"opacity (prob) pct: {pct(opacities)}")
    print(f"max_scale pct: {pct(max_scale)}")
    print(f"min_scale pct: {pct(min_scale)}")
    print()
    print("=== tracking accumulators ===")
    print(f"coverage (total pixel touches over window)  pct: {pct(coverage)}")
    print(f"coverage == 0: {(coverage == 0).sum()}  ({100*(coverage == 0).mean():.1f}%)")
    print()
    print(f"mag_total    pct: {pct(mag_total)}")
    print(f"mag_per_iter pct: {pct(mag_per_iter)}")
    print()

    # Breakdown: what percentage would be flagged at different thresholds?
    print("=== predicted action counts for various configs ===")
    # Defaults we've been using.
    configs = [
        {"t_grad": 1e-3, "t_split_scale": 0.01, "t_opacity": 0.005, "t_big_scale": 0.1, "name": "cactus-current"},
        {"t_grad": 1e-3, "t_split_scale": 0.01, "t_opacity": 0.005, "t_big_scale": 0.5, "name": "t_big_scale=0.5"},
        {"t_grad": 1e-3, "t_split_scale": 0.05, "t_opacity": 0.005, "t_big_scale": 0.5, "name": "t_split_scale=0.05 + t_big_scale=0.5"},
        {"t_grad": 2e-4, "t_split_scale": 0.01, "t_opacity": 0.005, "t_big_scale": 0.1, "name": "gsplat-default"},
        {"t_grad": 2e-4, "t_split_scale": 0.01, "t_opacity": 0.005, "t_big_scale": 0.5, "name": "gsplat-default + t_big_scale=0.5"},
    ]
    for cfg in configs:
        prune = (opacities < cfg["t_opacity"]) | (max_scale > cfg["t_big_scale"])
        grad_high = mag_per_iter > cfg["t_grad"]
        candidate_grow = grad_high & ~prune
        split = candidate_grow & (max_scale > cfg["t_split_scale"])
        clone = candidate_grow & (max_scale <= cfg["t_split_scale"])
        keep = ~prune & ~candidate_grow
        print(f"  {cfg['name']}:")
        print(f"    PRUNE={int(prune.sum()):5d}  SPLIT={int(split.sum()):5d}  "
              f"CLONE={int(clone.sum()):5d}  KEEP={int(keep.sum()):5d}  "
              f"TOTAL={int(prune.sum() + split.sum() + clone.sum() + keep.sum())}")
        # Why is prune high (if it is)?
        if prune.sum() > 0:
            opacity_prune = (opacities < cfg['t_opacity']).sum()
            scale_prune = (max_scale > cfg['t_big_scale']).sum()
            print(f"      prune breakdown: opacity={int(opacity_prune)}, scale={int(scale_prune)} "
                  f"(overlap allowed)")


if __name__ == "__main__":
    main()
