"""Time-series densification diagnostic.

Trains cactus for a short run and, at every densify boundary (and a few
plain iters), dumps:

- Histogram of |grad_param| per param group (post-Adam pre-clip).
- Histogram of mag_per_touch (the exact signal decide_actions sees).
- Histogram of visibility_count per gaussian.
- Histogram of max_scale per gaussian (both world-space).
- Histogram of sigmoid(opacity) per gaussian.
- Action counts for the config we'd apply (SPLIT/CLONE/PRUNE/KEEP).
- Prune breakdown: opacity vs scale.
- Spatial location of new children vs mean of scene: are splits
  concentrating in one region?

Goal: compare these to what we'd expect if densification were behaving
like gsplat's.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from trainer.colmap import load_colmap, load_image
from trainer.device import create_device
from trainer.pipeline import AdamLRs, DensifyConfig, Pipeline


def _pct(arr: np.ndarray, qs=(1, 10, 50, 90, 95, 99)) -> dict:
    if arr.size == 0:
        return {q: float("nan") for q in qs}
    return {q: float(np.percentile(arr, q)) for q in qs}


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    low = c / 12.92
    high = np.power(np.clip((c + 0.055) / 1.055, 1e-7, None), 2.4)
    return np.where(c <= 0.04045, low, high).astype(np.float32)


def _save_preview(img_lin: np.ndarray, path: Path) -> None:
    rgb = np.clip(img_lin, 0.0, 1.0)
    rgb_srgb = np.power(rgb, 1.0 / 2.2)
    rgba = np.concatenate([rgb_srgb, np.ones_like(rgb[..., :1])], axis=-1)
    Image.fromarray((rgba * 255.0 + 0.5).astype(np.uint8), mode="RGBA").save(path)


def _read_state(pipeline: Pipeline) -> dict:
    """Read gaussian params + tracking at current state."""
    n = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    raw = pipeline.buffers.gaussians.to_numpy()
    g = np.frombuffer(raw.tobytes(), dtype=np.float32).reshape(-1, 14)[:n]
    mean = g[:, 0:3]
    log_scale = g[:, 3:6]
    opacity_logit = g[:, 10]
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))
    max_scale = np.exp(log_scale).max(axis=-1)
    min_scale = np.exp(log_scale).min(axis=-1)

    tracking = pipeline.read_tracking()
    mag_total = tracking["grad_mean_2d_mag"][:n]
    visibility = tracking["visibility_count"].astype(np.int64)[:n]
    mag_avg = np.where(visibility > 0, mag_total / np.maximum(visibility, 1), 0.0)

    grads = pipeline.read_gradients()

    return {
        "n": n,
        "mean": mean,
        "opacity_logit": opacity_logit,
        "opacity": opacity,
        "max_scale": max_scale,
        "min_scale": min_scale,
        "log_scale": log_scale,
        "mag_total": mag_total,
        "visibility": visibility,
        "mag_avg": mag_avg,
        "grads": grads,
    }


def _predict_actions(state: dict, cfg: DensifyConfig) -> dict:
    opacity = state["opacity"]
    max_scale = state["max_scale"]
    mag_avg = state["mag_avg"]

    prune_op = opacity < cfg.t_opacity
    prune_scale = max_scale > cfg.t_big_scale
    prune = prune_op | prune_scale
    grad_high = mag_avg > cfg.t_grad
    grow = grad_high & ~prune
    split = grow & (max_scale > cfg.t_split_scale)
    clone = grow & (max_scale <= cfg.t_split_scale)
    keep = ~prune & ~grow

    return {
        "PRUNE": int(prune.sum()),
        "PRUNE_op": int(prune_op.sum()),
        "PRUNE_scale": int(prune_scale.sum()),
        "SPLIT": int(split.sum()),
        "CLONE": int(clone.sum()),
        "KEEP": int(keep.sum()),
    }


def _dump(pipeline: Pipeline, iter_idx: int, cfg: DensifyConfig, out: list) -> None:
    state = _read_state(pipeline)
    actions = _predict_actions(state, cfg)

    # Spatial: bbox of means + centroid.
    mn = state["mean"].min(axis=0).tolist()
    mx = state["mean"].max(axis=0).tolist()
    centroid = state["mean"].mean(axis=0).tolist()

    # Grad magnitudes per param (summarized).
    grads = state["grads"]
    grad_pcts = {
        name: _pct(np.abs(arr))
        for name, arr in grads.items()
    }

    entry = {
        "iter": iter_idx,
        "n_gaussians": state["n"],
        "centroid": centroid,
        "bbox_min": mn,
        "bbox_max": mx,
        "opacity_pct": _pct(state["opacity"]),
        "opacity_logit_pct": _pct(state["opacity_logit"]),
        "max_scale_pct": _pct(state["max_scale"]),
        "min_scale_pct": _pct(state["min_scale"]),
        "visibility_pct": _pct(state["visibility"].astype(np.float64)),
        "visibility_zero_frac": float((state["visibility"] == 0).mean()),
        "mag_avg_pct": _pct(state["mag_avg"]),
        "predicted_actions_at_cfg": actions,
        "grad_mag_pct": grad_pcts,
    }
    out.append(entry)

    # Pretty-print a short version.
    print(f"\n[iter {iter_idx}] n_g={state['n']}")
    print(f"  centroid={np.array(centroid).round(3).tolist()}")
    print(f"  bbox=[{np.array(mn).round(2).tolist()}, {np.array(mx).round(2).tolist()}]")
    print(f"  opacity  p50={entry['opacity_pct'][50]:.3f}  p90={entry['opacity_pct'][90]:.3f}  p99={entry['opacity_pct'][99]:.3f}")
    print(f"  max_scale p50={entry['max_scale_pct'][50]:.4f}  p90={entry['max_scale_pct'][90]:.4f}  p99={entry['max_scale_pct'][99]:.4f}")
    print(f"  visibility p50={entry['visibility_pct'][50]:.0f}  p90={entry['visibility_pct'][90]:.0f}  p99={entry['visibility_pct'][99]:.0f}  zero_frac={entry['visibility_zero_frac']:.3f}")
    print(f"  mag_avg p50={entry['mag_avg_pct'][50]:.2e}  p90={entry['mag_avg_pct'][90]:.2e}  p99={entry['mag_avg_pct'][99]:.2e}")
    print(f"  mean-grad  p50={entry['grad_mag_pct']['mean_x'][50]:.2e}  "
          f"log_scale-grad p50={entry['grad_mag_pct']['log_scale_x'][50]:.2e}  "
          f"opacity-grad p50={entry['grad_mag_pct']['opacity'][50]:.2e}  "
          f"color-grad p50={entry['grad_mag_pct']['color_r'][50]:.2e}")
    print(f"  actions-if-decided-now: {actions}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--densify-every", type=int, default=100)
    ap.add_argument("--densify-from", type=int, default=500)
    ap.add_argument("--t-grad", type=float, default=2e-4)
    ap.add_argument("--with-densify", action="store_true",
                    help="actually apply densify (otherwise just read what WOULD happen)")
    ap.add_argument("--run-name", type=str, default="densify_ts")
    args = ap.parse_args()

    W, H = 532, 947
    scene = load_colmap("datasets/scans/cactus", width=W, height=H)
    n_cams = len(scene.images)
    n_init = scene.initial_points["means"].shape[0]
    print(f"loaded {n_cams} cameras, {n_init} init points")

    device = create_device(backend="vulkan")
    pipeline = Pipeline(
        device=device, width=W, height=H,
        max_splats=500_000, max_sort_pairs=8_000_000,
    )
    pipeline.z_near_cull = 0.2
    pipeline.set_gaussians_from_numpy(**scene.initial_points)
    pipeline.clear_tracking()

    cfg = DensifyConfig(t_grad=args.t_grad)

    targets_srgb = [load_image(e, scene.image_dir, W, H) for e in scene.images]

    out_dir = Path("outputs/train") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    lr_mean_init = 1.6e-4
    lr_mean_final = lr_mean_init * 1e-2
    decay = (lr_mean_final / lr_mean_init) ** (1.0 / 10000)

    # Dump initial state.
    dumps: list[dict] = []
    _dump(pipeline, 0, cfg, dumps)

    t_start = time.time()
    for it in range(1, args.iters + 1):
        cam_idx = int(rng.integers(0, n_cams))
        entry = scene.images[cam_idx]
        pipeline.set_target_image(targets_srgb[cam_idx])
        lrs = AdamLRs(mean=lr_mean_init * (decay ** (it - 1)))
        pipeline.train_step(entry.camera, lrs=lrs)

        # At each densify boundary, dump state *before* the densify pass.
        if args.densify_from <= it and (it - args.densify_from) % args.densify_every == 0:
            _dump(pipeline, it, cfg, dumps)
            if args.with_densify:
                stats = pipeline.densify(cfg)
                pipeline.clear_tracking()
                print(f"  DENSIFIED: split={stats['n_split']} clone={stats['n_clone']} "
                      f"prune={stats['n_prune']} keep={stats['n_keep']} -> {stats['final_count']}")
                # Save preview of cam 0 after densify so we can see the artifact.
                pipeline.render(scene.images[0].camera)
                _save_preview(
                    pipeline.read_output_image()[..., :3],
                    out_dir / f"cam00_iter_{it:06d}.png",
                )

    # Final dump.
    _dump(pipeline, args.iters, cfg, dumps)

    # Render final cam 0.
    pipeline.render(scene.images[0].camera)
    _save_preview(
        pipeline.read_output_image()[..., :3],
        out_dir / f"cam00_iter_{args.iters:06d}.png",
    )

    (out_dir / "timeseries.json").write_text(json.dumps(dumps, indent=2, default=float))
    print(f"\ndone in {time.time()-t_start:.1f}s")
    print(f"dumped timeseries to {out_dir / 'timeseries.json'}")


if __name__ == "__main__":
    main()
