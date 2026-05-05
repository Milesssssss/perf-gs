"""v2 training entry: Suzanne overfit with GPU-driven densification.

Adds to v1:
- Periodic densification (split / clone / prune) on the GPU.
- Periodic opacity reset.
- Per-iter tracking accumulators cleared at the start of each window.

Usage:
    .venv/Scripts/python.exe scripts/v2_train.py [--iters 10000]
                                                 [--scene datasets/synthetic/suzanne_blender]
                                                 [--width 960] [--height 540]
                                                 [--run-name my_v2_run]

Output layout is identical to v1_train.py's (per-run folder with
cams/cam_NN/iter_*.png + gt.png, final_psnr.json, log.txt, log.jsonl,
config.json) plus a new ``densify_log.jsonl`` listing per-densify-step
action counts and resulting gaussian counts.
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

from trainer.colmap import load_colmap, load_image
from trainer.device import create_device
from trainer.export import render_to_png, save_ply
from trainer.pipeline import AdamLRs, DensifyConfig, Pipeline


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    low = c / 12.92
    high = np.power(np.clip((c + 0.055) / 1.055, 1e-7, None), 2.4)
    return np.where(c <= 0.04045, low, high).astype(np.float32)


def _psnr_linear(linear_rendered: np.ndarray, linear_target: np.ndarray) -> float:
    mse = float(np.mean((linear_rendered - linear_target) ** 2))
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


def _save_srgb_png(rgb_linear: np.ndarray, path: Path) -> None:
    rgb = np.clip(rgb_linear, 0.0, 1.0)
    rgb_srgb = np.power(rgb, 1.0 / 2.2)
    rgba = np.concatenate([rgb_srgb, np.ones_like(rgb[..., :1])], axis=-1)
    png = (rgba * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(png, mode="RGBA").save(path)


def _render_all_cameras(pipeline: Pipeline, scene, iter_idx: int, cams_dir: Path) -> None:
    for cam_i, entry in enumerate(scene.images):
        pipeline.render(entry.camera)
        img = pipeline.read_output_image()[..., :3]
        out_path = cams_dir / f"cam_{cam_i:02d}" / f"iter_{iter_idx:06d}.png"
        _save_srgb_png(img, out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="datasets/synthetic/suzanne_blender")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--preview-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--densify-every", type=int, default=100)
    ap.add_argument("--densify-from", type=int, default=3000)
    ap.add_argument("--densify-to", type=int, default=10000)
    ap.add_argument("--opacity-reset-iters", type=int, nargs="*", default=[3500, 7000])
    ap.add_argument("--n-scoring-views", type=int, default=10)
    ap.add_argument("--loss-thresh", type=float, default=0.1)
    ap.add_argument("--importance-thresh", type=float, default=5.0)
    ap.add_argument("--t-split-scale", type=float, default=0.01)
    ap.add_argument("--t-opacity", type=float, default=0.005)
    ap.add_argument("--t-big-scale", type=float, default=0.1)
    ap.add_argument("--growth-rate", type=float, default=0.3)
    ap.add_argument("--opacity-cap", type=float, default=0.8)
    ap.add_argument("--max-splats", type=int, default=500_000)
    ap.add_argument("--z-near-cull", type=float, default=0.01,
                    help="view-space depth floor for bin cull; raise for scenes "
                         "where points can land near a camera pose (e.g. 0.2)")
    ap.add_argument("--backend", type=str, default="vulkan", choices=["vulkan", "d3d12"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs/train") / run_name
    cams_dir = run_dir / "cams"
    run_dir.mkdir(parents=True, exist_ok=True)
    cams_dir.mkdir(parents=True, exist_ok=True)

    scene = load_colmap(args.scene, width=args.width, height=args.height)
    n_cams = len(scene.images)
    n_init = scene.initial_points["means"].shape[0]
    extent = scene.scene_extent
    print(f"loaded {n_cams} cameras, {n_init} init points, scene_extent={extent:.3f}")

    cfg = DensifyConfig(
        n_scoring_views=args.n_scoring_views,
        loss_thresh=args.loss_thresh,
        importance_thresh=args.importance_thresh,
        t_split_scale=args.t_split_scale,
        t_opacity=args.t_opacity,
        t_big_scale=args.t_big_scale * extent,
        growth_rate=args.growth_rate,
        opacity_cap=args.opacity_cap,
    )

    (run_dir / "config.json").write_text(json.dumps({
        "scene": args.scene,
        "iters": args.iters,
        "width": args.width, "height": args.height,
        "preview_every": args.preview_every, "log_every": args.log_every,
        "densify_every": args.densify_every,
        "densify_from": args.densify_from, "densify_to": args.densify_to,
        "opacity_reset_iters": args.opacity_reset_iters,
        "densify_config": {
            "n_scoring_views": cfg.n_scoring_views,
            "loss_thresh": cfg.loss_thresh,
            "importance_thresh": cfg.importance_thresh,
            "t_split_scale": cfg.t_split_scale,
            "t_opacity": cfg.t_opacity,
            "t_big_scale": cfg.t_big_scale,
            "growth_rate": cfg.growth_rate,
            "opacity_cap": cfg.opacity_cap,
        },
        "max_splats": args.max_splats,
        "z_near_cull": args.z_near_cull,
        "backend": args.backend, "seed": args.seed,
        "run_name": run_name, "n_cams": n_cams, "n_init_points": n_init,
    }, indent=2))

    targets_srgb = [load_image(e, scene.image_dir, args.width, args.height)
                    for e in scene.images]
    targets_linear = [_srgb_to_linear(t) for t in targets_srgb]
    for cam_i, (entry, tgt) in enumerate(zip(scene.images, targets_srgb)):
        cam_dir = cams_dir / f"cam_{cam_i:02d}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((tgt * 255 + 0.5).clip(0, 255).astype(np.uint8),
                        mode="RGB").save(cam_dir / "gt.png")

    device = create_device(backend=args.backend)
    pipeline = Pipeline(
        device=device, width=args.width, height=args.height,
        max_splats=args.max_splats,
    )
    pipeline.z_near_cull = args.z_near_cull
    pipeline.set_gaussians_from_numpy(**scene.initial_points)
    pipeline.clear_tracking()

    # Pre-upload every target image as its own GPU texture. Per-iter we
    # just swap the active reference — no CPU→GPU upload in the hot loop.
    target_textures = [pipeline.create_target_texture(t) for t in targets_srgb]

    _render_all_cameras(pipeline, scene, 0, cams_dir)

    log_txt = (run_dir / "log.txt").open("w", buffering=1)
    log_json = (run_dir / "log.jsonl").open("w", buffering=1)
    densify_json = (run_dir / "densify_log.jsonl").open("w", buffering=1)
    log_txt.write(f"# v2 training log — scene={args.scene}, backend={args.backend}\n")
    log_txt.write(f"# iters={args.iters}, resolution={args.width}x{args.height}, n_cams={n_cams}\n")
    log_txt.write(f"# densify: every {args.densify_every} iters in "
                  f"[{args.densify_from}, {args.densify_to}], opacity_reset@{args.opacity_reset_iters}\n")

    rng = np.random.default_rng(args.seed)

    lr_mean_init = 1.6e-4
    lr_mean_final = lr_mean_init * 1e-2
    decay = (lr_mean_final / lr_mean_init) ** (1.0 / max(args.iters, 1))

    t_start = time.time()
    rolling_loss = None
    # Track the current live gaussian count Python-side to avoid a
    # per-iter counter readback. Updated from densify() return values.
    cur_n_gaussians = n_init

    for it in range(1, args.iters + 1):
        cam_idx = int(rng.integers(0, n_cams))
        entry = scene.images[cam_idx]
        pipeline.set_target_texture(target_textures[cam_idx])

        lrs = AdamLRs(mean=lr_mean_init * (decay ** (it - 1)))
        pipeline.train_step(entry.camera, lrs=lrs)

        # Densification schedule — still in float32 `it`, cheap predicate.
        do_densify = (args.densify_from <= it <= args.densify_to
                      and (it - args.densify_from) % args.densify_every == 0)
        do_opacity_reset = it in args.opacity_reset_iters
        do_log = (it % args.log_every == 0) or (it == 1) or (it == args.iters)
        do_preview = (it % args.preview_every == 0) or (it == args.iters)

        if do_opacity_reset:
            pipeline.reset_opacity(cfg)
            densify_json.write(json.dumps({"iter": it, "event": "opacity_reset"}) + "\n")
            print(f"iter {it}: opacity reset")

        if do_densify:
            # Suppress opacity pruning before the first opacity reset.
            densify_cfg = cfg
            if args.opacity_reset_iters and it < args.opacity_reset_iters[0]:
                from dataclasses import replace
                densify_cfg = replace(cfg, t_opacity=0.0)

            # Sample N cameras for importance scoring.
            n_score = min(densify_cfg.n_scoring_views, n_cams)
            score_indices = rng.choice(n_cams, size=n_score, replace=False)
            score_cameras = [scene.images[i].camera for i in score_indices]
            score_targets = [target_textures[i] for i in score_indices]

            pipeline.compute_importance_scores(score_cameras, score_targets, densify_cfg)
            stats = pipeline.densify(densify_cfg)
            cur_n_gaussians = stats["final_count"]
            densify_json.write(json.dumps({"iter": it, "event": "densify", **stats}) + "\n")
            print(f"iter {it}: densify  "
                  f"keep={stats['n_keep']} split={stats['n_split']} "
                  f"clone={stats['n_clone']} prune={stats['n_prune']} "
                  f"| pre={stats['pre_count']} -> post={stats['final_count']}")

        # Loss/PSNR require a GPU→CPU readback of the output image.
        # Sampling this every step would serialize the training loop and
        # tank GPU utilization; do it only on log boundaries. Same for
        # the NaN/Inf safety check (PLAN.md Q17): if the forward has
        # gone non-finite, it's not going to recover in a few hundred
        # iters, so checking at log cadence is plenty timely.
        if do_log or do_preview:
            pipeline.device.wait_for_idle()
            rendered = pipeline.read_output_image()[..., :3]
            if not np.all(np.isfinite(rendered)):
                n_nan = int(np.sum(~np.isfinite(rendered)))
                msg = (f"iter {it}: rendered image has {n_nan} non-finite pixels "
                       f"(out of {rendered.size}). Training aborted.")
                print(msg)
                log_txt.write(f"# ABORT: {msg}\n")
                break
            mse = float(np.mean((rendered - targets_linear[cam_idx]) ** 2))
            rolling_loss = mse if rolling_loss is None else 0.98 * rolling_loss + 0.02 * mse
            psnr = _psnr_linear(rendered, targets_linear[cam_idx])

            elapsed = time.time() - t_start
            it_per_s = it / elapsed
            line = (f"iter {it:6d}  cam {cam_idx:2d}  "
                    f"loss={rolling_loss:.5f}  psnr={psnr:5.2f} dB  "
                    f"lr_mean={lrs.mean:.2e}  n_g={cur_n_gaussians:6d}  {it_per_s:5.1f} it/s")
            print(line)
            log_txt.write(line + "\n")
            log_json.write(json.dumps({
                "iter": it, "cam_idx": cam_idx,
                "loss_rolling": rolling_loss, "loss_instant": mse,
                "psnr_instant": psnr, "lr_mean": lrs.mean,
                "n_gaussians": cur_n_gaussians, "it_per_s": it_per_s,
                "elapsed_s": elapsed,
            }) + "\n")

        if do_preview:
            _render_all_cameras(pipeline, scene, it, cams_dir)

    log_txt.close()
    log_json.close()
    densify_json.close()

    # Post-training per-camera metrics. Include BOTH the full-frame PSNR
    # (linear and sRGB) and the foreground-masked sRGB PSNR — the latter
    # is the honest silhouette metric (avoids the black-background inflation
    # that misled us on the v1 10k run).
    print("computing final per-camera metrics...")
    per_cam: list[dict] = []
    total_mse_lin = 0.0
    total_mse_srgb = 0.0
    total_mse_srgb_mask = 0.0
    n_mask_used = 0
    for cam_i, entry in enumerate(scene.images):
        pipeline.render(entry.camera)
        r_lin = pipeline.read_output_image()[..., :3]
        t_lin = targets_linear[cam_i]
        mse_lin = float(np.mean((r_lin - t_lin) ** 2))

        r_srgb = np.clip(r_lin, 0.0, 1.0) ** (1.0 / 2.2)
        t_srgb = targets_srgb[cam_i]
        mse_srgb = float(np.mean((r_srgb - t_srgb) ** 2))

        mask = (t_srgb.sum(-1) > 0.05) | (r_srgb.sum(-1) > 0.05)
        if mask.sum() > 0:
            mse_srgb_mask = float(np.mean((r_srgb[mask] - t_srgb[mask]) ** 2))
            total_mse_srgb_mask += mse_srgb_mask
            n_mask_used += 1
        else:
            mse_srgb_mask = 0.0

        total_mse_lin += mse_lin
        total_mse_srgb += mse_srgb
        per_cam.append({
            "cam_idx": cam_i, "image_name": entry.image_name,
            "mse_linear": mse_lin,
            "psnr_linear": _psnr_linear(r_lin, t_lin),
            "mse_srgb": mse_srgb,
            "psnr_srgb": -10.0 * math.log10(max(mse_srgb, 1e-12)),
            "mse_srgb_masked": mse_srgb_mask,
            "psnr_srgb_masked": -10.0 * math.log10(max(mse_srgb_mask, 1e-12)),
        })

    def psnr(m): return -10.0 * math.log10(max(m, 1e-12))
    summary = {
        "avg_mse_linear": total_mse_lin / n_cams,
        "avg_psnr_linear": psnr(total_mse_lin / n_cams),
        "avg_mse_srgb": total_mse_srgb / n_cams,
        "avg_psnr_srgb": psnr(total_mse_srgb / n_cams),
        "avg_mse_srgb_masked": total_mse_srgb_mask / max(n_mask_used, 1),
        "avg_psnr_srgb_masked": psnr(total_mse_srgb_mask / max(n_mask_used, 1)),
        "per_cam": per_cam,
        "n_cams": n_cams,
        "iters": args.iters,
        "final_n_gaussians": int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0]),
    }
    (run_dir / "final_psnr.json").write_text(json.dumps(summary, indent=2))

    ply_path = run_dir / "output.ply"
    save_ply(pipeline, ply_path)

    print(f"done in {time.time() - t_start:.1f}s")
    print(f"avg PSNR  linear full:   {summary['avg_psnr_linear']:6.2f} dB")
    print(f"avg PSNR  sRGB full:     {summary['avg_psnr_srgb']:6.2f} dB")
    print(f"avg PSNR  sRGB masked:   {summary['avg_psnr_srgb_masked']:6.2f} dB  <- honest silhouette")
    print(f"final gaussian count:    {summary['final_n_gaussians']}")
    print(f"saved ply:               {ply_path}")
    print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
