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
from trainer.viewer import Viewer
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
    ap.add_argument("--scene", type=str, default="datasets/360_v2/bicycle")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--downsample", type=int, default=1,
                    help="downsample factor (1=full res, 2=half, 4=quarter). "
                         "Uses images_N dir if it exists, otherwise resizes on load.")
    ap.add_argument("--width", type=int, default=None,
                    help="force render width (overrides --downsample)")
    ap.add_argument("--height", type=int, default=None,
                    help="force render height (overrides --downsample)")
    ap.add_argument("--preview-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--densify-every", type=int, default=100)
    ap.add_argument("--densify-from", type=int, default=500)
    ap.add_argument("--densify-to", type=int, default=15000)
    #ap.add_argument("--opacity-reset-iters", type=int, nargs="*", default=[3000])
    ap.add_argument("--n-scoring-views", type=int, default=10)
    ap.add_argument("--loss-thresh", type=float, default=0.1)
    ap.add_argument("--importance-thresh", type=float, default=5.0)
    ap.add_argument("--prune-thresh", type=float, default=50.0)
    ap.add_argument("--t-split-scale", type=float, default=0.001)
    ap.add_argument("--t-opacity", type=float, default=0.005)
    ap.add_argument("--t-big-scale", type=float, default=0.1)
    ap.add_argument("--growth-rate", type=float, default=1.0)
    ap.add_argument("--opacity-cap", type=float, default=0.8)
    ap.add_argument("--opacity-reg", type=float, default=0.01)
    ap.add_argument("--scale-reg", type=float, default=0.01)
    ap.add_argument("--w-alpha", type=float, default=0.2,
                    help="alpha loss weight; set > 0 for RGBA targets (e.g. 1.0)")
    ap.add_argument("--lambda-ssim", type=float, default=0.2,
                    help="SSIM loss weight (3DGS default 0.2); 0 = pure L2")
    ap.add_argument("--max-splats", type=int, default=1_000_000)
    ap.add_argument("--z-near-cull", type=float, default=0.01,
                    help="view-space depth floor for bin cull; raise for scenes "
                         "where points can land near a camera pose (e.g. 0.2)")
    ap.add_argument("--profile-every", type=int, default=0,
                    help="profile one full step every N iters (0 to disable)")
    ap.add_argument("--no-viewer", action="store_true", help="disable live viewer")
    ap.add_argument("--backend", type=str, default="vulkan", choices=["vulkan", "d3d12"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs/train") / run_name
    cams_dir = run_dir / "cams"
    run_dir.mkdir(parents=True, exist_ok=True)
    cams_dir.mkdir(parents=True, exist_ok=True)

    scene = load_colmap(args.scene, width=args.width, height=args.height,
                        downsample=args.downsample)
    n_cams = len(scene.images)
    n_init = scene.initial_points["means"].shape[0]
    extent = scene.scene_extent
    # Resolved width/height from the scene (may differ from args if args were None).
    W, H = scene.images[0].camera.image_size
    print(f"loaded {n_cams} cameras, {n_init} init points, scene_extent={extent:.3f}")
    #print(f"schedule: densify [{args.densify_from}, {args.densify_to}] every {args.densify_every}, "
    #      f"opacity_reset@{args.opacity_reset_iters or 'none'}")

    print(f"schedule: densify [{args.densify_from}, {args.densify_to}] every {args.densify_every}")

    cfg = DensifyConfig(
        n_scoring_views=args.n_scoring_views,
        loss_thresh=args.loss_thresh,
        importance_thresh=args.importance_thresh,
        prune_thresh=args.prune_thresh,
        t_split_scale=args.t_split_scale * extent,
        t_opacity=args.t_opacity,
        t_big_scale=args.t_big_scale * extent,
        growth_rate=args.growth_rate,
        opacity_cap=args.opacity_cap,
    )

    (run_dir / "config.json").write_text(json.dumps({
        "scene": args.scene,
        "iters": args.iters,
        "width": W, "height": H,
        "preview_every": args.preview_every, "log_every": args.log_every,
        "densify_every": args.densify_every,
        "densify_from": args.densify_from, "densify_to": args.densify_to,
        #"opacity_reset_iters": args.opacity_reset_iters,
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

    use_alpha = args.w_alpha > 0
    targets_srgb = [load_image(e, scene.image_dir, W, H,
                               load_alpha=use_alpha)
                    for e in scene.images]
    targets_linear = [_srgb_to_linear(t[..., :3]) for t in targets_srgb]
    for cam_i, (entry, tgt) in enumerate(zip(scene.images, targets_srgb)):
        cam_dir = cams_dir / f"cam_{cam_i:02d}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        if use_alpha and tgt.shape[-1] == 4:
            # Save RGBA so transparent background is visible in the GT.
            gt_rgba = (tgt * 255 + 0.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(gt_rgba, mode="RGBA").save(cam_dir / "gt.png")
        else:
            gt_rgb = (tgt[..., :3] * 255 + 0.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(gt_rgb, mode="RGB").save(cam_dir / "gt.png")

    device = create_device(backend=args.backend)
    pipeline = Pipeline(
        device=device, width=W, height=H,
        max_splats=args.max_splats,
    )
    pipeline.z_near_cull = args.z_near_cull
    pipeline.opacity_reg = args.opacity_reg
    pipeline.scale_reg = args.scale_reg
    pipeline.w_alpha = args.w_alpha
    pipeline.lambda_ssim = args.lambda_ssim
    pipeline.set_gaussians_from_numpy(**scene.initial_points)
    pipeline.clear_tracking()

    # Launch live viewer.
    viewer = None
    if not args.no_viewer:
        viewer = Viewer(device, width=W, height=H,
                        max_splats=args.max_splats)
        viewer.start()

    # Pre-upload every target image as its own GPU texture. Per-iter we
    # just swap the active reference — no CPU→GPU upload in the hot loop.
    target_textures = [pipeline.create_target_texture(t) for t in targets_srgb]

    _render_all_cameras(pipeline, scene, 0, cams_dir)

    profile_json = (run_dir / "profile_log.jsonl").open("w", buffering=1) if args.profile_every > 0 else None
    log_txt = (run_dir / "log.txt").open("w", buffering=1)
    log_json = (run_dir / "log.jsonl").open("w", buffering=1)
    densify_json = (run_dir / "densify_log.jsonl").open("w", buffering=1)
    log_txt.write(f"# v2 training log — scene={args.scene}, backend={args.backend}\n")
    log_txt.write(f"# iters={args.iters}, resolution={W}x{H}, n_cams={n_cams}\n")
    log_txt.write(f"# densify: every {args.densify_every} iters in "
                  f"[{args.densify_from}, {args.densify_to}]\n")

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

        # Update viewer every 100 iters.
        if viewer and it % 100 == 0:
            viewer.update_gaussians(pipeline)

        # Densification schedule.
        do_densify = (args.densify_from <= it <= args.densify_to
                      and (it - args.densify_from) % args.densify_every == 0)
        do_opacity_reset = (it % 3000 == 0 and it >= args.densify_from and it <= args.densify_to)
        do_log = args.log_every > 0 and ((it % args.log_every == 0) or (it == 1) or (it == args.iters))
        do_preview = args.preview_every > 0 and ((it % args.preview_every == 0) or (it == args.iters))

        if do_opacity_reset:
            pipeline.reset_opacity(cfg)
            densify_json.write(json.dumps({"iter": it, "event": "opacity_reset"}) + "\n")
            print(f"iter {it}: opacity reset")

        if do_densify:
            densify_cfg = cfg

            # Sample N cameras for importance scoring.
            n_score = min(densify_cfg.n_scoring_views, n_cams)
            score_indices = rng.choice(n_cams, size=n_score, replace=False)
            score_cameras = [scene.images[i].camera for i in score_indices]
            score_targets = [target_textures[i] for i in score_indices]

            pipeline.compute_importance_scores(score_cameras, score_targets, densify_cfg)
            stats = pipeline.densify(densify_cfg)
            pipeline.clear_tracking()
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
        if it % 10 == 0:
            print(f"iter {it}: train step done.")

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

        # Per-phase profiling: time each phase of a full training step.
        if profile_json and args.profile_every > 0 and it % args.profile_every == 0:
            import time as _time
            pipeline.device.wait_for_idle()

            prof_cam_idx = int(rng.integers(0, n_cams))
            prof_entry = scene.images[prof_cam_idx]
            pipeline.set_target_texture(target_textures[prof_cam_idx])
            prof_lrs = AdamLRs(mean=lr_mean_init * (decay ** (it - 1)))

            # Warm up (avoid cold-cache effects).
            pipeline._clear_grads()
            pipeline._run_forward(prof_entry.camera)
            pipeline._run_loss()
            pipeline._run_backward()
            pipeline._run_regularization()
            pipeline._run_adam(prof_lrs)
            pipeline.device.wait_for_idle()

            timings = {}

            t0 = _time.perf_counter()
            pipeline._clear_grads()
            pipeline.device.wait_for_idle()
            timings["clear_grads_ms"] = (_time.perf_counter() - t0) * 1000

            t0 = _time.perf_counter()
            pipeline._run_forward(prof_entry.camera)
            pipeline.device.wait_for_idle()
            timings["forward_ms"] = (_time.perf_counter() - t0) * 1000

            t0 = _time.perf_counter()
            pipeline._run_loss()
            pipeline.device.wait_for_idle()
            timings["loss_ms"] = (_time.perf_counter() - t0) * 1000

            t0 = _time.perf_counter()
            pipeline._run_backward()
            pipeline.device.wait_for_idle()
            timings["backward_ms"] = (_time.perf_counter() - t0) * 1000

            t0 = _time.perf_counter()
            pipeline._run_regularization()
            pipeline.device.wait_for_idle()
            timings["regularization_ms"] = (_time.perf_counter() - t0) * 1000

            t0 = _time.perf_counter()
            pipeline._run_adam(prof_lrs)
            pipeline.device.wait_for_idle()
            timings["adam_ms"] = (_time.perf_counter() - t0) * 1000

            timings["total_ms"] = sum(timings.values())
            timings["iter"] = it
            timings["n_gaussians"] = cur_n_gaussians

            profile_json.write(json.dumps(timings) + "\n")
            print(f"iter {it}: PROFILE  "
                  f"fwd={timings['forward_ms']:.1f}  loss={timings['loss_ms']:.1f}  "
                  f"bwd={timings['backward_ms']:.1f}  reg={timings['regularization_ms']:.1f}  "
                  f"adam={timings['adam_ms']:.1f}  total={timings['total_ms']:.1f} ms  "
                  f"n_g={cur_n_gaussians}")

    if viewer:
        viewer.stop()
    if profile_json:
        profile_json.close()
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
        t_srgb = targets_srgb[cam_i][..., :3]
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
