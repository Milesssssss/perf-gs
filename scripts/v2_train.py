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
import sys
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


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """True sRGB OETF — the exact inverse of _srgb_to_linear (used on GT in
    the loss). Using an approximate power(1/2.2) instead lifts darks
    (~+0.03 at 0.05) and compresses contrast, so a converged render displays
    greyer than its GT even at zero loss. Matching the real curve makes the
    preview and GT line up."""
    c = np.clip(c, 0.0, 1.0)
    low = c * 12.92
    high = 1.055 * np.power(np.clip(c, 1e-7, None), 1.0 / 2.4) - 0.055
    return np.where(c <= 0.0031308, low, high)


def _save_srgb_png(rgb_linear: np.ndarray, path: Path) -> None:
    rgb_srgb = _linear_to_srgb(rgb_linear)
    rgba = np.concatenate([rgb_srgb, np.ones_like(rgb_srgb[..., :1])], axis=-1)
    png = (rgba * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(png, mode="RGBA").save(path)


def _render_all_cameras(pipeline: Pipeline, scene, iter_idx: int, cams_dir: Path) -> None:
    for cam_i, entry in enumerate(scene.images):
        pipeline.render(entry.camera)
        img = pipeline.read_output_image()[..., :3]
        out_path = cams_dir / f"cam_{cam_i:02d}" / f"iter_{iter_idx:06d}.png"
        _save_srgb_png(img, out_path)

def run_profiling_step(pipeline: Pipeline, scene, n_cams: int, rng, lr_mean_init, it: int, decay: float) -> None:
    import time as _time
    pipeline.device.wait_for_idle()

    prof_cam_idx = int(rng.integers(0, n_cams))
    prof_entry = scene.images[prof_cam_idx]
    pipeline.set_target_texture(scene.images[prof_cam_idx].target_texture)
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
    timings["n_gaussians"] = pipeline.n_gaussians

    profile_json.write(json.dumps(timings) + "\n")
    print(f"iter {it}: PROFILE  "
            f"fwd={timings['forward_ms']:.1f}  loss={timings['loss_ms']:.1f}  "
            f"bwd={timings['backward_ms']:.1f}  reg={timings['regularization_ms']:.1f}  "
            f"adam={timings['adam_ms']:.1f}  total={timings['total_ms']:.1f} ms  "
            f"n_g={pipeline.n_gaussians}")




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="datasets/360_v2/garden")
    #ap.add_argument("--scene", type=str, default="datasets/synthetic/suzanne_blender")

    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--downsample", type=int, default=4,
                    help="downsample factor (1=full res, 2=half, 4=quarter). "
                         "Uses images_N dir if it exists, otherwise resizes on load.")
    ap.add_argument("--width", type=int, default=None,
                    help="force render width (overrides --downsample)")
    ap.add_argument("--height", type=int, default=None,
                    help="force render height (overrides --downsample)")
    ap.add_argument("--preview-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=10000) 
    ap.add_argument("--densify-every", type=int, default=100)
    ap.add_argument("--densify-from", type=int, default=500)
    ap.add_argument("--densify-to", type=int, default=15000)
    #ap.add_argument("--opacity-reset-iters", type=int, nargs="*", default=[3000])
    ap.add_argument("--n-scoring-views", type=int, default=10)
    ap.add_argument("--loss-thresh", type=float, default=0.1)
    ap.add_argument("--t-opacity", type=float, default=0.005,
                    help="prune splats with sigmoid(opacity) below this. MUST stay BELOW "
                         "--opacity-reset-value (default 0.01): opacity reset clamps every "
                         "splat down to the reset value, so a floor above it prunes the "
                         "entire scene the instant reset fires. 3DGS: prune 0.005 < reset 0.01.")
    ap.add_argument("--t-big-scale", type=float, default=0.1)
    ap.add_argument("--prune-scale3d", type=float, default=0.1,
                    help="prune splats whose max 3D scale exceeds this * scene_extent "
                         "(gsplat prune_scale3d). Only active after the first opacity reset.")
    ap.add_argument("--reset-every", type=int, default=3000,
                    help="opacity-reset interval (gsplat reset_every). Big-scale prune is "
                         "disabled until after the first reset.")
    ap.add_argument("--opacity-cap", type=float, default=0.8)
    ap.add_argument("--opacity-reg", type=float, default=0.00)
    ap.add_argument("--scale-reg", type=float, default=0.00)
    ap.add_argument("--w-alpha", type=float, default=0.0,
                    help="alpha loss weight; set > 0 for RGBA targets (e.g. 1.0)")
    ap.add_argument("--lambda-ssim", type=float, default=0.2,
                    help="SSIM loss weight (3DGS default 0.2); 0 = pure L2")
    ap.add_argument("--max-splats", type=int, default=6_500_000)
    ap.add_argument("--z-near-cull", type=float, default=0.01,
                    help="view-space depth floor for bin cull; raise for scenes "
                         "where points can land near a camera pose (e.g. 0.2)")
    ap.add_argument("--profile-every", type=int, default=0,
                    help="profile one full step every N iters (0 to disable)")
    ap.add_argument("--simple-densify", action="store_true",
                    help="use reference 3DGS threshold-based densification instead of score-ranked")
    ap.add_argument("--grad-thresh", type=float, default=0.0004,
                    help="2D-gradient densification threshold. gsplat uses 0.0002 for "
                         "vanilla grad and ~0.0008 for absgrad; our accumulator is an "
                         "abs-sum (absgrad-like), so 0.0008 matches. Lower = more growth.")
    ap.add_argument("--no-viewer", action="store_true", help="disable live viewer stream")
    ap.add_argument("--launch-viewer", action="store_true",
                    help="auto-launch the standalone viewer as a child process "
                         "(still fully decoupled; a viewer crash won't affect training)")
    ap.add_argument("--stream-max-splats", type=int, default=2_000_000,
                    help="capacity of the live-viewer shared-memory stream (splats). "
                         "Snapshots beyond this are truncated for viewing only.")
    ap.add_argument("--stream-every", type=int, default=200,
                    help="stream a gaussian snapshot to the live viewer every N iters "
                         "(each push does a GPU readback; keep coarse).")
    ap.add_argument("--backend", type=str, default="vulkan", choices=["vulkan", "d3d12"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--init-opacity", type=float, default=0.1, help="initial opacity for all splats")
    ap.add_argument("--grow-scale3d", type=float, default=0.01, help="GSs with 3d scale (normalized by scene_scale below this "
                    "value will be duplicated. Above will be split. Default is 0.01")
    ap.add_argument("--opacity-reset-value", type=float, default=0.01, help="value to reset opacity to when doing opacity reset")
    ap.add_argument("--profile-run", action="store_true",
                    help="Nsight Graphics profiling mode: skip all I/O/previews/"
                         "densify/final-metrics, open a window and Present() one frame "
                         "per training step so Nsight Graphics has a frame boundary to "
                         "capture on. Nsight Graphics is a graphics-API debugger that "
                         "hooks the Vulkan device and delimits work by Present, so it "
                         "needs a real windowed, presenting Vulkan app (it cannot "
                         "connect to a headless process). Runs until the window is "
                         "closed; each captured frame is one full train_step with its "
                         "push_debug_group phase markers (clear_grads/forward/loss/"
                         "backward/regularization/adam).")
    ap.add_argument("--profile-warmup", type=int, default=50,
                    help="train_steps to run before entering the present loop under "
                         "--profile-run (reach steady state / warm caches / compile "
                         "kernels so a capture reflects real hot-loop cost).")
    args = ap.parse_args()

    if args.profile_run:
        # A profile run is a clean, self-contained presenting Vulkan app: no
        # viewer stream (a second device/window pollutes the capture), and no
        # validation layers (the Khronos validation layer conflicts with
        # Nsight's injection layer and breaks the connection).
        args.no_viewer = True
        args.backend = "vulkan"

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs/train") / run_name
    cams_dir = run_dir / "cams"
    run_dir.mkdir(parents=True, exist_ok=True)
    cams_dir.mkdir(parents=True, exist_ok=True)

    scene = load_colmap(args.scene, width=args.width, height=args.height,
                        downsample=args.downsample, init_opacity=args.init_opacity)
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
        t_opacity=args.t_opacity,
        t_big_scale=args.t_big_scale * extent,
        opacity_cap=args.opacity_cap,
        t_opacity_reset_value=args.opacity_reset_value
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
            "t_opacity": cfg.t_opacity,
            "t_big_scale": cfg.t_big_scale,
            "opacity_cap": cfg.opacity_cap,
            "grow_scale3d": args.grow_scale3d
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

    # Linear targets are only needed for the PSNR/MSE readback (current cam
    # on log iters + all cams in the final metrics pass). Computing all N
    # upfront wastes ~38ms/cam on scenes with hundreds of views; memoize
    # each on first use instead.
    _targets_linear_cache: dict[int, np.ndarray] = {}
    def targets_linear(cam_i: int) -> np.ndarray:
        t = _targets_linear_cache.get(cam_i)
        if t is None:
            t = _srgb_to_linear(targets_srgb[cam_i][..., :3])
            _targets_linear_cache[cam_i] = t
        return t

    # gt.png is a debug/preview artifact, not used by training. Re-encoding
    # and writing every GT costs ~100ms/cam; skip any that already exist.
    for cam_i, (entry, tgt) in enumerate(zip(scene.images, targets_srgb)):
        cam_dir = cams_dir / f"cam_{cam_i:02d}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        gt_path = cam_dir / "gt.png"
        if gt_path.exists():
            continue
        if use_alpha and tgt.shape[-1] == 4:
            # Save RGBA so transparent background is visible in the GT.
            gt_rgba = (tgt * 255 + 0.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(gt_rgba, mode="RGBA").save(gt_path)
        else:
            gt_rgb = (tgt[..., :3] * 255 + 0.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(gt_rgb, mode="RGB").save(gt_path)

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

    # Live viewing is fully decoupled: instead of an in-process viewer sharing
    # this device's GPU queue, we stream CPU snapshots of the gaussian buffer
    # into shared memory. A separate `scripts/stream_viewer.py` process renders
    # them independently — no shared device, no cross-thread queue submission,
    # and the viewer crashing/closing cannot affect training. Launch the viewer
    # with the stream name printed below (or it auto-reads outputs/.splat_stream).
    stream_writer = None
    if not args.no_viewer:
        from trainer.splat_stream import SplatStreamWriter, STREAM_LOCK_FILE_DEFAULT
        stream_writer = SplatStreamWriter(capacity=args.stream_max_splats)
        try:
            lock_path = Path(STREAM_LOCK_FILE_DEFAULT)
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(stream_writer.name)
        except OSError:
            pass
        print(f"[SplatStream] serving on {stream_writer.name!r} "
              f"(capacity {args.stream_max_splats}). "
              f"View with: python scripts/stream_viewer.py --stream {stream_writer.name}")

    # Optionally spawn the viewer as a CHILD PROCESS. It still runs fully
    # independently (its own device/queue/window); we just launch and reap it
    # for convenience. A viewer crash does not affect training; when training
    # exits we terminate the child.
    viewer_proc = None
    if stream_writer is not None and args.launch_viewer:
        import subprocess
        viewer_script = str(Path(__file__).resolve().parent / "stream_viewer.py")
        # Pass the scene + downsample so the child viewer loads the same camera
        # poses and offers the snap-to-camera dropdown.
        cmd = [sys.executable, viewer_script,
               "--stream", stream_writer.name,
               "--backend", args.backend,
               "--scene", args.scene,
               "--downsample", str(args.downsample)]
        try:
            viewer_proc = subprocess.Popen(cmd)
            print(f"[SplatStream] launched viewer subprocess (pid {viewer_proc.pid})")
        except OSError as e:
            print(f"[SplatStream] failed to launch viewer: {e}")

    # Pre-upload every target image as its own GPU texture. Per-iter we
    # just swap the active reference — no CPU→GPU upload in the hot loop.
    target_textures = [pipeline.create_target_texture(t) for t in targets_srgb]

    # ------------------------------------------------------------------
    # Nsight Graphics profiling mode.
    #
    # Runs ONLY the hot loop (train_step) with no readbacks/previews/densify/IO,
    # opening a window and Present()ing one frame per step so Nsight Graphics
    # has a frame boundary to capture on. Launch this process as the Nsight
    # Graphics target (or connect to it), then capture a frame — each frame is
    # one full train_step, and the pipeline's push_debug_group phases
    # (clear_grads/forward/loss/backward/regularization/adam) show as the scope
    # tree in the capture.
    if args.profile_run:
        import slangpy as spy
        rng = np.random.default_rng(args.seed)
        lr = 1.6e-4 * extent

        window = spy.Window(width=W, height=H, title="3DGS profile (Nsight Graphics)")
        surface = pipeline.device.create_surface(window)
        surface.configure(width=W, height=H)

        def _present() -> None:
            """Best-effort present of the latest render (the frame boundary
            Nsight captures on). acquire_next_image can RAISE (not just return
            None) when the window is minimized/resized/loses the swapchain, so
            we catch it and reconfigure — a dropped frame must never kill the
            training loop we're profiling."""
            nonlocal surface
            surf_tex = None
            try:
                surf_tex = surface.acquire_next_image()
            except Exception:
                surf_tex = None
            if surf_tex is None:
                # Swapchain went stale (resize/minimize). Rebuild it.
                try:
                    w, h = max(window.width, 1), max(window.height, 1)
                    pipeline.device.wait_for_idle()
                    surface.unconfigure()
                    surface.configure(width=w, height=h)
                except Exception:
                    pass
                return
            enc = pipeline.device.create_command_encoder()
            enc.blit(surf_tex, pipeline.buffers.output_image)
            pipeline.device.submit_command_buffer(enc.finish())
            surface.present()

        def _one_step(i: int) -> None:
            ci = int(rng.integers(0, n_cams))
            pipeline.set_target_texture(target_textures[ci])
            cam = scene.images[ci].camera
            # Training ALWAYS runs (this is what we profile). train_step ran the
            # forward pass for this camera, so output_image holds a valid render.
            pipeline.train_step(cam, lrs=AdamLRs(mean=lr))
            window.process_events()
            _present()

        cur_n = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
        print(f"[profile-run] backend={args.backend} res={W}x{H} n_gaussians={cur_n}")

        # Warm up: reach steady state, warm caches, force kernel compilation so
        # a capture reflects real hot-loop cost, not startup.
        print(f"[profile-run] warmup {args.profile_warmup} steps...")
        for i in range(args.profile_warmup):
            _one_step(i)
        pipeline.device.wait_for_idle()

        print("[profile-run] presenting one frame per train_step for Nsight "
              "Graphics. Launch this process as the ngfx target (or connect), "
              "then capture a frame: each frame is one full train_step "
              "(forward/loss/backward/adam) with its debug-group markers. "
              "Close the window to quit.")
        i = 0
        t_hb = time.perf_counter()
        while not window.should_close():
            _one_step(i)
            i += 1
            if i % 500 == 0:
                pipeline.device.wait_for_idle()
                now = time.perf_counter()
                print(f"[profile-run] {i} frames  ({500/(now-t_hb):.1f} fps)  n_g={cur_n}")
                t_hb = now
        print(f"\n[profile-run] window closed after {i} frames.")
        try:
            window.close()
        except Exception:
            pass
        if stream_writer is not None:
            stream_writer.close()
        return


    profile_json = (run_dir / "profile_log.jsonl").open("w", buffering=1) if args.profile_every > 0 else None
    log_txt = (run_dir / "log.txt").open("w", buffering=1)
    log_json = (run_dir / "log.jsonl").open("w", buffering=1)
    densify_json = (run_dir / "densify_log.jsonl").open("w", buffering=1)
    log_txt.write(f"# v2 training log — scene={args.scene}, backend={args.backend}\n")
    log_txt.write(f"# iters={args.iters}, resolution={W}x{H}, n_cams={n_cams}\n")
    log_txt.write(f"# densify: every {args.densify_every} iters in "
                  f"[{args.densify_from}, {args.densify_to}]\n")

    rng = np.random.default_rng(args.seed)

    # gsplat/3DGS scale the means LR by scene extent (getNerfppNorm radius):
    # means are the only world-unit parameter, so their LR must track scene
    # scale. Without this, centers move ~scene_extent times too slowly and
    # cannot migrate to the surface, leaving elongated/stuck splats.
    lr_mean_init = 1.6e-4 * extent
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

        # Stream a gaussian snapshot to the live viewer every N iters. This
        # does a GPU->CPU readback (which also drains the async queue, bounding
        # how far the CPU runs ahead), so keep the cadence coarse.
        if stream_writer is not None and it % args.stream_every == 0:
            stream_writer.push(pipeline, it)

        # Densification schedule.
        do_densify = (args.densify_from <= it <= args.densify_to
                      and (it - args.densify_from) % args.densify_every == 0)
        do_opacity_reset = (it % args.reset_every == 0 and it >= args.densify_from and it <= args.densify_to)
        do_log = args.log_every > 0 and ((it % args.log_every == 0) or (it == 1) or (it == args.iters))
        do_preview = args.preview_every > 0 and ((it % args.preview_every == 0) or (it == args.iters))

        densify_cfg = cfg
        # Skip densify on reset steps.
        if do_densify and do_opacity_reset:
            do_densify = False

        if do_densify:

            stats = {}

            if args.simple_densify:
                # we gate the too-big prune on step > reset_every: don't delete large kNN-init blobs before densification can split them
                #And normalize the prune threshold by scene extent
                allow_scale_prune = it > args.reset_every
                stats = pipeline.densify_simple(
                    densify_cfg,
                    grad_thresh=args.grad_thresh,
                    scale_thresh=extent * args.grow_scale3d,
                    prune_big_scale=extent * args.prune_scale3d,
                    allow_scale_prune=allow_scale_prune,
                )
            else:
                # Score-ranked densification with importance scoring.
                n_score = min(densify_cfg.n_scoring_views, n_cams)
                score_indices = rng.choice(n_cams, size=n_score, replace=False)
                score_cameras = [scene.images[i].camera for i in score_indices]
                score_targets = [target_textures[i] for i in score_indices]

                n_densify_steps = (args.densify_to - args.densify_from) // args.densify_every + 1
                densify_step = (it - args.densify_from) // args.densify_every + 1

                pipeline.compute_importance_scores(score_cameras, score_targets, densify_cfg)
                stats = pipeline.densify(densify_cfg,
                                            densify_step=densify_step,
                                            n_densify_steps=n_densify_steps,
                                            init_count=n_init)

            pipeline.clear_tracking()
            cur_n_gaussians = stats["final_count"]
            densify_json.write(json.dumps({"iter": it, "event": "densify", **stats}) + "\n")
            print(f"iter {it}: densify  "
                    f"keep={stats['n_keep']} split={stats['n_split']} "
                    f"clone={stats['n_clone']} prune={stats['n_prune']} "
                    f"| pre={stats['pre_count']} -> post={stats['final_count']}")

        # Opacity reset AFTER densify (gsplat resets at the end of
        # step_post_backward). Skipped on densify steps above, so they never
        # collide within one iteration.
        if do_opacity_reset:
            pipeline.reset_opacity(cfg)
            densify_json.write(json.dumps({"iter": it, "event": "opacity_reset"}) + "\n")
            print(f"iter {it}: opacity reset")


        if it % 10 == 0:
            print(f"iter {it}: train step done.")

        if do_log or do_preview:
            #TODO: why is this render here and gated by the densify?
            if do_densify:
                pipeline.render(entry.camera)

            pipeline.device.wait_for_idle()
            rendered = pipeline.read_output_image()[..., :3]
            if not np.all(np.isfinite(rendered)):
                n_nan = int(np.sum(~np.isfinite(rendered)))
                msg = (f"iter {it}: rendered image has {n_nan} non-finite pixels "
                       f"(out of {rendered.size}). Training aborted.")
                print(msg)
                log_txt.write(f"# ABORT: {msg}\n")
                break
            mse = float(np.mean((rendered - targets_linear(cam_idx)) ** 2))
            rolling_loss = mse if rolling_loss is None else 0.98 * rolling_loss + 0.02 * mse
            psnr = _psnr_linear(rendered, targets_linear(cam_idx))

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
            run_profiling_step(pipeline, scene, n_cams, rng, lr_mean_init, it, decay)

    if stream_writer is not None:
        # Signal the viewer to exit, let it see the flag and detach, THEN
        # unlink the shared memory (so the reader never touches freed memory).
        stream_writer.signal_closed()
        if viewer_proc is not None:
            try:
                viewer_proc.wait(timeout=5)
            except Exception:
                viewer_proc.terminate()
        stream_writer.close()

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
        t_lin = targets_linear(cam_i)
        mse_lin = float(np.mean((r_lin - t_lin) ** 2))

        # True sRGB OETF (matches the GT's own encoding), not power(1/2.2),
        # so the sRGB PSNR compares like-for-like against t_srgb.
        r_srgb = _linear_to_srgb(r_lin)
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
