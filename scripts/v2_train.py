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
    c = np.clip(c, 0.0, 1.0)
    low = c * 12.92
    high = 1.055 * np.power(np.clip(c, 1e-7, None), 1.0 / 2.4) - 0.055
    return np.where(c <= 0.0031308, low, high)


def _render_all_cameras(pipeline: Pipeline, scene, iter_idx: int, cams_dir: Path) -> None:
    for cam_i, entry in enumerate(scene.images):
        pipeline.render(entry.camera)
        img = pipeline.read_output_image()[..., :3]
        out_path = cams_dir / f"cam_{cam_i:02d}" / f"iter_{iter_idx:06d}.png"
        rgb_srgb = _linear_to_srgb(img)
        rgba = np.concatenate([rgb_srgb, np.ones_like(rgb_srgb[..., :1])], axis=-1)
        png = (rgba * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(png, mode="RGBA").save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="datasets/360_v2/bicycle")
    #ap.add_argument("--scene", type=str, default="datasets/synthetic/suzanne_blender")

    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--downsample", type=int, default=4,
                    help="downsample factor (1=full res, 2=half, 4=quarter). "
                         "Uses images_N dir if it exists, otherwise resizes on load.")
    ap.add_argument("--width", type=int, default=None,
                    help="force render width (overrides --downsample)")
    ap.add_argument("--height", type=int, default=None,
                    help="force render height (overrides --downsample)")
    ap.add_argument("--preview-every", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--render-all-every", type=int, default=0,
                    help="dump PNGs for ALL cameras every N iters. 0 (default) "
                         "= follow the --preview-every tick (original behavior). "
                         "Set a coarse value to dump less often: each sweep "
                         "renders + reads back every camera under gpu_lock, "
                         "briefly freezing the in-process viewer.")
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
    ap.add_argument("--grad-thresh", type=float, default=0.00065,
                    help="2D-gradient densification threshold. gsplat uses 0.0002 for "
                         "vanilla grad and ~0.0008 for absgrad; our accumulator is an "
                         "abs-sum (absgrad-like), so 0.0008 matches. Lower = more growth.")
    ap.add_argument("--no-viewer", action="store_true", help="disable live viewer stream")
    ap.add_argument("--launch-viewer", action="store_true",
                    help="DEPRECATED / no-op: the live viewer is now in-process "
                         "and starts automatically unless --no-viewer is given.")
    ap.add_argument("--wait-for-viewer", action="store_true",
                    help="after initializing the gaussians, stream the initial "
                         "state and BLOCK until the viewer's 'Start Training' "
                         "button is pressed (or Enter here). Lets you inspect "
                         "the init before training begins. No-op with --no-viewer.")
    ap.add_argument("--stream-max-splats", type=int, default=6_500_000,
                    help="capacity of the live-viewer shared-memory stream (splats). "
                         "Snapshots beyond this are truncated for viewing only.")
    ap.add_argument("--stream-every", type=int, default=40000,
                    help="push a gaussian snapshot to the in-process viewer every "
                         "N iters. Now a zero-copy GPU->GPU blit, so a fine cadence "
                         "is cheap (was coarse when it required a GPU->CPU readback).")
    ap.add_argument("--backend", type=str, default="vulkan", choices=["vulkan", "d3d12"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--init-opacity", type=float, default=0.1, help="initial opacity for all splats")
    ap.add_argument("--grow-scale3d", type=float, default=0.01, help="GSs with 3d scale (normalized by scene_scale below this "
                    "value will be duplicated. Above will be split. Default is 0.01")
    ap.add_argument("--opacity-reset-value", type=float, default=0.01, help="value to reset opacity to when doing opacity reset")
    ap.add_argument("--profile-run", action="store_true",
                    help="Nsight Graphics profiling mode. Runs a NORMAL training loop "
                         "(densify + logs still on, per your schedule) but also opens "
                         "a Vulkan window and Present()s one frame per train_step so "
                         "Nsight has a frame boundary to capture on. Auto-disables the "
                         "viewer stream (a second device pollutes the capture) and the "
                         "final I/O-heavy metrics pass.")
    ap.add_argument("--capture-iter", type=int, default=0,
                    help="Programmatic Nsight Graphics capture: trigger a capture at "
                         "this iteration for apples-to-apples before/after comparisons. "
                         "Requires --profile-run and the Nsight Graphics SDK. Launch "
                         "this process DIRECTLY (from terminal / VSCode) - NOT from "
                         "Nsight UI - because self-injection is the only API path that "
                         "supports programmatic triggering. Set to 0 (default) to "
                         "disable. See src/trainer/ngfx_capture.py for details.")
    ap.add_argument("--capture-activity", type=str, default="trace",
                    choices=["trace", "frame"],
                    help="Which Nsight activity to inject: 'trace' = GPU Trace (HW "
                         "counters, SM/L2/VRAM throughputs); 'frame' = Frame Debugger "
                         "(API-call tree + per-dispatch timings, no HW counters).")
    ap.add_argument("--capture-project", type=str, default=None,
                    help="Path to an .ngfx-proj file saved from Nsight UI "
                         "(File > Save Project As). REQUIRED for 'trace' to "
                         "record any metrics; optional for 'frame'.")
    args = ap.parse_args()

    if args.profile_run:
        # A profile run is a clean, self-contained presenting Vulkan app: no
        # viewer stream (a second device/window pollutes the capture), and no
        # previews (frame-boundary I/O breaks Nsight's step-per-frame cadence).
        # Vulkan is the only backend Nsight Graphics can hook.
        args.no_viewer = True
        args.backend = "vulkan"
        args.preview_every = 0

    # Programmatic Nsight capture: inject BEFORE any Vulkan driver load. If the
    # SDK isn't installed or injection fails, we log and continue without
    # captures - training still runs normally.
    ngfx_cap = None
    if args.capture_iter > 0:
        if not args.profile_run:
            print("[capture] --capture-iter requires --profile-run. ignoring.")
        else:
            from trainer.ngfx_capture import NgfxCapture
            ngfx_cap = NgfxCapture()
            if not ngfx_cap.inject(activity=args.capture_activity,
                                   project_path=args.capture_project):
                ngfx_cap = None
            else:
                proj_msg = (f" (project={args.capture_project})"
                            if args.capture_project else "")
                print(f"[capture] will trigger Nsight {args.capture_activity!r} "
                      f"capture at iter {args.capture_iter}{proj_msg}")

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

    # Under --profile-run, compile shaders with source-level debug info so
    # Nsight Graphics can map counters / SASS back to Slang source lines.
    # (On Vulkan there are no PDBs; this embeds NonSemantic.Shader.DebugInfo in
    # the SPIR-V.) Optimization stays at the default level so the profiled code
    # is the real optimized code.
    device = create_device(backend=args.backend,
                           shader_debug_info=args.profile_run)
    if args.profile_run:
        print("[profile-run] shaders compiled with source-level debug info "
              "(SPIR-V NonSemantic.Shader.DebugInfo) for Nsight correlation.")

    # In-process live viewer runs its render loop on a background thread sharing
    # this device's VkQueue, so training + viewer submits must be serialized by
    # a shared lock (see Pipeline.gpu_lock). Only needed when the viewer is on.
    import threading as _threading
    # RLock, NOT Lock: training-thread paths legitimately nest gpu_lock. render()
    # takes it coarsely, then the transient-encoder branch of _dispatch /
    # _dispatch_indirect inside the forward (radix sort) takes it AGAIN per
    # submit. A non-reentrant Lock self-deadlocks the training thread the first
    # time render() runs (the densify+preview tick around iter 2000). RLock lets
    # the same thread re-enter while still blocking the viewer thread.
    gpu_lock = _threading.RLock() if not args.no_viewer else None

    pipeline = Pipeline(
        device=device, width=W, height=H,
        max_splats=args.max_splats,
        gpu_lock=gpu_lock,
    )
    pipeline.z_near_cull = args.z_near_cull
    pipeline.opacity_reg = args.opacity_reg
    pipeline.scale_reg = args.scale_reg
    pipeline.w_alpha = args.w_alpha
    pipeline.lambda_ssim = args.lambda_ssim
    pipeline.set_gaussians_from_numpy(**scene.initial_points)
    pipeline.clear_tracking()

    # Live viewing is IN-PROCESS: a background-thread Viewer shares this
    # device (serialized by gpu_lock). Streaming a snapshot is a zero-copy
    # GPU->GPU buffer blit (viewer.update_gaussians) — no CPU readback, no
    # numpy, no shared-memory re-upload — so it's cheap enough to run every
    # few iters. Trade-off vs. the old shared-memory stream: the viewer shares
    # the training queue, so a viewer crash can stall training.
    viewer = None
    if not args.no_viewer:
        from trainer.viewer import Viewer
        viewer = Viewer(
            device=device, width=W, height=H,
            max_splats=args.max_splats, fovy=args.fovy if hasattr(args, "fovy") else 45.0,
            gpu_lock=gpu_lock,
        )
        viewer.update_gaussians(pipeline)  # show the initialized gaussians
        viewer.start()
        print("[viewer] in-process live viewer started "
              "(GPU->GPU streaming; window has a Start Training button).")

    # Pre-upload every target image as its own GPU texture. Per-iter we
    # just swap the active reference — no CPU→GPU upload in the hot loop.
    target_textures = [pipeline.create_target_texture(t) for t in targets_srgb]

    # ------------------------------------------------------------------
    # Nsight Graphics profiling: window + present hook.
    #
    # Nsight Graphics is a graphics-API debugger that hooks the Vulkan device
    # and delimits work by Present, so profiling needs a real windowed,
    # presenting Vulkan app. When --profile-run is set we open a window,
    # create a swapchain surface, and Present() one frame per iter — same
    # normal training loop otherwise (densify + logs still run). The viewer
    # stream is disabled to avoid a second device/window polluting the
    # capture. Launch this process as the Nsight target (or connect), then
    # capture a frame: each frame is one full train_step (clear_grads/
    # forward/loss/backward/regularization/adam), plus densify on schedule.
    profile_window = None
    profile_surface = None
    if args.profile_run:
        import slangpy as spy
        profile_window = spy.Window(width=W, height=H,
                                    title="3DGS profile (Nsight Graphics)")
        profile_surface = pipeline.device.create_surface(profile_window)
        profile_surface.configure(width=W, height=H)
        print("[profile-run] presenting one frame per train_step for Nsight "
              "Graphics. Launch this process as the ngfx target (or connect), "
              "then capture a frame. Densify + logs still run on schedule; "
              "close the window (or reach --iters) to quit.")

    def _profile_present() -> None:
        """Best-effort present of the latest render (Nsight's frame boundary).
        acquire_next_image can RAISE (not just return None) when the window is
        minimized/resized/loses the swapchain, so we catch it and reconfigure —
        a dropped frame must never kill the training loop we're profiling."""
        if profile_surface is None:
            return
        nonlocal_surface = profile_surface
        surf_tex = None
        try:
            surf_tex = nonlocal_surface.acquire_next_image()
        except Exception:
            surf_tex = None
        if surf_tex is None:
            try:
                w, h = max(profile_window.width, 1), max(profile_window.height, 1)
                pipeline.device.wait_for_idle()
                nonlocal_surface.unconfigure()
                nonlocal_surface.configure(width=w, height=h)
            except Exception:
                pass
            return
        enc = pipeline.device.create_command_encoder()
        enc.blit(surf_tex, pipeline.buffers.output_image)
        pipeline.device.submit_command_buffer(enc.finish())
        nonlocal_surface.present()


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
    lr_mean_final = 1.6e-6 * extent
    decay = (lr_mean_final / lr_mean_init) ** (1.0 / max(args.iters, 1))

    # Optionally hold before training so the viewer can inspect the initialized
    # gaussians. We push the initial snapshot (iter 0) so the viewer has data,
    # then block until the viewer presses "Start Training" (sets the shared
    # `start` flag) — or the user presses Enter in this terminal as a headless
    # escape hatch. The wait sits BEFORE t_start so it isn't counted as train
    # time. Only meaningful when a stream exists.
    if args.wait_for_viewer and viewer is not None:
        print("[wait-for-viewer] showing initialized gaussians. Waiting for the "
              "viewer's 'Start Training' button (or press Enter here)...")
        _enter = _threading.Event()

        def _wait_enter():
            try:
                input()
            except EOFError:
                pass
            _enter.set()

        _threading.Thread(target=_wait_enter, daemon=True).start()
        while not viewer.start_requested() and not _enter.is_set():
            if viewer._thread is not None and not viewer._thread.is_alive():
                print("[wait-for-viewer] viewer window closed; starting training.")
                break
            time.sleep(0.05)
        print("[wait-for-viewer] starting training.")

    if viewer is not None:
        viewer.set_training_active(True)

    t_start = time.time()
    t_last = t_start
    it_last = 0
    rolling_loss = None
    # Track the current live gaussian count Python-side to avoid a
    # per-iter counter readback. Updated from densify() return values.
    cur_n_gaussians = n_init

    # ---- Queue-depth backpressure (fence-based) ----
    # Nothing in the hot loop drains the GPU queue anymore (wait_for_idle
    # removed from densify_simple). Without backpressure, the CPU races
    # ahead: train_step submits accumulate faster than the GPU can retire
    # them, and any subsequent wait_for_idle (e.g. the log-every render at
    # iter 10000) drains a huge backlog — showing up as multi-minute stalls.
    #
    # Fix: keep a rolling window of the last MAX_QUEUE_DEPTH submit IDs.
    # Before recording a new one, wait for the oldest to complete. This
    # bounds queue depth to ~MAX_QUEUE_DEPTH iterations of latency without
    # ever calling wait_for_idle in the hot path.
    MAX_QUEUE_DEPTH = 1
    submit_id_ring: list[int] = []

    def _bound_queue():
        # Wait on the oldest submit if the queue is full.
        if len(submit_id_ring) >= MAX_QUEUE_DEPTH:
            oldest = submit_id_ring.pop(0)
            if oldest:  # 0 means "no submit yet"
                pipeline.device.wait_for_submit(oldest)

    # Loop termination: normally we stop at args.iters, but under
    # --profile-run we also stop when the user closes the profile window
    # (so Nsight can capture and quit cleanly).
    it = 0
    # State machine for the capture:
    #   armed: at iter == capture_iter we drain the queue and fire the trigger
    #          BEFORE Present, so the presented frame contains exactly one
    #          fully-completed iter's work.
    #   hold : after firing, we keep Present()ing the SAME output_image
    #          (no more train_steps) for capture_hold_frames iters so
    #          Nsight's multi-frame collection sees identical content and
    #          the capture doesn't drift across whatever number of frames
    #          it actually samples.
    # Capture protocol: when we reach iter == capture_iter, we drain the
    # queue after that iter's train_step, run the trigger inline, then
    # burn a fixed number of Presents of the frozen image so Nsight's
    # counter-sampling window falls entirely inside frames of identical
    # content. Then training resumes.
    while it < args.iters:
        if profile_window is not None and profile_window.should_close():
            break
        it += 1

        cam_idx = int(rng.integers(0, n_cams))
        entry = scene.images[cam_idx]
        pipeline.set_target_texture(target_textures[cam_idx])

        # Print cam_idx for the first iter so seed determinism is
        # verifiable across runs (should always print the same value
        # given the same --seed).
        if it == 1:
            print(f"[determinism] iter 1 cam_idx={cam_idx} (should match "
                  f"across same-seed runs)")

        # Backpressure before submitting more work.
        _bound_queue()

        lrs = AdamLRs(mean=lr_mean_init * (decay ** (it - 1)))

        # ---- Deterministic capture protocol for GPU Trace ----
        #
        # GPU Trace is NOT a single-frame debugger. It is a continuous
        # hardware-counter profiler: after arm it samples counters across
        # a collection window (~1-2 seconds) of CONTINUOUSLY-PRESENTED
        # frames, replaying/re-sampling the recurring per-frame workload.
        # It needs a steady stream of representative frames for the whole
        # window - freezing on a static blit after one train_step gives
        # it nothing to sample (that's why earlier attempts were empty /
        # truncated / marker-less).
        #
        # So at capture_iter we enter a dedicated capture loop that keeps
        # presenting representative frames, made deterministic + stable by:
        #   - PIN the camera to capture_iter's camera. Every frame in the
        #     window renders the same view, so whichever frame Nsight
        #     samples shows the same camera across runs.
        #   - lr = 0 so the Adam step still runs (full backward workload
        #     IS profiled) but gaussians never move -> every frame is
        #     bit-identical -> the sampled counters are reproducible.
        #   - No densify / opacity-reset / readback in the loop, so the
        #     workload shape is a pure, stable train_step.
        # The process stays alive (loop until window close) so Nsight's
        # overlay can open the capture UI; killing it early tears the
        # overlay down before the UI appears.
        if ngfx_cap is not None and it == args.capture_iter:
            print(f"[capture] draining queue at iter {it}...")
            pipeline.device.wait_for_idle()
            submit_id_ring.clear()
            try:
                n_g_capture = int(
                    pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
            except Exception:
                n_g_capture = -1
            print(f"[capture] state @ iter {it}: cam_idx={cam_idx}  "
                  f"n_gaussians={n_g_capture}")

            pinned_camera = entry.camera
            # Zero ALL learning rates (mean/scale/rot/opacity/color) so the
            # Adam step runs its full workload but no parameter moves -> every
            # frame in the collection window is bit-identical.
            frozen_lrs = AdamLRs(mean=0.0, scale=0.0, rot=0.0,
                                 opacity=0.0, color=0.0)

            # Warm up a few frames of the pinned, frozen workload BEFORE
            # arming so Nsight sees a steady state (caches warm, no
            # first-frame compile/alloc spikes) when it starts sampling.
            print("[capture] warming up pinned/frozen frames before arm...")
            for _ in range(8):
                if profile_window is not None:
                    profile_window.process_events()
                pipeline.train_step(pinned_camera, lrs=frozen_lrs)
                if profile_window is not None:
                    _profile_present()
            pipeline.device.wait_for_idle()

            print(f"[capture] triggering Nsight capture at iter {it}")
            ngfx_cap.trigger()

            # Continuous capture loop: keep presenting the identical,
            # representative frame so Nsight can complete its multi-sample
            # collection. Runs until the profile window is closed.
            print("[capture] presenting pinned/frozen frames continuously for "
                  "GPU Trace; close the profile window when Nsight is done.")
            import time as _t
            while True:
                if profile_window is not None:
                    if profile_window.should_close():
                        break
                    profile_window.process_events()
                pipeline.train_step(pinned_camera, lrs=frozen_lrs)
                if profile_window is not None:
                    _profile_present()
                else:
                    _t.sleep(0.01)

            if profile_window is not None:
                try:
                    profile_window.close()
                except Exception:
                    pass
            log_txt.close()
            log_json.close()
            densify_json.close()
            if profile_json:
                profile_json.close()
            print("[capture] window closed, exiting.")
            sys.exit(0)

        pipeline.train_step(entry.camera, lrs=lrs)
        submit_id_ring.append(pipeline.last_submit_id)

        # Profile hook: Present() the just-rendered frame so Nsight has a
        # frame boundary to capture on. train_step ran the forward for this
        # camera, so output_image is fresh. Pump events first so window
        # close/resize is picked up promptly.
        if profile_window is not None:
            profile_window.process_events()
            _profile_present()

        # Push a gaussian snapshot to the in-process viewer every N iters.
        # This is a zero-copy GPU->GPU buffer blit (no readback, no numpy, no
        # device drain), so it can run at a fine cadence without stalling the
        # training pipeline. Serialized against the viewer's render via gpu_lock.
        if viewer is not None and it % args.stream_every == 0:
            viewer.update_gaussians(pipeline)

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

            # densify_simple did its own submit_command_buffer inside
            # (one encoder for the whole pass). Add its submit to the
            # backpressure ring so it's subject to the same queue-depth cap.
            if pipeline.last_submit_id:
                submit_id_ring.append(pipeline.last_submit_id)
            pipeline.clear_tracking()
            # densify_simple runs wait-free and returns None for counts
            # (they live GPU-side); the log-every branch below reads the
            # counter on its own cadence. densify()'s stats are still
            # populated synchronously.
            if stats.get("final_count") is not None:
                cur_n_gaussians = stats["final_count"]
            densify_json.write(json.dumps({"iter": it, "event": "densify", **stats}) + "\n")
            pre_s = stats.get("pre_count")
            final_s = stats.get("final_count")
            pre_str = "?" if pre_s is None else str(pre_s)
            final_str = "?" if final_s is None else str(final_s)
            print(f"iter {it}: densify  "
                    f"| pre={pre_str} -> post={final_str}")

        # Opacity reset AFTER densify (gsplat resets at the end of
        # step_post_backward). Skipped on densify steps above, so they never
        # collide within one iteration.
        if do_opacity_reset:
            pipeline.reset_opacity(cfg)
            densify_json.write(json.dumps({"iter": it, "event": "opacity_reset"}) + "\n")
            print(f"iter {it}: opacity reset")


        if it % 10 == 0:
            print(f"iter {it}: train step done. It/s:" f"{(it - it_last) / max(time.time() - t_last, 1e-9):.1f}" " total training time:" f"{(time.time() - t_start) // 60:.0f}min {(time.time() - t_start) % 60:.1f}sec")
            t_last = time.time()
            it_last = it

        if it > 1 and (do_log or do_preview):

            pipeline.render(entry.camera)
            # Refresh cur_n_gaussians on log ticks — wait-free densify_simple
            # keeps the count GPU-side, so we sample it here at log cadence.
            cur_n_gaussians = int(
                pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0]
            )

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

        # All-camera PNG dump. Cadence: --render-all-every if set, else it
        # follows the preview tick (restores the original behavior of dumping
        # on preview). Each camera is a forward + readback under gpu_lock, which
        # briefly freezes the in-process viewer for the sweep — set
        # --render-all-every to a coarse value to make that rarer.
        if args.render_all_every > 0:
            do_render_all = (it % args.render_all_every == 0) or (it == args.iters)
        else:
            do_render_all = do_preview
        if it > 1 and do_render_all:
            _render_all_cameras(pipeline, scene, it, cams_dir)

    # Pure training wall-clock: measured right after the loop, BEFORE the
    # final metrics pass and any teardown, so it's a clean before/after
    # benchmark of the training loop itself. Drain the GPU queue first —
    # queue-depth backpressure lets the CPU run ahead, so without this the
    # timer would miss in-flight GPU work.
    pipeline.device.wait_for_idle()
    train_wall_s = time.time() - t_start
    completed_iters = it
    print(f"[train] loop done: {completed_iters} iters in {train_wall_s:.1f}s "
          f"({completed_iters / max(train_wall_s, 1e-9):.1f} it/s)")
    log_json.write(json.dumps({
        "event": "train_complete",
        "iters": completed_iters,
        "train_wall_s": train_wall_s,
        "it_per_s": completed_iters / max(train_wall_s, 1e-9),
    }) + "\n")

    if profile_window is not None:
        try:
            profile_window.close()
        except Exception:
            pass

    if viewer is not None:
        # Push the final state so the window shows the trained result, then
        # leave the viewer running so the user can inspect it; it stops when
        # they close the window or the process exits.
        try:
            viewer.update_gaussians(pipeline)
        except Exception:
            pass
        print("[viewer] training done — viewer still live; close its window to exit.")
        try:
            if viewer._thread is not None:
                viewer._thread.join()
        except KeyboardInterrupt:
            viewer.stop()

    if profile_json:
        profile_json.close()

    log_txt.close()
    log_json.close()
    densify_json.close()

    # Under --profile-run we skip the final I/O-heavy metrics pass:
    # per-camera renders + PSNR + PLY export pollute the capture window
    # cadence and add minutes of stalled I/O. The profile is done when
    # the user closes the window (or iters completes).
    if args.profile_run:
        print("[profile-run] finished; skipping final metrics / PLY export.")
        return

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

    #render all cameras in the scene and store them as PNGs in the run_dir/cams/cam_XX/ folder, and compute PSNR/MSE metrics

    _render_all_cameras(pipeline, scene, args.iters, cams_dir)

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
        # Pure training-loop wall-clock (excludes this final metrics pass),
        # captured right after the loop with the GPU drained.
        "train_wall_s": train_wall_s,
        "train_it_per_s": completed_iters / max(train_wall_s, 1e-9),
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
