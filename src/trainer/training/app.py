"""v2 training application: the full training loop + orchestration (``main()``).

Extracted verbatim from ``scripts/v2_train.py``, which is now a thin entry that
just calls :func:`main` here. Further decomposition of ``main()`` into
``resources`` / ``outputs`` / ``evaluation`` / ``loop`` modules is a follow-up;
behavior is currently identical to the pre-extraction script.

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
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from trainer.colmap import load_colmap, load_image
from trainer.device import create_device
from trainer.color import linear_to_srgb, srgb_to_linear
from trainer.config import AdamLRs, RenderConfig
from trainer.metrics import psnr as _psnr_image
from trainer.pipeline import Pipeline
from trainer.config_types import ScheduleConfig
from trainer.training.cli import build_parser
from trainer.training.lr_schedule import (
    ExtinctionHardnessSchedule,
    MeanLearningRateSchedule,
)
from trainer.training.queue_limiter import QueueLimiter
from trainer.training.schedule import TrainingSchedule
from trainer.training.densification import DensificationController
from trainer.training.evaluation import (
    eval_test_psnr_srgb,
    render_all_cameras,
    run_eval_sweep,
)
from trainer.training.outputs import RunArtifacts
from trainer.training.post_training import run_post_training
from trainer.training.resources import (
    build_densify_configs,
    compute_morton_bounds,
    resolve_holdout_split,
    resolve_loss_and_lrs,
)
from trainer.training.viewer_control import ViewerSession

# Sibling scripts (scripts/ is on sys.path[0] when v2_train is run directly).
from _v2_profiling import Profiler
from _v2_sampling import CameraSampler


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return srgb_to_linear(c)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    # Clip to displayable [0, 1] before the OETF (preview PNGs + sRGB PSNR).
    return linear_to_srgb(np.clip(c, 0.0, 1.0))


# build_parser() was extracted to trainer.training.cli to thin this entry
# script; imported at the top of this module. The CLI surface + defaults are
# frozen by tests/test_cli_defaults.py.


def main() -> None:
    args = build_parser().parse_args()

    # All Nsight Graphics profiling lives in the Profiler (_v2_profiling.py):
    # session tweaks, self-injection, the present window, and the capture loop.
    # Every hook is a no-op unless --profile-run / --capture-* is set.
    profiler = Profiler(args)
    profiler.configure_session()   # profile runs disable the viewer + previews
    profiler.inject()              # self-injection when --capture-iter

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
    IMAGES_WIDTH, IMAGES_HEIGHT = scene.images[0].camera.image_size
    print(f"loaded {n_cams} cameras, {n_init} init points, scene_extent={extent:.3f}")

    # --- Train/test split (paper eval protocol) ---
    stride, test_indices, train_indices = resolve_holdout_split(args, n_cams)
    # Morton normalization AABB (only used when --morton-every > 0). A stale
    # bbox is fine — the shader saturates out-of-box means.
    morton_world_min, morton_world_max = compute_morton_bounds(
        scene.initial_points["means"])
    #print(f"schedule: densify [{args.densify_from}, {args.densify_to}] every {args.densify_every}, "
    #      f"opacity_reset@{args.opacity_reset_iters or 'none'}")

    print(f"schedule: densify [{args.densify_from}, {args.densify_to}] every {args.densify_every}")

    cfg, structure_cfg = build_densify_configs(args, extent)

    # --- Resolved opacity-bundle settings (CLI -> config object) ---
    # Printed at startup and at every structure pass so it is unambiguous whether
    # the SADGS opacity bundle is actually active. The bundle is OFF by DEFAULT:
    # structure_min_opacity=0.005 (not 0.1) and opacity_cap=0.0 (disabled), and
    # relative_reset off. To match the reference structure pass, run with:
    #   --sad-structure-min-opacity 0.1 --sad-opacity-cap 0.8
    #   --sad-relative-reset --sad-reset-decay 0.1
    def _opacity_bundle_settings() -> dict:
        cap_on = bool(structure_cfg.opacity_cap and structure_cfg.opacity_cap > 0.0)
        return {
            "structure_min_opacity": structure_cfg.t_opacity,
            "opacity_cap_enabled": cap_on,
            "opacity_cap": structure_cfg.opacity_cap,
            "relative_reset_enabled": bool(cfg.reset_relative),
            "reset_decay": cfg.reset_decay,
            "structure_grad_thresh": structure_cfg.grad_threshold,
            "absgs_grad_thresh": args.grad_thresh,
        }

    def _print_opacity_bundle(where: str) -> None:
        s = _opacity_bundle_settings()
        print(f"[opacity-bundle] resolved @ {where}: "
              f"structure_min_opacity={s['structure_min_opacity']} "
              f"opacity_cap_enabled={s['opacity_cap_enabled']} "
              f"opacity_cap={s['opacity_cap']} "
              f"relative_reset_enabled={s['relative_reset_enabled']} "
              f"reset_decay={s['reset_decay']} "
              f"structure_grad_thresh={s['structure_grad_thresh']} "
              f"absgs_grad_thresh={s['absgs_grad_thresh']}", flush=True)

    if args.structure_densify:
        _print_opacity_bundle("startup")
        _s0 = _opacity_bundle_settings()
        if (not _s0["opacity_cap_enabled"]) or _s0["structure_min_opacity"] < 0.1:
            print("[opacity-bundle] WARNING: bundle is INACTIVE at reference "
                  "strength — the 0.8 cap and/or 0.1 floor will NOT run. To match "
                  "SADGS pass: --sad-structure-min-opacity 0.1 --sad-opacity-cap 0.8 "
                  "--sad-relative-reset. Running as configured.", flush=True)

    (run_dir / "config.json").write_text(json.dumps({
        "scene": args.scene,
        "iters": args.iters,
        "width": IMAGES_WIDTH, "height": IMAGES_HEIGHT,
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
        "sample_mode": args.sample_mode,
        "mip_enabled": bool(args.mip_splatting),
        "mip_kernel_size": (args.mip_kernel_size if args.mip_splatting else 0.3),
        "run_name": run_name, "n_cams": n_cams, "n_init_points": n_init,
    }, indent=2))

    use_alpha = args.w_alpha > 0
    targets_srgb = [load_image(e, scene.image_dir, IMAGES_WIDTH, IMAGES_HEIGHT,
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
    # Compile-time shader define for the backward pixel-subsampling stride.
    # Omitted at the default so the shipped path stays bit-identical. (Extinction
    # hardness is a runtime uniform via RenderConfig, not a define.)
    _defines: dict[str, str] = {}
    if args.backward_pixel_stride != 1:
        _defines["BACKWARD_PIXEL_STRIDE"] = f"{args.backward_pixel_stride}u"
    device = create_device(backend=args.backend,
                           shader_debug_info=args.profile_run,
                           defines=_defines or None)
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

    eff_lambda_l2, eff_lambda_ssim, base_lrs = resolve_loss_and_lrs(args)

    pipeline = Pipeline(
        device=device, width=IMAGES_WIDTH, height=IMAGES_HEIGHT,
        max_splats=args.max_splats,
        gpu_lock=gpu_lock,
        cfg=RenderConfig(
            z_near_cull=args.z_near_cull,
            opacity_reg=args.opacity_reg,
            scale_reg=args.scale_reg,
            w_alpha=args.w_alpha,
            lambda_ssim=eff_lambda_ssim,
            lambda_l2=eff_lambda_l2,
            cluster_cull=args.cluster_cull,
            mip_enabled=args.mip_splatting,
            kernel_size=(args.mip_kernel_size if args.mip_splatting else 0.3),
            extinction_hardness=args.extinction_hardness,
        ),
    )
    pipeline.set_gaussians_from_numpy(**scene.initial_points)
    pipeline.clear_tracking()
    # Flat (warp-per-bucket) backward opt-in for A/B + tracing. Off by default.
    pipeline.use_flat_bucket_backward = args.flat_bucket_backward
    # One-time full grad clear. Seeds the "scalar grads are zero at start-of-
    # iter" invariant that lets train_step's hot-path _clear_2d_grad_seeds skip
    # the 14 scalar-grad clears every iter (adam zeros them per-thread at
    # end-of-iter, so subsequent iters are already zero). NDBuffer contents
    # aren't guaranteed zero on allocation, so this initial pass is required.
    pipeline._clear_grads()
    if args.structure_densify:
        # Zero the eta accumulators before the first online-accumulation window
        # (NDBuffer contents are not guaranteed zeroed on allocation).
        pipeline.clear_structure_scores()

    # Mip-Splatting filter-3D logging: summarize buffers.filter_3D over the
    # ACTIVE gaussians so it's verifiable that the filter actually changes
    # across recomputes (init / post-densify / periodic / final). The `sum`
    # checksum catches any change even when min/mean/max round-print the same.
    def _log_filter3d(tag: str, it_val: int | None = None) -> None:
        if not args.mip_splatting:
            return
        n = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
        if n == 0:
            print(f"[filter3d] {tag}: no active gaussians", flush=True)
            return
        f = np.frombuffer(pipeline.buffers.filter_3D.to_numpy().tobytes(),
                          dtype=np.float32)[:n].astype(np.float64)
        it_str = f"it={it_val} " if it_val is not None else ""
        print(f"[filter3d] {tag} {it_str}n={n} min={f.min():.6g} "
              f"mean={f.mean():.6g} max={f.max():.6g} "
              f"nonzero={np.count_nonzero(f) / n:.3f} sum={f.sum():.6g}", flush=True)

    # Mip-Splatting: upload the TRAIN cameras once and compute the initial
    # per-gaussian 3D smoothing filter (recomputed after each densify, since
    # the means move). Held-out test views are excluded to match mip-splatting
    # / SADGS. Off unless --mip-splatting.
    if args.mip_splatting:
        train_cameras = [scene.images[i].camera for i in train_indices]
        pipeline.set_cameras(train_cameras)
        pipeline.compute_3d_filter()
        print(f"[mip] enabled: kernel_size={args.mip_kernel_size}, 3D filter over "
              f"{len(train_cameras)} train cameras", flush=True)
        _log_filter3d("init")

    # Live viewing is IN-PROCESS: a background-thread Viewer shares this
    # device (serialized by gpu_lock). Streaming a snapshot is a zero-copy
    # GPU->GPU buffer blit (viewer.update_gaussians) — no CPU readback, no
    # numpy, no shared-memory re-upload — so it's cheap enough to run every
    # few iters. Trade-off vs. the old shared-memory stream: the viewer shares
    # the training queue, so a viewer crash can stall training.
    viewer = ViewerSession.create(
        enabled=not args.no_viewer, device=device,
        width=IMAGES_WIDTH, height=IMAGES_HEIGHT,
        max_splats=args.max_splats,
        fovy=args.fovy if hasattr(args, "fovy") else 45.0,
        gpu_lock=gpu_lock, pipeline=pipeline,
    )

    # Pre-upload every target image as its own GPU texture. Per-iter we
    # just swap the active reference — no CPU→GPU upload in the hot loop.
    target_textures = [pipeline.create_target_texture(t) for t in targets_srgb]

    # Structure-aware densification: get the multi-scale structure-tensor field
    # S̄ per view and upload each as an rgba32_float texture packing
    # (Sxx, Sxy, Syy, 0). Prefer the on-disk cache built by
    # scripts/preprocess_structure.py (fields take seconds/image to compute);
    # fall back to computing on the fly (with a warning) if none is present.
    structure_textures = []
    if args.structure_densify:
        import slangpy as spy
        from trainer.structure_analysis import cache_path, compute_structure_tensor_field

        def _make_structure_tex(sbar: np.ndarray, label: str):
            packed = np.zeros((IMAGES_HEIGHT, IMAGES_WIDTH, 4), dtype=np.float32)
            packed[..., :3] = sbar
            tex = device.create_texture(
                type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
                width=IMAGES_WIDTH, height=IMAGES_HEIGHT,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label=label)
            tex.copy_from_numpy(np.ascontiguousarray(packed))
            return tex

        cpath = cache_path(args.scene, IMAGES_WIDTH, IMAGES_HEIGHT)
        cached_fields = None
        if cpath.exists():
            size_mb = cpath.stat().st_size / (1024 * 1024)
            print(f"[SAD] loading cache {cpath.name} ({size_mb:.0f} MB)...", flush=True)
            _t_load = time.time()
            data = np.load(cpath)
            # Guard against a stale cache (resolution/count drift). The cache
            # key already encodes resolution+params, but verify count too.
            if int(data["n_cams"]) == n_cams and tuple(data["fields"].shape[1:3]) == (IMAGES_HEIGHT, IMAGES_WIDTH):
                cached_fields = data["fields"]   # forces disk read here
                print(f"[SAD] loaded {n_cams} cached structure-tensor fields "
                      f"in {time.time() - _t_load:.1f}s", flush=True)
            else:
                print(f"[SAD] cache at {cpath} mismatches scene "
                      f"(cached n_cams={int(data['n_cams'])}, "
                      f"shape={data['fields'].shape[1:3]}); recomputing", flush=True)

        # No cache: compute the fields on the GPU (default — fast enough that
        # the preprocess/cache step is no longer required). Fall back to the CPU
        # reference only if the GPU path can't be built.
        st_gpu = None
        if cached_fields is None:
            try:
                from trainer.structure_analysis_gpu import StructureTensorGPU
                st_gpu = StructureTensorGPU(device, IMAGES_WIDTH, IMAGES_HEIGHT)
                print(f"[SAD] no cache; computing {n_cams} fields on GPU...",
                      flush=True)
            except Exception as e:  # noqa: BLE001 - any GPU failure -> CPU path
                print(f"[SAD] GPU structure-tensor unavailable ({e}); computing "
                      f"{n_cams} fields on CPU (slow)...", flush=True)
        else:
            print(f"[SAD] uploading {n_cams} structure-tensor textures to GPU...",
                  flush=True)
        t_sad = time.time()
        _log_stride = max(1, n_cams // 6)  # ~6 progress lines total
        for cam_i in range(n_cams):
            if cached_fields is not None:
                sbar = cached_fields[cam_i]
            elif st_gpu is not None:
                # SADGS computes the structure tensor on the RAW sRGB image
                # (train.py:184 feeds cam.original_image straight in), NOT on
                # linearized RGB. The GPU kernels match the CPU reference.
                sbar = st_gpu.compute(targets_srgb[cam_i][..., :3])
            else:
                sbar = compute_structure_tensor_field(targets_srgb[cam_i][..., :3])
            structure_textures.append(_make_structure_tex(sbar, f"structure_tensor_{cam_i}"))
            if (cam_i + 1) % _log_stride == 0 or cam_i == n_cams - 1:
                print(f"[SAD]   {cam_i + 1:3d}/{n_cams} "
                      f"({time.time() - t_sad:.1f}s elapsed)", flush=True)
        if cached_fields is None:
            _how = "GPU" if st_gpu is not None else "CPU"
            print(f"[SAD] computed {n_cams} fields on {_how} in "
                  f"{time.time() - t_sad:.1f}s", flush=True)

    # All setup uploads (target textures + structure-tensor textures) are done;
    # safe to start the viewer render thread now. See the deferred-start
    # rationale at the viewer construction site above.
    viewer.start()

    # Nsight Graphics hooks the Vulkan device and delimits work by Present, so
    # profiling needs a real windowed, presenting app; open one for
    # --profile-run (no-op otherwise). Each presented frame is one train_step.
    profiler.open_window(pipeline, IMAGES_WIDTH, IMAGES_HEIGHT)

    artifacts = RunArtifacts.create(run_dir, args)
    profile_json = artifacts.profile_json
    log_txt = artifacts.log_txt
    log_json = artifacts.log_json
    densify_json = artifacts.densify_json
    densify_diag_json = artifacts.densify_diag_json
    eval_curve_json = artifacts.eval_curve_json
    psnr_curve_json = artifacts.psnr_curve_json
    bin_viz_json = artifacts.bin_viz_json
    popstat = artifacts.popstat

    _diag = None
    _diag_grad_internal = None
    if densify_diag_json is not None:
        import densify_diag as _diag
        # grad gate floor the structure decision uses internally (host pre-scales
        # the SADGS mean-loss threshold by W*H*3 into our sum-of-abs convention).
        _diag_grad_internal = args.sad_grad_threshold * (IMAGES_WIDTH * IMAGES_HEIGHT * 3.0)
    psnr_curve_records: list[dict] = []
    log_txt.write(f"# v2 training log — scene={args.scene}, backend={args.backend}\n")
    log_txt.write(f"# iters={args.iters}, resolution={IMAGES_WIDTH}x{IMAGES_HEIGHT}, n_cams={n_cams}\n")
    log_txt.write(f"# densify: every {args.densify_every} iters in "
                  f"[{args.densify_from}, {args.densify_to}]\n")

    rng = np.random.default_rng(args.seed)
    # Per-iteration train-camera draw. 'ours' shares this rng so its stream
    # stays interleaved with the densify view-scoring draws; 'sadgs' replicates
    # the reference shuffle/pop (see scripts/_v2_sampling.py).
    sampler = CameraSampler(args.sample_mode, train_indices, test_indices,
                            args.seed, rng)

    # Per-iteration event schedule (pure; extracted to trainer.training.schedule).
    # Owns the densify/reset/structure/preview/render-all/log/eval/psnr/morton/
    # mip timing so the loop just reads events_at(it).
    schedule = TrainingSchedule(ScheduleConfig.from_args(args))

    mean_lr = MeanLearningRateSchedule(
        lr_init=1.6e-4 * extent, lr_final=1.6e-6 * extent,
        max_steps=args.position_lr_max_steps)
    extinction = ExtinctionHardnessSchedule(
        target=args.extinction_hardness, warmup=args.extinction_hardness_warmup)

    if args.wait_for_viewer:
        viewer.wait_for_start()

    viewer.set_training_active(True)

    t_start = time.time()
    t_last = t_start
    it_last = 0
    rolling_loss = None
    # Total time spent inside the --eval-every sweeps. Subtracted from
    # (now - t_start) to yield the pure-training elapsed reported to
    # eval_curve.jsonl, so the convergence plot's X-axis is fair.
    eval_time_total = 0.0
    # Track the current live gaussian count Python-side to avoid a
    # per-iter counter readback. Updated from densify() return values.
    cur_n_gaussians = n_init

    # Queue-depth backpressure (fence-based; trainer.training.queue_limiter).
    # Nothing in the hot loop drains the GPU queue, so without this the CPU
    # QueueLimiter keeps a rolling window of the last max_depth submit
    # IDs and waits on the oldest before recording a new one, no wait_for_idle
    # in the hot path.
    queue_limiter = QueueLimiter(pipeline.device, max_depth=1)

    # Densification subsystem (all four densify modes + eta-window carry, Morton
    # re-sort, mip recompute, popstat + diagnostics). Constructed once with its
    # dependencies; the loop calls densifier.run(it, ev, count) on densify ticks.
    densifier = DensificationController(
        pipeline=pipeline, args=args, densify_cfg=cfg, structure_cfg=structure_cfg,
        extent=extent, n_init=n_init, n_cams=n_cams, rng=rng, scene=scene,
        target_textures=target_textures, image_width=IMAGES_WIDTH,
        image_height=IMAGES_HEIGHT, morton_world_min=morton_world_min,
        morton_world_max=morton_world_max, extinction=extinction,
        queue_limiter=queue_limiter, artifacts=artifacts, diag_module=_diag,
        diag_grad_internal=_diag_grad_internal,
        print_opacity_bundle=_print_opacity_bundle,
    )

    it = 0
    while it < args.iters:

        if profiler.should_stop():
            break
        it += 1

        cam_idx = sampler.next()
        entry = scene.images[cam_idx]
        pipeline.set_target_texture(target_textures[cam_idx])

        if it == 1:
            print(f"[determinism] iter 1 cam_idx={cam_idx} (should match "
                  f"across same-seed runs)")

        # Backpressure before submitting more work.
        queue_limiter.before_submit()

        lrs = AdamLRs(mean=mean_lr.at(it - 1), scale=base_lrs.scale,
                      rot=base_lrs.rot, opacity=base_lrs.opacity, color=base_lrs.color)

        # Deterministic Nsight GPU-Trace capture at --capture-iter (no-op
        # otherwise). When a capture runs it never returns to training: the
        # process finalizes its own logs here and exits.
        if profiler.run_capture_if_due(pipeline, it, entry, cam_idx, queue_limiter):
            artifacts.close_all()
            sys.exit(0)

        # SAD-GS: accumulate this view's eta sample online (paper Sec 3.6) so
        # the multiview fraction gate has ~window-many samples by densify time.
        # CADENCE (SADGS #1): accumulate only every N iters (SADGS
        # `iteration % 10 == 0`), not every iter — eta_max is a running max, so
        # sampling 10x more views only inflates it and oversizes splits.
        # PER-VIEW GRAD GATE (SADGS #2): only fold eta from high-per-view-grad
        # views. SADGS's freq_grad_threshold (0.00002) applies to its per-view
        # NDC mean-loss gradient; our per-view grad seed is a SUM-of-abs over
        # covered pixels (no 1/N), so — exactly as the window gate does — we
        # rescale the floor by W*H*channels into our accumulator's convention.
        struct_tex = None
        struct_kwargs = None
        if (args.structure_densify and args.densify_from < it <= args.densify_to
                and it % args.sad_accum_every == 0):
            struct_tex = structure_textures[cam_idx]
            struct_kwargs = {
                "per_view_grad_thresh": (args.sad_freq_grad_threshold
                                         * IMAGES_WIDTH * IMAGES_HEIGHT * 3.0),
            }

        pipeline.set_extinction_hardness(extinction.at(it))

        pipeline.train_step(entry.camera, lrs=lrs, structure_tensor=struct_tex,
                            structure_kwargs=struct_kwargs)

        queue_limiter.record(pipeline.last_submit_id)

        # Profile hook: present the just-rendered frame so Nsight has a fresh
        # frame boundary to capture on (no-op outside a profile run).
        profiler.present_frame()

        # Push a gaussian snapshot to the in-process viewer every N iters.
        # This is a zero-copy GPU->GPU buffer blit (no readback, no numpy, no
        # device drain), so it can run at a fine cadence without stalling the
        # training pipeline. Serialized against the viewer's render via gpu_lock.
        if it % args.stream_every == 0:
            viewer.push_snapshot(pipeline)

        # Per-iteration schedule (pure; trainer.training.schedule). The tricky
        # invariants — reset-wins-over-densify, ABSOLUTE structure-pass alignment
        # (SADGS `iteration % densification_interval == 0`, NOT phase-offset by
        # densify_from), and preview buffering (+50, or +350 on a reset iter) —
        # all live in events_at() and are unit-tested (tests/test_training_schedule.py).
        ev = schedule.events_at(it)
        do_densify = ev.densify
        do_opacity_reset = ev.opacity_reset
        do_log = ev.log
        do_preview = ev.preview
        if do_densify:
            cur_n_gaussians = densifier.run(it, ev, cur_n_gaussians)

        if do_opacity_reset:
            pipeline.reset_opacity(cfg)
            densify_json.write(json.dumps({"iter": it, "event": "opacity_reset"}) + "\n")
            print(f"iter {it}: opacity reset")

        # Periodic standalone Morton re-sort (in addition to the post-densify
        # sort above). Only fires on non-densify iters, since densify already
        # did one this iter. clear_tracking follows because Morton permutes
        # slots and the grad/visibility window would otherwise reference the
        # wrong gaussians. On the SAD path the eta window must SURVIVE this
        # sort (it's only reset at a structure tick), so we carry eta through
        # the permutation instead of clearing it — otherwise enabling
        # --morton-every would silently wipe the multiview window mid-interval.
        if ev.morton:
            pipeline.morton_sort(morton_world_min, morton_world_max,
                                 carry_eta=args.structure_densify)
            pipeline.clear_tracking()
            if pipeline.last_submit_id:
                queue_limiter.record(pipeline.last_submit_id)
            densify_json.write(json.dumps({"iter": it, "event": "morton_sort"}) + "\n")

        # Mip-Splatting: keep the 3D filter fresh AFTER densification ends.
        if ev.mip_periodic:
            pipeline.compute_3d_filter()
            _log_filter3d("periodic", it)


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
            # Render + target are both sRGB (INRIA/gsplat/SADGS convention);
            # compare directly. This is the reported PSNR metric.
            tgt = targets_srgb[cam_idx][..., :3]
            rendered_clip = np.clip(rendered, 0.0, 1.0)
            mse = float(np.mean((rendered_clip - tgt) ** 2))
            rolling_loss = mse if rolling_loss is None else 0.98 * rolling_loss + 0.02 * mse
            psnr = _psnr_image(rendered_clip, tgt)

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

        do_render_all = ev.render_all
        if it > 1 and do_render_all:
            render_all_cameras(pipeline, scene, it, cams_dir)

        # --- Pre-densify PSNR curve (--psnr-curve): sample TEST-set PSNR one
        # iter before each densify step (it+1 is the densify boundary), so the
        # reading reflects the settled state rather than a just-perturbed one.
        if psnr_curve_json is not None and ev.psnr_curve:
            _pc_t0 = time.time()
            _pc_psnr, _pc_ng = eval_test_psnr_srgb(pipeline, scene, test_indices, targets_srgb)
            _pc_rec = {"iter": it, "psnr_srgb": _pc_psnr, "n_gaussians": _pc_ng}
            psnr_curve_records.append(_pc_rec)
            psnr_curve_json.write(json.dumps(_pc_rec) + "\n")
            eval_time_total += time.time() - _pc_t0
            print(f"[psnr-curve] iter {it:6d}  psnr_srgb={_pc_psnr:5.2f}  "
                  f"n_g={_pc_ng} (pre-densify)")

        # --- PSNR eval sweep (--eval-every): held-out TEST-split paper metric
        # (mean of per-image PSNR/SSIM) into eval_curve.jsonl. Sweep time is
        # excluded from the reported train clock; see run_eval_sweep. ---
        if eval_curve_json is not None and ev.eval:
            eval_time_total = run_eval_sweep(
                pipeline=pipeline, scene=scene, it=it, test_indices=test_indices,
                targets_srgb=targets_srgb, targets_linear=targets_linear,
                cams_dir=cams_dir, eval_curve_json=eval_curve_json,
                bin_viz_json=bin_viz_json, t_start=t_start,
                eval_time_total=eval_time_total, eval_holdout=bool(args.eval_holdout),
            )

    # Drain the GPU queue first queue-depth backpressure lets the CPU run ahead, so without this the
    # timer would miss in-flight GPU work.
    pipeline.device.wait_for_idle()
    train_wall_s = time.time() - t_start
    completed_iters = it
    # Post-training renders (final test PSNR, final metrics, viewer) use the
    # run's TARGET hardness, not whatever mid-ramp value a short run left behind.
    pipeline.set_extinction_hardness(extinction.target)
    print(f"[train] loop done: {completed_iters} iters in {train_wall_s:.1f}s "
          f"({completed_iters / max(train_wall_s, 1e-9):.1f} it/s)")
    log_json.write(json.dumps({
        "event": "train_complete",
        "iters": completed_iters,
        "train_wall_s": train_wall_s,
        "it_per_s": completed_iters / max(train_wall_s, 1e-9),
    }) + "\n")


    if args.final_test_psnr and not args.profile_run:
        _tp_psnr, _tp_ng = eval_test_psnr_srgb(pipeline, scene, test_indices, targets_srgb)
        (run_dir / "test_psnr.json").write_text(json.dumps({
            "psnr_srgb": _tp_psnr,
            "n_gaussians": _tp_ng,
            "n_test_cams": len(test_indices),
            "train_wall_s": train_wall_s,
            "iters": completed_iters,
            "flat_bucket_backward": bool(args.flat_bucket_backward),
        }, indent=2))
        print(f"[final-test-psnr] TEST PSNR sRGB = {_tp_psnr:.3f} dB over "
              f"{len(test_indices)} held-out cams  (n_gaussians={_tp_ng})")

    profiler.close()

    viewer.hold_after_training(pipeline)

    artifacts.close_all()

    # --- PSNR curve (--psnr-curve): write the per-run psnr_curve.json (arrays
    # for cross-run averaging) and render psnr_curve.png (PSNR vs iteration).
    # Done before the --profile-run early return so it always saves. ---
    if args.psnr_curve:
        curve = {
            "scene": args.scene,
            "run_name": run_name,
            "eval_holdout": bool(args.eval_holdout),
            "eval_holdout_stride": stride,
            "n_eval_cams": len(test_indices),
            "psnr_curve_every": args.psnr_curve_every,
            "iters": [r["iter"] for r in psnr_curve_records],
            "psnr_srgb": [r["psnr_srgb"] for r in psnr_curve_records],
            "n_gaussians": [r["n_gaussians"] for r in psnr_curve_records],
        }
        (run_dir / "psnr_curve.json").write_text(json.dumps(curve, indent=2))
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless: no display needed
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
            ax.plot(curve["iters"], curve["psnr_srgb"], "-o", ms=3, lw=1.6,
                    color="#1f77b4")
            ax.set_xlabel("iteration")
            ax.set_ylabel("test PSNR sRGB (dB)")
            _scene_name = Path(args.scene).name or args.scene
            ax.set_title(f"PSNR vs iteration — {_scene_name} ({run_name})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(run_dir / "psnr_curve.png")
            plt.close(fig)
            print(f"[psnr-curve] wrote {run_dir / 'psnr_curve.json'} + psnr_curve.png "
                  f"({len(curve['iters'])} points)")
        except Exception as e:  # noqa: BLE001 - plotting is best-effort
            print(f"[psnr-curve] wrote psnr_curve.json but the PNG failed ({e}); "
                  f"`.venv/Scripts/python.exe -m pip install matplotlib` to enable it.")

    if args.profile_run:
        print("[profile-run] finished; skipping final metrics / PLY export.")
        return

    run_post_training(
        pipeline=pipeline, scene=scene, args=args, run_dir=run_dir,
        cams_dir=cams_dir, targets_srgb=targets_srgb, targets_linear=targets_linear,
        test_indices=test_indices, stride=stride, n_cams=n_cams,
        train_wall_s=train_wall_s, completed_iters=completed_iters, t_start=t_start,
    )


if __name__ == "__main__":
    main()
