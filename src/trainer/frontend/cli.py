"""CLI parser for the v2 trainer.

Owns all ``argparse`` work — the single boundary that produces the untyped
Namespace. Extracted from ``scripts/v2_train.py`` to thin the entry script; flag
names, defaults, and help text are preserved EXACTLY (frozen by
tests/test_cli_defaults.py). Callers should resolve the Namespace into the typed
config (``trainer.config_types``) rather than threading it around.
"""

from __future__ import annotations

import argparse

from trainer.config.config_types import DensificationMode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="datasets/360_v2/bicycle")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--position-lr-max-steps", type=int, default=3000,
                    help="horizon for the exponential mean-LR decay (3DGS "
                         "position_lr_max_steps). independent of --iters. Ideally should match --iters though")
    parser.add_argument("--downsample", type=int, default=4,
                    help="downsample factor (1=full res, 2=half, 4=quarter). "
                         "Uses images_N dir if it exists, otherwise resizes on load.")
    parser.add_argument("--width", type=int, default=None,
                    help="force render width (overrides --downsample)")
    parser.add_argument("--height", type=int, default=None,
                    help="force render height (overrides --downsample)")
    parser.add_argument("--preview-every", type=int, default=3000)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--render-all-every", type=int, default=0,
                    help="dump PNGs for ALL cameras every N iters")
    parser.add_argument("--eval-every", type=int, default=3000,
                    help="every N iters, render ALL cameras and compute the honest "
                         "average PSNR (same metric as the final_psnr.json summary). "
                         "Writes results to eval_curve.jsonl in the run dir with "
                         "iter, elapsed_s (pure training wall-clock, excluding eval "
                         "time), and avg_psnr_lin/srgb/srgb_masked.")
    parser.add_argument("--eval", dest="eval_holdout", type=int, default=1,
                    help="paper eval protocol (3DGS/SAD-GS --eval, llffhold=8): hold "
                         "out every 8th camera (name-sorted) from training and report "
                         "PSNR on ONLY those held-out views, as the mean of per-image "
                         "PSNR (the paper metric). 1 (default) = on; 0 = train on all "
                         "cameras and report over all cameras (internal/legacy). When "
                         "on, held-out frames are never sampled by the training loop "
                         "and never feed the structure-densify eta accumulator.")
    parser.add_argument("--eval-holdout-stride", type=int, default=8,
                    help="holdout stride for --eval (3DGS llffhold=8). Test set is "
                         "cameras where index %% stride == 0.")
    parser.add_argument("--psnr-curve", action="store_true",
                    help="capture held-out TEST-set PSNR every --psnr-curve-every iters, "
                         "sampled one iteration BEFORE each densify step (densify-1) so the "
                         "reading reflects the settled pre-densify state. Writes "
                         "psnr_curve.jsonl (streamed) + psnr_curve.json (iters[]/psnr_srgb[], "
                         "built for cross-run averaging) and renders psnr_curve.png (PSNR vs "
                         "iter) into the run dir at the end. Independent of --eval-every. "
                         "Average many runs with scripts/plot_psnr_curves.py.", default=False)
    parser.add_argument("--psnr-curve-every", type=int, default=0,
                    help="capture cadence for --psnr-curve. A sample is taken at iter `it` "
                         "when (it+1) %% this == 0, i.e. one iter before each densify boundary "
                         "when --densify-every divides this (default 100 aligns with the "
                         "default densify schedule).")
    parser.add_argument("--debug-bin-viz", action="store_true",
                    help="during the --eval-every sweep, write per-view debug "
                         "heatmaps: a per-tile binned-splat-count map, a per-pixel "
                         "blended-overlap map, and a per-view overlap histogram + "
                         "stats to <run>/cams/cam_NN/binviz_* and binviz.jsonl. "
                         "Reads buffers the forward already fills (no extra kernel) "
                         "but adds readback + PNG encode per view, so it's off by "
                         "default. Requires --eval-every > 0.", default=False)
    parser.add_argument("--densify-every", type=int, default=100)
    parser.add_argument("--densify-from", type=int, default=100)
    parser.add_argument("--densify-to", type=int, default=1900)
    #ap.add_argument("--opacity-reset-iters", type=int, nargs="*", default=[3000])
    parser.add_argument("--n-scoring-views", type=int, default=10)
    parser.add_argument("--t-opacity", type=float, default=0.005,
                    help="prune splats with sigmoid(opacity) below this. MUST stay BELOW "
                         "--opacity-reset-value (default 0.01): opacity reset clamps every "
                         "splat down to the reset value, so a floor above it prunes the "
                         "entire scene the instant reset fires. 3DGS: prune 0.005 < reset 0.01.")
    parser.add_argument("--t-big-scale", type=float, default=0.1)
    parser.add_argument("--prune-scale3d", type=float, default=0.1,
                    help="prune splats whose max 3D scale exceeds this * scene_extent "
                         "(gsplat prune_scale3d). Only active after the first opacity reset.")
    parser.add_argument("--reset-every", type=int, default=1200,
                    help="opacity-reset interval (gsplat reset_every). Big-scale prune is "
                         "disabled until after the first reset.")
    parser.add_argument("--opacity-reg", type=float, default=0.00)
    parser.add_argument("--scale-reg", type=float, default=0.00)
    parser.add_argument("--w-alpha", type=float, default=0.0,
                    help="alpha loss weight; set > 0 for RGBA targets (e.g. 1.0)")
    parser.add_argument("--lambda-ssim", type=float, default=0.2,
                    help="SSIM loss weight (3DGS default 0.2); 0 = pure L2")
    parser.add_argument("--max-splats", type=int, default=10_000_000)
    parser.add_argument("--z-near-cull", type=float, default=0.01,
                    help="view-space depth floor for bin cull; raise for scenes "
                         "where points can land near a camera pose (e.g. 0.2)")
    parser.add_argument("--profile-every", type=int, default=0,
                    help="profile one full step every N iters (0 to disable)")
    parser.add_argument("--morton-every", type=int, default=0,
                    help="Morton-order (Z-curve) re-sort of the gaussian + Adam "
                         "buffers every N iters, PLUS once after each densify. 0 "
                         "(default) = disabled. Improves cache locality of "
                         "subsequent per-splat reads (bin_count, blend backward, "
                         "adam) as the array grows large. Follows clear_tracking "
                         "(and clear_structure_scores on SAD runs) since window "
                         "accumulators are slot-indexed and must be reset when "
                         "slots are permuted.")
    parser.add_argument("--cluster-cull", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Cluster-frustum cull before each forward: bound the "
                         "128-splat clusters (Morton-adjacent) by a per-cluster "
                         "AABB and skip clusters fully outside the view frustum. "
                         "On by default; pass --no-cluster-cull to disable. "
                         "Requires --morton-every > 0 for the clusters to be "
                         "spatially coherent (without it, each cluster spans the "
                         "whole scene). Mask-in-bin_count design (no compaction, "
                         "no slot remap) — every downstream pass is unaware.")
    parser.add_argument("--densify-mode",
                    choices=[m.value for m in DensificationMode],
                    default=DensificationMode.STRUCTURE_AWARE.value,
                    help="densification strategy the loop runs on a densify tick "
                         "(default: structure_aware). Modes: 'none' = no "
                         "densification; 'simple' "
                         "= reference 3DGS threshold-based clone/split; "
                         "'structure_aware' = SAD-GS anisotropic grid-split "
                         "(arXiv 2604.28016) with an interleaved AbsGS gradient "
                         "filler on the non-structure densify ticks.")
    parser.add_argument("--sad-tau-split", type=float, default=0.4,
                    help="SAD split_ratio_threshold: split if high_ratio (fraction of views "
                         "with scalar eta>1) exceeds this AND the gradient is high (SADGS 0.8)")
    parser.add_argument("--sad-tau-prune", type=float, default=0.8,
                    help="SAD prune_ratio_threshold: prune if low_ratio (fraction of views "
                         "with eta<0.1) exceeds this (SADGS 0.8)")
    parser.add_argument("--sad-grad-threshold", type=float, default=1e-5,
                    help="SAD is_grad_high floor: avg 2D-grad magnitude a gaussian must exceed "
                         "to be split-eligible (SADGS hardcodes 1e-5)")
    parser.add_argument("--sad-accum-every", type=int, default=10,
                    help="accumulate an eta sample every N iters (SADGS iteration%%10==0). "
                         "eta_max is a running max, so sampling every iter (=1) inflates it "
                         "and oversizes splits. Default 10 matches SADGS.")
    parser.add_argument("--sad-structure-every", type=int, default=500,
                    help="DUAL SCHEDULE (paper Sec 4): run the aggressive structure-aware "
                         "densify every N iters (paper: 500), while the gentle AbsGS gradient "
                         "densify runs every --densify-every (paper: 100) to populate empty "
                         "regions in between. Set equal to --densify-every to disable the dual "
                         "schedule (structure densify every step, no AbsGS interleave).")
    parser.add_argument("--sad-freq-grad-threshold", type=float, default=0.00002,
                    help="SAD is_active_vis per-view grad floor (SADGS freq_grad_threshold). "
                         "eta is only folded in from views where the gaussian has a high "
                         "per-view screen gradient; rescaled by W*H*3 into our accumulator's "
                         "sum-of-abs convention. 0 disables the per-view grad gate.")
    parser.add_argument("--sad-split-power", type=float, default=0.5,
                    help="SAD: per-axis split factor n_k = ceil(eta_max ** power); 0.5 = sqrt (paper)")
    parser.add_argument("--sad-max-split-factor", type=int, default=8,
                    help="SAD: per-axis cap on the grid split factor n_k (SADGS max_clones_per_axis=8)")
    parser.add_argument("--sad-ks-scale-power", type=float, default=1.0,
                    help="SAD: child scale along axis k = sigma_k / n_k**power (SADGS "
                         "ks_scale_power). 1.0 for Mip-NeRF 360 (variance-preserving with the "
                         "sqrt(12) child spacing); 1.2 for Tanks & Temples / Deep Blending.")
    parser.add_argument("--sad-structure-min-opacity", type=float, default=0.1,
                    help="structure-pass opacity prune floor. SADGS uses 0.1 (20x the "
                         "warmup 0.005), pruning the post-split population hard so weak "
                         "children don't survive to compound into the next structure pass. "
                         "Default 0.005 = current behavior; set 0.1 to match the reference.")
    parser.add_argument("--sad-opacity-cap", type=float, default=0.8,
                    help="post-densify opacity cap applied to ALL survivors after the "
                         "structure prune (SADGS inverse_sigmoid(min(opacity,0.8))). "
                         "0 = off (current). Set 0.8 to match the reference (measured: "
                         "ref it500-post opacity p90=p99=max=0.8).")
    parser.add_argument("--sad-relative-reset", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="use SADGS RELATIVE opacity reset (opacity *= decay) instead of "
                         "the gsplat CLAMP-to-0.01. Preserves strong/weak ordering through "
                         "reset. On by default; pass --no-sad-relative-reset to disable. "
                         "REQUIRED when --sad-structure-min-opacity=0.1 so the floor "
                         "doesn't wipe the scene at the first post-reset structure pass.")
    parser.add_argument("--sad-reset-decay", type=float, default=0.1,
                    help="RELATIVE-reset multiplier (SADGS opacity_reset_decay=0.1). "
                         "Only used with --sad-relative-reset.")
    parser.add_argument("--densify-diag", action="store_true",
                    help="DIAGNOSTIC: read back GPU state around every densify "
                         "tick and log growth trajectory, anisotropic fan-out "
                         "distribution, eta-gate selectivity, and spatial-"
                         "concentration (Gini) metrics to densify_diag.jsonl. "
                         "Opt-in — adds a wait_for_idle + full buffer readback per "
                         "densify tick (~17 ticks/run), off the hot path but not "
                         "free. Use to debug runaway / untargeted densification.")
    parser.add_argument("--sad-match-loss-lrs", type=int, default=1,
                    help="when --densify-mode structure_aware: 1 (default) = use SADGS's loss (lambda_l2) "
                         "and per-parameter LRs so we train exactly as they do; 0 = keep our "
                         "own loss/LR defaults and only swap the densification strategy.")
    parser.add_argument("--sad-lambda-l2", type=float, default=2.0,
                    help="SADGS additive L2 loss weight (their opt.lambda_l2). Applied when "
                         "--densify-mode structure_aware and --sad-match-loss-lrs (else --lambda-l2 is used).")
    parser.add_argument("--lambda-l2", type=float, default=0.0,
                    help="additive L2 loss weight for non-SAD runs (0 = pure L1(+SSIM))")
    parser.add_argument("--grad-thresh", type=float, default=0.0005,
                    help="2D-gradient densification threshold. gsplat uses 0.0002 for "
                         "vanilla grad and ~0.0008 for absgrad; our accumulator is an "
                         "abs-sum (absgrad-like), so 0.0008 matches. Lower = more growth.")
    parser.add_argument("--no-viewer", action="store_true", help="disable live viewer stream")
    parser.add_argument("--launch-viewer", action="store_true",
                    help="DEPRECATED / no-op: the live viewer is now in-process "
                         "and starts automatically unless --no-viewer is given.")
    parser.add_argument("--wait-for-viewer", action="store_true",
                    help="after initializing the gaussians, stream the initial "
                         "state and BLOCK until the viewer's 'Start Training' "
                         "button is pressed (or Enter here). Lets you inspect "
                         "the init before training begins. No-op with --no-viewer.")
    parser.add_argument("--stream-max-splats", type=int, default=7_000_000,
                    help="capacity of the live-viewer shared-memory stream (splats). "
                         "Snapshots beyond this are truncated for viewing only.")
    parser.add_argument("--stream-every", type=int, default=10,
                    help="push a gaussian snapshot to the in-process viewer every "
                         "N iters. Now a zero-copy GPU->GPU blit, so a fine cadence "
                         "is cheap (was coarse when it required a GPU->CPU readback).")
    parser.add_argument("--backend", type=str, default="vulkan", choices=["vulkan", "d3d12"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mip-splatting", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Enable Mip-Splatting (Yu et al., CVPR 2024): a per-gaussian 3D "
                         "smoothing filter (from the TRAIN cameras) folded into the "
                         "projected covariance, plus the 2D Mip antialiasing filter with "
                         "differentiable opacity compensation. Targets the high-frequency "
                         "detail gap vs SADGS (which uses the same 3D filter). Default on; "
                         "pass --no-mip-splatting for the bit-identical vanilla renderer.")
    parser.add_argument("--mip-kernel-size", type=float, default=0.1,
                    help="Mip-Splatting 2D low-pass dilation (mip canonical 0.1). Only used "
                         "with --mip-splatting; vanilla uses 0.3 with no compensation.")
    parser.add_argument("--sh-degree", type=int, default=0, choices=[0, 1, 2, 3],
                    help="Spherical-harmonics degree for view-dependent color (classic "
                         "3DGS). 0 (default) = view-independent SH band-0 (DC) only, the "
                         "original bit-identical path. 1-3 enables higher bands, stored in "
                         "a separate features_rest buffer allocated only when >0, activated "
                         "one band every --sh-band-interval iters (oneupSHdegree).")
    parser.add_argument("--sh-band-interval", type=int, default=1000,
                    help="Iterations between activating successive SH bands (classic 3DGS "
                         "oneupSHdegree). Only used with --sh-degree > 0.")
    parser.add_argument("--sample-mode", type=str, default="sadgs", choices=["ours", "sadgs"],
                    help="per-iteration train-camera draw. 'ours' (default): "
                         "np.random.default_rng(seed).integers over the name-sorted "
                         "train indices (with replacement, 1/iter). 'sadgs': replicate "
                         "the reference SADGS trainer's EXACT draw — Python random.seed(seed), "
                         "a load-time random.shuffle of train then test cameras (INRIA "
                         "Scene shuffle=True), first epoch consumed in that load-shuffled "
                         "order, then random.shuffle on every refill (epoch cycling, "
                         "without replacement), batch_size=1. Use for a like-for-like "
                         "PSNR comparison that isolates the renderer from the data order.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--init-opacity", type=float, default=0.1, help="initial opacity for all splats")
    parser.add_argument("--grow-scale3d", type=float, default=0.001,
                    help="Scale factor for 3D growth of Gaussians (for split/clone routing).")
    parser.add_argument("--opacity-reset-value", type=float, default=0.01, help="value to reset opacity to when doing opacity reset")
    parser.add_argument("--profile-run", action="store_true",
                    help="Nsight Graphics profiling mode. Runs a NORMAL training loop "
                         "(densify + logs still on, per your schedule) but also opens "
                         "a Vulkan window and Present()s one frame per train_step so "
                         "Nsight has a frame boundary to capture on. Auto-disables the "
                         "viewer stream (a second device pollutes the capture) and the "
                         "final I/O-heavy metrics pass.")
    parser.add_argument("--capture-iter", type=int, default=0,
                    help="Iteration at which scripts/running/capture_gpu_trace.py "
                         "freezes the workload for the GPU-Trace collection window "
                         "(external capture, see --capture-external). Presents a "
                         "bit-identical frozen frame from here so ngfx.exe can sample "
                         "HW counters over a stable steady state. 0 (default) = no "
                         "capture.")
    parser.add_argument("--capture-external", action="store_true", default=False,
                    help="Headless capture driven by the Nsight CLI launcher "
                         "(ngfx.exe), NOT by self-injection. In this mode we do "
                         "NOT load NGFX_Injection (ngfx.exe injects for us) and "
                         "do NOT trigger a capture; ngfx.exe starts/stops the "
                         "trace via --start-after-frames/--limit-to-frames and "
                         "AUTO-SAVES the .ngfx-gputrace to its --output-dir (no "
                         "HUD Save/Open needed). The trainer just presents a "
                         "stable frozen workload at --capture-iter for exactly "
                         "--capture-frames frames, then exits. Launch this via "
                         "scripts/capture_gpu_trace.py, which computes the frame "
                         "counts and invokes ngfx.exe. Implies --profile-run.")
    parser.add_argument("--capture-frames", type=int, default=240,
                    help="External-capture only: number of frozen, bit-identical "
                         "frames to Present() at --capture-iter before the "
                         "process exits. Must comfortably exceed the CLI's "
                         "start-after + limit-to window so the trace lands "
                         "entirely inside the frozen steady state.")
    parser.add_argument("--flat-bucket-backward",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Use the flat warp-per-bucket backward (live-bucket "
                         "compaction + indirect one-warp-per-bucket dispatch) "
                         "instead of the per-tile per-splat backward. "
                         "Gradient-equivalent. Pass --no-flat-bucket-backward "
                         "for the per-tile path (e.g. A/B timing).")
    parser.add_argument("--backward-pixel-stride", type=int, default=1,
                    choices=[1, 2, 4, 8, 16],
                    help="Stochastic pixel-subsampling stride for the backward "
                         "rasterizer (BACKWARD_PIXEL_STRIDE shader define). "
                         "1 = exact full-pixel backward; N>1 visits 1/N of tile "
                         "pixels per iter (unbiased SGD estimator, rotating "
                         "phase) for a faster backward at some gradient-noise "
                         "cost. Injected as a compile-time define.")
    parser.add_argument("--extinction-hardness", type=float, default=1.0,
                    help="Compositor extinction hardness k (a shader UNIFORM, so "
                         "no recompile). 1.0 (default) = standard alpha-over "
                         "(bit-identical). k>1 hardens the per-splat transmittance "
                         "to (1-alpha)^k so T decays faster => stronger occlusion "
                         "culling (earlier early-out in forward AND backward) at a "
                         "quality tradeoff. Beer-Lambert-style; requires the "
                         "per-splat backward (the default).")
    parser.add_argument("--extinction-hardness-warmup", type=int, default=500,
                    help="Steps to ramp the extinction hardness from 1.0 up to "
                         "--extinction-hardness AFTER the first densification. "
                         "k is held at 1.0 (standard alpha-over) until the first "
                         "densify fires, then log-linearly ramps 1 -> target over "
                         "this many steps (like the mean-LR schedule), so "
                         "extinction only sharpens once geometry starts forming. "
                         "0 = jump straight to the target after the first densify; "
                         "<0 = disable the schedule (constant target from step 0, "
                         "the pre-schedule behavior). No-op when "
                         "--extinction-hardness <= 1.")
    parser.add_argument("--final-test-psnr", action="store_true", default=False,
                    help="After the timed loop, compute ONLY the held-out "
                         "TEST-set PSNR (mean per-image sRGB, the paper metric "
                         "behind sadgs_vs_slang) and write test_psnr.json. "
                         "Lightweight: renders just the test cams, no full "
                         "per-camera PNG dump / PLY. Does not affect train_wall_s.")
    parser.add_argument("--compute-final-metrics", action="store_true", default=False)
    return parser
