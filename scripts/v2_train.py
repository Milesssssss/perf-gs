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
from trainer.export import save_ply
from trainer.color import linear_to_srgb, srgb_to_linear
from trainer.config import AdamLRs, RenderConfig
from trainer.densify import StructureConfig
from trainer.metrics import psnr as _psnr_image, ssim as _ssim_image
from trainer.pipeline import DensifyConfig, Pipeline

# Sibling scripts (scripts/ is on sys.path[0] when v2_train is run directly).
from bin_viz import write_bin_viz
from _v2_profiling import Profiler
from _v2_sampling import CameraSampler


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return srgb_to_linear(c)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    # Clip to displayable [0, 1] before the OETF (preview PNGs + sRGB PSNR).
    return linear_to_srgb(np.clip(c, 0.0, 1.0))


def _render_all_cameras(pipeline: Pipeline, scene, iter_idx: int, cams_dir: Path) -> None:
    for cam_i, entry in enumerate(scene.images):
        pipeline.render(entry.camera)
        # Render is already sRGB (renderer composites in sRGB) — write directly.
        img = np.clip(pipeline.read_output_image()[..., :3], 0.0, 1.0)
        out_path = cams_dir / f"cam_{cam_i:02d}" / f"iter_{iter_idx:06d}.png"
        rgba = np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)
        png = (rgba * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(png, mode="RGBA").save(out_path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    #ap.add_argument("--scene", type=str, default="datasets/360_v2/bicycle") V
    #ap.add_argument("--scene", type=str, default="datasets/360_v2/counter") V
    ap.add_argument("--scene", type=str, default="datasets/360_v2/bonsai")
    #ap.add_argument("--scene", type=str, default="datasets/360_v2/garden")
    #ap.add_argument("--scene", type=str, default="datasets/360_v2/kitchen")
    #ap.add_argument("--scene", type=str, default="datasets/360_v2/room")
    #ap.add_argument("--scene", type=str, default="datasets/360_v2/stump")
    #ap.add_argument("--scene", type=str, default="datasets/360_extra_scenes/flowers")
    #ap.add_argument("--scene", type=str, default="datasets/360_extra_scenes/treehill")

    #ap.add_argument("--scene", type=str, default="datasets/synthetic/suzanne_blender")

    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--position-lr-max-steps", type=int, default=3000,
                    help="horizon for the exponential mean-LR decay (3DGS "
                         "position_lr_max_steps). FIXED, independent of --iters: "
                         "the LR reaches lr_final at this step and holds. Set it "
                         "to roughly your intended --iters for a full decay; do "
                         "NOT let it track a short --iters or the means freeze early.")
    ap.add_argument("--downsample", type=int, default=4,
                    help="downsample factor (1=full res, 2=half, 4=quarter). "
                         "Uses images_N dir if it exists, otherwise resizes on load.")
    ap.add_argument("--width", type=int, default=None,
                    help="force render width (overrides --downsample)")
    ap.add_argument("--height", type=int, default=None,
                    help="force render height (overrides --downsample)")
    ap.add_argument("--preview-every", type=int, default=3000)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--render-all-every", type=int, default=0,
                    help="dump PNGs for ALL cameras every N iters. 0 (default) "
                         "= follow the --preview-every tick (original behavior). "
                         "Set a coarse value to dump less often: each sweep "
                         "renders + reads back every camera under gpu_lock, "
                         "briefly freezing the in-process viewer.")
    ap.add_argument("--eval-every", type=int, default=3000,
                    help="every N iters, render ALL cameras and compute the honest "
                         "average PSNR (same metric as the final_psnr.json summary). "
                         "Writes results to eval_curve.jsonl in the run dir with "
                         "iter, elapsed_s (pure training wall-clock, excluding eval "
                         "time), and avg_psnr_lin/srgb/srgb_masked. 0 (default) = "
                         "disable. Cost: ~2s/eval at 194 cams @ quarter-res.")
    ap.add_argument("--eval", dest="eval_holdout", type=int, default=1,
                    help="paper eval protocol (3DGS/SAD-GS --eval, llffhold=8): hold "
                         "out every 8th camera (name-sorted) from training and report "
                         "PSNR on ONLY those held-out views, as the mean of per-image "
                         "PSNR (the paper metric). 1 (default) = on; 0 = train on all "
                         "cameras and report over all cameras (internal/legacy). When "
                         "on, held-out frames are never sampled by the training loop "
                         "and never feed the structure-densify eta accumulator.")
    ap.add_argument("--eval-holdout-stride", type=int, default=8,
                    help="holdout stride for --eval (3DGS llffhold=8). Test set is "
                         "cameras where index %% stride == 0.")
    ap.add_argument("--psnr-curve", action="store_true",
                    help="capture held-out TEST-set PSNR every --psnr-curve-every iters, "
                         "sampled one iteration BEFORE each densify step (densify-1) so the "
                         "reading reflects the settled pre-densify state. Writes "
                         "psnr_curve.jsonl (streamed) + psnr_curve.json (iters[]/psnr_srgb[], "
                         "built for cross-run averaging) and renders psnr_curve.png (PSNR vs "
                         "iter) into the run dir at the end. Independent of --eval-every. "
                         "Average many runs with scripts/plot_psnr_curves.py.", default=False)
    ap.add_argument("--psnr-curve-every", type=int, default=0,
                    help="capture cadence for --psnr-curve. A sample is taken at iter `it` "
                         "when (it+1) %% this == 0, i.e. one iter before each densify boundary "
                         "when --densify-every divides this (default 100 aligns with the "
                         "default densify schedule).")
    ap.add_argument("--debug-bin-viz", action="store_true",
                    help="during the --eval-every sweep, write per-view debug "
                         "heatmaps: a per-tile binned-splat-count map, a per-pixel "
                         "blended-overlap map, and a per-view overlap histogram + "
                         "stats to <run>/cams/cam_NN/binviz_* and binviz.jsonl. "
                         "Reads buffers the forward already fills (no extra kernel) "
                         "but adds readback + PNG encode per view, so it's off by "
                         "default. Requires --eval-every > 0.", default=False)
    ap.add_argument("--densify-every", type=int, default=100)
    ap.add_argument("--densify-from", type=int, default=100)
    ap.add_argument("--densify-to", type=int, default=1900)
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
    ap.add_argument("--reset-every", type=int, default=1200,
                    help="opacity-reset interval (gsplat reset_every). Big-scale prune is "
                         "disabled until after the first reset.")
    ap.add_argument("--opacity-cap", type=float, default=1.0)
    ap.add_argument("--opacity-reg", type=float, default=0.00)
    ap.add_argument("--scale-reg", type=float, default=0.00)
    ap.add_argument("--w-alpha", type=float, default=0.0,
                    help="alpha loss weight; set > 0 for RGBA targets (e.g. 1.0)")
    ap.add_argument("--lambda-ssim", type=float, default=0.2,
                    help="SSIM loss weight (3DGS default 0.2); 0 = pure L2")
    ap.add_argument("--max-splats", type=int, default=10_000_000)
    ap.add_argument("--z-near-cull", type=float, default=0.01,
                    help="view-space depth floor for bin cull; raise for scenes "
                         "where points can land near a camera pose (e.g. 0.2)")
    ap.add_argument("--profile-every", type=int, default=0,
                    help="profile one full step every N iters (0 to disable)")
    ap.add_argument("--morton-every", type=int, default=0,
                    help="Morton-order (Z-curve) re-sort of the gaussian + Adam "
                         "buffers every N iters, PLUS once after each densify. 0 "
                         "(default) = disabled. Improves cache locality of "
                         "subsequent per-splat reads (bin_count, blend backward, "
                         "adam) as the array grows large. Follows clear_tracking "
                         "(and clear_structure_scores on SAD runs) since window "
                         "accumulators are slot-indexed and must be reset when "
                         "slots are permuted.")
    ap.add_argument("--cluster-cull", action="store_true",
                    help="Cluster-frustum cull before each forward: bound the "
                         "128-splat clusters (Morton-adjacent) by a per-cluster "
                         "AABB and skip clusters fully outside the view frustum. "
                         "Requires --morton-every > 0 for the clusters to be "
                         "spatially coherent (without it, each cluster spans the "
                         "whole scene). Mask-in-bin_count design (no compaction, "
                         "no slot remap) — every downstream pass is unaware.", default=True)
    ap.add_argument("--simple-densify", action="store_true",
                    help="use reference 3DGS threshold-based densification instead of score-ranked")
    ap.add_argument("--importance-score-densify", action="store_true",
                   help="use score-ranked densification instead of reference 3DGS threshold-based (FastGS)")
    ap.add_argument("--structure-densify", action="store_true",
                    help="use structure-aware anisotropic densification (SAD-GS, arXiv 2604.28016): "
                         "precomputes a multi-scale structure-tensor field per view and grid-splits "
                         "Gaussians that under-resolve local texture frequency. Overrides --simple-densify.", default=True)
    ap.add_argument("--sad-tau-split", type=float, default=0.4,
                    help="SAD split_ratio_threshold: split if high_ratio (fraction of views "
                         "with scalar eta>1) exceeds this AND the gradient is high (SADGS 0.8)")
    ap.add_argument("--sad-tau-prune", type=float, default=0.8,
                    help="SAD prune_ratio_threshold: prune if low_ratio (fraction of views "
                         "with eta<0.1) exceeds this (SADGS 0.8)")
    ap.add_argument("--sad-grad-threshold", type=float, default=1e-5,
                    help="SAD is_grad_high floor: avg 2D-grad magnitude a gaussian must exceed "
                         "to be split-eligible (SADGS hardcodes 1e-5)")
    ap.add_argument("--sad-accum-every", type=int, default=10,
                    help="accumulate an eta sample every N iters (SADGS iteration%%10==0). "
                         "eta_max is a running max, so sampling every iter (=1) inflates it "
                         "and oversizes splits. Default 10 matches SADGS.")
    ap.add_argument("--sad-structure-every", type=int, default=500,
                    help="DUAL SCHEDULE (paper Sec 4): run the aggressive structure-aware "
                         "densify every N iters (paper: 500), while the gentle AbsGS gradient "
                         "densify runs every --densify-every (paper: 100) to populate empty "
                         "regions in between. Set equal to --densify-every to disable the dual "
                         "schedule (structure densify every step, no AbsGS interleave).")
    ap.add_argument("--sad-freq-grad-threshold", type=float, default=0.00002,
                    help="SAD is_active_vis per-view grad floor (SADGS freq_grad_threshold). "
                         "eta is only folded in from views where the gaussian has a high "
                         "per-view screen gradient; rescaled by W*H*3 into our accumulator's "
                         "sum-of-abs convention. 0 disables the per-view grad gate.")
    ap.add_argument("--sad-split-power", type=float, default=0.5,
                    help="SAD: per-axis split factor n_k = ceil(eta_max ** power); 0.5 = sqrt (paper)")
    ap.add_argument("--sad-max-split-factor", type=int, default=8,
                    help="SAD: per-axis cap on the grid split factor n_k (SADGS max_clones_per_axis=8)")
    ap.add_argument("--sad-ks-scale-power", type=float, default=1.0,
                    help="SAD: child scale along axis k = sigma_k / n_k**power (SADGS "
                         "ks_scale_power). 1.0 for Mip-NeRF 360 (variance-preserving with the "
                         "sqrt(12) child spacing); 1.2 for Tanks & Temples / Deep Blending.")
    # --- Opacity bundle (SADGS persistence control) + A/B/C ablation knobs ---
    ap.add_argument("--sad-structure-min-opacity", type=float, default=0.1,
                    help="structure-pass opacity prune floor. SADGS uses 0.1 (20x the "
                         "warmup 0.005), pruning the post-split population hard so weak "
                         "children don't survive to compound into the next structure pass. "
                         "Default 0.005 = current behavior; set 0.1 to match the reference.")
    ap.add_argument("--sad-opacity-cap", type=float, default=0.8,
                    help="post-densify opacity cap applied to ALL survivors after the "
                         "structure prune (SADGS inverse_sigmoid(min(opacity,0.8))). "
                         "0 = off (current). Set 0.8 to match the reference (measured: "
                         "ref it500-post opacity p90=p99=max=0.8).")
    ap.add_argument("--sad-relative-reset", action="store_true",
                    help="use SADGS RELATIVE opacity reset (opacity *= decay) instead of "
                         "the gsplat CLAMP-to-0.01. Preserves strong/weak ordering through "
                         "reset. REQUIRED when --sad-structure-min-opacity=0.1 so the floor "
                         "doesn't wipe the scene at the first post-reset structure pass.", default=True)
    ap.add_argument("--sad-reset-decay", type=float, default=0.1,
                    help="RELATIVE-reset multiplier (SADGS opacity_reset_decay=0.1). "
                         "Only used with --sad-relative-reset.")
    ap.add_argument("--densify-diag", action="store_true",
                    help="DIAGNOSTIC: read back GPU state around every densify "
                         "tick and log growth trajectory, anisotropic fan-out "
                         "distribution, eta-gate selectivity, and spatial-"
                         "concentration (Gini) metrics to densify_diag.jsonl. "
                         "Opt-in — adds a wait_for_idle + full buffer readback per "
                         "densify tick (~17 ticks/run), off the hot path but not "
                         "free. Use to debug runaway / untargeted densification.")
    ap.add_argument("--sad-match-loss-lrs", type=int, default=1,
                    help="when --structure-densify: 1 (default) = use SADGS's loss (lambda_l2) "
                         "and per-parameter LRs so we train exactly as they do; 0 = keep our "
                         "own loss/LR defaults and only swap the densification strategy.")
    ap.add_argument("--sad-lambda-l2", type=float, default=2.0,
                    help="SADGS additive L2 loss weight (their opt.lambda_l2). Applied when "
                         "--structure-densify and --sad-match-loss-lrs (else --lambda-l2 is used).")
    ap.add_argument("--lambda-l2", type=float, default=0.0,
                    help="additive L2 loss weight for non-SAD runs (0 = pure L1(+SSIM))")
    ap.add_argument("--grad-thresh", type=float, default=0.0005,
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
    ap.add_argument("--stream-max-splats", type=int, default=7_000_000,
                    help="capacity of the live-viewer shared-memory stream (splats). "
                         "Snapshots beyond this are truncated for viewing only.")
    ap.add_argument("--stream-every", type=int, default=10,
                    help="push a gaussian snapshot to the in-process viewer every "
                         "N iters. Now a zero-copy GPU->GPU blit, so a fine cadence "
                         "is cheap (was coarse when it required a GPU->CPU readback).")
    ap.add_argument("--backend", type=str, default="vulkan", choices=["vulkan", "d3d12"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mip-splatting", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Enable Mip-Splatting (Yu et al., CVPR 2024): a per-gaussian 3D "
                         "smoothing filter (from the TRAIN cameras) folded into the "
                         "projected covariance, plus the 2D Mip antialiasing filter with "
                         "differentiable opacity compensation. Targets the high-frequency "
                         "detail gap vs SADGS (which uses the same 3D filter). Default on; "
                         "pass --no-mip-splatting for the bit-identical vanilla renderer.")
    ap.add_argument("--mip-kernel-size", type=float, default=0.1,
                    help="Mip-Splatting 2D low-pass dilation (mip canonical 0.1). Only used "
                         "with --mip-splatting; vanilla uses 0.3 with no compensation.")
    ap.add_argument("--sample-mode", type=str, default="sadgs", choices=["ours", "sadgs"],
                    help="per-iteration train-camera draw. 'ours' (default): "
                         "np.random.default_rng(seed).integers over the name-sorted "
                         "train indices (with replacement, 1/iter). 'sadgs': replicate "
                         "the reference SADGS trainer's EXACT draw — Python random.seed(seed), "
                         "a load-time random.shuffle of train then test cameras (INRIA "
                         "Scene shuffle=True), first epoch consumed in that load-shuffled "
                         "order, then random.shuffle on every refill (epoch cycling, "
                         "without replacement), batch_size=1. Use for a like-for-like "
                         "PSNR comparison that isolates the renderer from the data order.")
    ap.add_argument("--popstat", action="store_true",
                    help="write outputs/train/<run>/ref_popstat.jsonl with a pre/post "
                         "population snapshot around every densify tick, in the EXACT "
                         "schema of the SADGS reference's ref_popstat.jsonl (count, "
                         "world_max_scale + opacity percentiles, and for structure "
                         "ticks the eta_max/high_ratio/fanout stats). Lets us diff our "
                         "population trajectory line-for-line against "
                         "outputs/sadgs/<run>/ref_popstat.jsonl to locate the gap. "
                         "Reads back GPU buffers on each densify tick (opt-in).")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--init-opacity", type=float, default=0.1, help="initial opacity for all splats")
    ap.add_argument("--grow-scale3d", type=float, default=0.001,
                    help="Scale factor for 3D growth of Gaussians (for split/clone routing).")
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
    ap.add_argument("--capture-external", action="store_true", default=False,
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
    ap.add_argument("--capture-frames", type=int, default=240,
                    help="External-capture only: number of frozen, bit-identical "
                         "frames to Present() at --capture-iter before the "
                         "process exits. Must comfortably exceed the CLI's "
                         "start-after + limit-to window so the trace lands "
                         "entirely inside the frozen steady state.")
    ap.add_argument("--flat-bucket-backward",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Use the flat warp-per-bucket backward (live-bucket "
                         "compaction + indirect one-warp-per-bucket dispatch) "
                         "instead of the per-tile per-splat backward. "
                         "Gradient-equivalent. Pass --no-flat-bucket-backward "
                         "for the per-tile path (e.g. A/B timing).")
    ap.add_argument("--backward-pixel-stride", type=int, default=1,
                    choices=[1, 2, 4, 8, 16],
                    help="Stochastic pixel-subsampling stride for the backward "
                         "rasterizer (BACKWARD_PIXEL_STRIDE shader define). "
                         "1 = exact full-pixel backward; N>1 visits 1/N of tile "
                         "pixels per iter (unbiased SGD estimator, rotating "
                         "phase) for a faster backward at some gradient-noise "
                         "cost. Injected as a compile-time define.")
    ap.add_argument("--extinction-hardness", type=float, default=1.0,
                    help="Compositor extinction hardness k (a shader UNIFORM, so "
                         "no recompile). 1.0 (default) = standard alpha-over "
                         "(bit-identical). k>1 hardens the per-splat transmittance "
                         "to (1-alpha)^k so T decays faster => stronger occlusion "
                         "culling (earlier early-out in forward AND backward) at a "
                         "quality tradeoff. Beer-Lambert-style; requires the "
                         "per-splat backward (the default).")
    ap.add_argument("--final-test-psnr", action="store_true", default=False,
                    help="After the timed loop, compute ONLY the held-out "
                         "TEST-set PSNR (mean per-image sRGB, the paper metric "
                         "behind sadgs_vs_slang) and write test_psnr.json. "
                         "Lightweight: renders just the test cams, no full "
                         "per-camera PNG dump / PLY. Does not affect train_wall_s.")
    ap.add_argument("--compute-final-metrics", action="store_true", default=False)
    return ap


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
    # 3DGS/SAD-GS hold out every Nth camera (name-sorted; the loader already
    # sorts by image name) and report PSNR on ONLY the held-out set. With --eval
    # on, the training loop samples exclusively from train_indices, so test
    # frames are never optimized against and never feed the structure-densify
    # eta accumulator (which reads whatever camera the loop sampled). With --eval
    # off, every camera is both trained and evaluated (internal/legacy behavior).
    stride = max(args.eval_holdout_stride, 1)
    if args.eval_holdout:
        test_indices = [i for i in range(n_cams) if i % stride == 0]
        train_indices = [i for i in range(n_cams) if i % stride != 0]
        print(f"[eval] holdout protocol ON (stride={stride}): "
              f"{len(train_indices)} train / {len(test_indices)} test cameras. "
              f"Training samples train views only; PSNR reported on test views "
              f"as mean-of-per-image dB (paper metric).")
    else:
        test_indices = list(range(n_cams))
        train_indices = list(range(n_cams))
        print(f"[eval] holdout protocol OFF: training on all {n_cams} cameras, "
              f"reporting over all cameras (internal metric).")
    # AABB of the initial point cloud, padded 20% per axis. Used only if
    # --morton-every > 0 (Morton quantization needs a normalized coord frame).
    # A stale bbox is fine — the shader saturates out-of-box means, so
    # densification-induced drift outside these bounds is a cache-locality
    # miss, not a correctness bug. Recompute is skipped for simplicity: the
    # bbox is a heuristic, not an invariant.
    _init_means = scene.initial_points["means"].astype(np.float32).reshape(-1, 3)
    _bbox_min = _init_means.min(axis=0)
    _bbox_max = _init_means.max(axis=0)
    _bbox_extent = np.maximum(_bbox_max - _bbox_min, 1e-6)
    morton_world_min = _bbox_min - 0.2 * _bbox_extent
    morton_world_max = _bbox_max + 0.2 * _bbox_extent
    #print(f"schedule: densify [{args.densify_from}, {args.densify_to}] every {args.densify_every}, "
    #      f"opacity_reset@{args.opacity_reset_iters or 'none'}")

    print(f"schedule: densify [{args.densify_from}, {args.densify_to}] every {args.densify_every}")

    cfg = DensifyConfig(
        n_scoring_views=args.n_scoring_views,
        loss_thresh=args.loss_thresh,
        t_opacity=args.t_opacity,
        t_big_scale=args.t_big_scale * extent,
        opacity_cap=args.opacity_cap,
        t_opacity_reset_value=args.opacity_reset_value,
        # SADGS RELATIVE opacity reset (opacity bundle ablation).
        reset_relative=args.sad_relative_reset,
        reset_decay=args.sad_reset_decay,
    )

    structure_cfg = StructureConfig(
        n_scoring_views=args.n_scoring_views,
        tau_split=args.sad_tau_split,
        tau_prune=args.sad_tau_prune,
        grad_threshold=args.sad_grad_threshold,
        split_power=args.sad_split_power,
        max_split_factor=args.sad_max_split_factor,
        ks_scale_power=args.sad_ks_scale_power,
        # Structure-pass opacity floor (0.005 baseline / 0.1 = SADGS reference).
        t_opacity=args.sad_structure_min_opacity,
        # Post-densify opacity cap (0 off / 0.8 = SADGS reference).
        opacity_cap=args.sad_opacity_cap,
        # clone (small) vs split (large) threshold = SADGS args.dense * extent.
        dense_scale=args.grow_scale3d * extent,
    )

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

    # When running structure-aware densification, optionally train with the
    # EXACT loss + learning rates SADGS uses, so the only variable vs. their
    # setup is our renderer — not the optimization recipe. Controlled by
    # --sad-match-loss-lrs (on by default). See arguments/__init__.py in the
    # SADGS repo for these values (lambda_l2=2.0, lambda_dssim=0.2, and the
    # per-parameter LRs opacity/scaling/rotation/feature).
    sad_match = bool(args.structure_densify and args.sad_match_loss_lrs)
    if sad_match:
        eff_lambda_l2 = args.sad_lambda_l2      # SADGS opt.lambda_l2 = 2.0
        # SSIM lambda: default 0.2 (SADGS opt.lambda_dssim). --lambda-ssim
        # overrides — any explicitly-passed value wins over the sad_match
        # default. This lets A/B experiments like "does turning SSIM off
        # change our convergence?" work without also flipping the LR schedule
        # and L2 term (which --sad-match-loss-lrs=0 would do).
        eff_lambda_ssim = args.lambda_ssim      # default 0.2 matches SADGS
        # SADGS per-parameter LRs (their opt.*_lr). mean/position is scheduled
        # separately below (position_lr_init already matches at 1.6e-4*extent).
        base_lrs = AdamLRs(scale=0.01, rot=0.002, opacity=0.05, color=0.0025)
        print(f"[SAD] matching SADGS loss + LRs: lambda_l2={eff_lambda_l2}, "
              f"lambda_ssim={eff_lambda_ssim}, scale_lr=0.01, rot_lr=0.002, "
              f"opacity_lr=0.05, sh0_lr=0.0025")
    else:
        eff_lambda_l2 = args.lambda_l2
        eff_lambda_ssim = args.lambda_ssim
        base_lrs = AdamLRs()                    # our defaults

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
    viewer = None
    if not args.no_viewer:
        from trainer.viewer import Viewer
        viewer = Viewer(
            device=device, width=IMAGES_WIDTH, height=IMAGES_HEIGHT,
            max_splats=args.max_splats, fovy=args.fovy if hasattr(args, "fovy") else 45.0,
            gpu_lock=gpu_lock,
        )
        # Push the initial snapshot, but DO NOT start the render thread yet.
        # The remaining setup submits ~200 texture uploads (target images,
        # 194 structure-tensor textures) on the trainer thread WITHOUT
        # gpu_lock. If the viewer thread were running, its per-frame submits
        # would race those uploads on the shared VkQueue — Vulkan requires
        # external queue synchronization; violating it manifests as a
        # non-deterministic driver stall that looks like a hang (last
        # console line "[SAD] loading cache ..." with no progress after).
        # viewer.start() moves after the SAD upload loop below, once the
        # trainer setup is entirely done and only the hot loop remains.
        viewer.update_gaussians(pipeline)  # show the initialized gaussians
        print("[viewer] created (render thread starts after setup completes).")

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
    if viewer is not None:
        viewer.start()
        print("[viewer] render thread started.", flush=True)

    # Nsight Graphics hooks the Vulkan device and delimits work by Present, so
    # profiling needs a real windowed, presenting app; open one for
    # --profile-run (no-op otherwise). Each presented frame is one train_step.
    profiler.open_window(pipeline, IMAGES_WIDTH, IMAGES_HEIGHT)

    profile_json = (run_dir / "profile_log.jsonl").open("w", buffering=1) if args.profile_every > 0 else None
    log_txt = (run_dir / "log.txt").open("w", buffering=1)
    log_json = (run_dir / "log.jsonl").open("w", buffering=1)
    densify_json = (run_dir / "densify_log.jsonl").open("w", buffering=1)
    densify_diag_json = (
        (run_dir / "densify_diag.jsonl").open("w", buffering=1)
        if args.densify_diag else None
    )
    if densify_diag_json is not None:
        import densify_diag as _diag
        # grad gate floor the structure decision uses internally (host pre-scales
        # the SADGS mean-loss threshold by W*H*3 into our sum-of-abs convention).
        _diag_grad_internal = args.sad_grad_threshold * (IMAGES_WIDTH * IMAGES_HEIGHT * 3.0)
    # SADGS-schema population trajectory (--popstat): one pre/post snapshot per
    # densify tick into ref_popstat.jsonl, directly diffable against the
    # reference's outputs/sadgs/<run>/ref_popstat.jsonl.
    popstat = None
    if args.popstat:
        from popstat import PopStat
        popstat = PopStat(run_dir / "ref_popstat.jsonl")
        print(f"[popstat] writing SADGS-schema population trajectory to "
              f"{popstat.path}")
    # Full-dataset average-PSNR curve, opened only when --eval-every > 0.
    # Records one JSON line per eval sweep for post-hoc convergence plotting.
    eval_curve_json = (
        (run_dir / "eval_curve.jsonl").open("w", buffering=1)
        if args.eval_every > 0 else None
    )
    # Pre-densify PSNR curve (--psnr-curve): a focused TEST-set PSNR sample
    # taken one iter before each densify step, streamed to psnr_curve.jsonl and
    # collected in-memory for the end-of-run psnr_curve.json + psnr_curve.png.
    psnr_curve_json = (
        (run_dir / "psnr_curve.jsonl").open("w", buffering=1)
        if args.psnr_curve else None
    )
    psnr_curve_records: list[dict] = []
    # Binning/overlap debug viz (--debug-bin-viz): one JSONL line per rendered
    # eval view, plus per-view heatmap PNGs under cams/cam_NN/. Only when both
    # the flag and --eval-every are on.
    bin_viz_json = (
        (run_dir / "binviz.jsonl").open("w", buffering=1)
        if (args.debug_bin_viz and args.eval_every > 0) else None
    )
    if args.debug_bin_viz and args.eval_every <= 0:
        print("[debug-bin-viz] ignored: requires --eval-every > 0.")
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

    # gsplat/3DGS scale the means LR by scene extent (getNerfppNorm radius):
    # means are the only world-unit parameter, so their LR must track scene
    # scale. Without this, centers move ~scene_extent times too slowly and
    # cannot migrate to the surface, leaving elongated/stuck splats.
    lr_mean_init = 1.6e-4 * extent
    lr_mean_final = 1.6e-6 * extent

    # Exponential (log-linear) mean-LR schedule, matching 3DGS/SADGS
    # get_expon_lr_func. CRITICAL: the decay horizon is a FIXED max_steps
    # (position_lr_max_steps), NOT args.iters — otherwise a short --iters run
    # compresses the full 1000x decay into a few hundred steps and freezes the
    # means before any geometry forms (PSNR collapses). t is clamped to [0,1]
    # so the LR saturates at lr_final instead of overshooting toward zero.
    lr_max_steps = args.position_lr_max_steps

    def _mean_lr(step: int) -> float:
        t = min(max(step / max(lr_max_steps, 1), 0.0), 1.0)
        return float(np.exp(np.log(lr_mean_init) * (1.0 - t)
                            + np.log(lr_mean_final) * t))

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
    # Total time spent inside the --eval-every sweeps. Subtracted from
    # (now - t_start) to yield the pure-training elapsed reported to
    # eval_curve.jsonl, so the convergence plot's X-axis is fair.
    eval_time_total = 0.0
    # Track the current live gaussian count Python-side to avoid a
    # per-iter counter readback. Updated from densify() return values.
    cur_n_gaussians = n_init

    def _eval_test_psnr_srgb() -> tuple[float, int]:
        """Mean per-image sRGB PSNR over the held-out TEST split (paper metric)
        + the live gaussian count. Drains the GPU queue so the count and the
        per-cam renders are settled. Used by the --psnr-curve capture."""
        pipeline.device.wait_for_idle()
        n_g = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
        frames = []
        for cam_i in test_indices:
            pipeline.render(scene.images[cam_i].camera)
            r = np.clip(pipeline.read_output_image()[..., :3], 0.0, 1.0)
            frames.append(_psnr_image(r, targets_srgb[cam_i][..., :3]))
        return (float(np.mean(frames)) if frames else 0.0), n_g

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
    while it < args.iters:
        if profiler.should_stop():
            break
        it += 1

        # Sample a TRAIN camera. With --eval, train_indices excludes the
        # held-out test views, so they are never optimized against (nor fed to
        # the structure-densify accumulator, which uses this same camera).
        cam_idx = sampler.next()
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

        lrs = AdamLRs(mean=_mean_lr(it - 1), scale=base_lrs.scale,
                      rot=base_lrs.rot, opacity=base_lrs.opacity, color=base_lrs.color)

        # Deterministic Nsight GPU-Trace capture at --capture-iter (no-op
        # otherwise). When a capture runs it never returns to training: the
        # process finalizes its own logs here and exits.
        if profiler.run_capture_if_due(pipeline, it, entry, cam_idx, submit_id_ring):
            log_txt.close()
            log_json.close()
            densify_json.close()
            if densify_diag_json is not None:
                densify_diag_json.close()
            if eval_curve_json is not None:
                eval_curve_json.close()
            if bin_viz_json is not None:
                bin_viz_json.close()
            if profile_json:
                profile_json.close()
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
        pipeline.train_step(entry.camera, lrs=lrs, structure_tensor=struct_tex,
                            structure_kwargs=struct_kwargs)
        submit_id_ring.append(pipeline.last_submit_id)

        # Profile hook: present the just-rendered frame so Nsight has a fresh
        # frame boundary to capture on (no-op outside a profile run).
        profiler.present_frame()

        # Push a gaussian snapshot to the in-process viewer every N iters.
        # This is a zero-copy GPU->GPU buffer blit (no readback, no numpy, no
        # device drain), so it can run at a fine cadence without stalling the
        # training pipeline. Serialized against the viewer's render via gpu_lock.
        if viewer is not None and it % args.stream_every == 0:
            viewer.update_gaussians(pipeline)

        # Densification schedule.
        do_densify = (args.densify_from < it <= args.densify_to
                      and (it - args.densify_from) % args.densify_every == 0)
        do_opacity_reset = (args.reset_every > 0 and it % args.reset_every == 0 and it >= args.densify_from and it <= args.densify_to)
        do_log = args.log_every > 0 and ((it % args.log_every == 0) or (it == 1) or (it == args.iters))

        # so we leave a buffer of 50 iterations after either opacity resets or densifies
        # TODO: hacky, im just assuming we'll space them more than 350
        do_preview = False
        if args.preview_every > 0:
            bufferred_iters = [itr+350 if itr % args.reset_every == 0 else itr+50 for itr in range(args.preview_every, args.iters, args.preview_every)]
            # Always include the final iteration so a preview is dumped at the
            # end even when preview_every >= iters (range() would otherwise be
            # empty) — and regardless of --compute-final-metrics.
            do_preview = (it in bufferred_iters) or (it == args.iters)

        densify_cfg = cfg
        # Skip densify on reset steps.
        if do_densify and do_opacity_reset:
            do_densify = False

        # DUAL SCHEDULE (paper Sec 4): on structure-densify runs, each densify
        # tick (--densify-every, paper 100) is EITHER the aggressive structure
        # densify (every --sad-structure-every, paper 500) OR the gentle AbsGS
        # gradient densify (the other ticks) that populates empty regions so the
        # structure pass can leverage existing primitives. A tick is a structure
        # step on ABSOLUTE iteration multiples of --sad-structure-every (it %
        # every == 0 -> 500/1000/1500), matching the SADGS reference's
        # `iteration % densification_interval == 0` — NOT phase-offset by
        # densify_from — so our structure ticks land on the same iters as the
        # reference for a line-for-line popstat comparison.
        do_structure_step = (
            args.structure_densify and do_densify
            and (it % args.sad_structure_every == 0)
        )

        if do_densify:

            stats = {}

            # DIAGNOSTIC pre-capture: read back eta/grad/scale + spatial state
            # BEFORE the densify mutates anything, so gate selectivity and
            # fan-out reflect the decision this tick actually made.
            _diag_rec = None
            if densify_diag_json is not None:
                _diag_rec = _diag.capture_pre(
                    pipeline, structure=args.structure_densify,
                    width=IMAGES_WIDTH, height=IMAGES_HEIGHT,
                    structure_cfg=structure_cfg,
                    grad_thresh_internal=_diag_grad_internal,
                    dense_scale=structure_cfg.dense_scale)

            # POPSTAT pre-densify snapshot (SADGS schema). kind mirrors the
            # reference: 'structure' on a SAD structure tick, 'warmup' on the
            # gradient/AbsGS filler ticks. Taken BEFORE densify mutates the
            # buffers, while the eta accumulators are still live.
            _popstat_kind = "structure" if do_structure_step else "warmup"
            if popstat is not None:
                popstat.snapshot(pipeline, it, "pre", _popstat_kind,
                                 extent=extent, structure_cfg=structure_cfg,
                                 split_ratio_threshold=args.sad_tau_split)

            # SAD-GS eta window: a FILLER (AbsGS) tick carries the eta
            # accumulators through its slot permutations (scatterCompactEta +
            # mortonScatterEta) so the multiview frequency window accumulates
            # over the whole structure interval (~50 samples) instead of being
            # reset every filler tick (~10). A STRUCTURE tick instead clears eta
            # afterward to start a fresh window (SADGS zeroes eta right after a
            # normal/structure densification). Non-SAD paths never touch eta.
            sad_carry_eta = args.structure_densify and not do_structure_step

            if do_structure_step:
                # Print the resolved bundle settings at every structure pass so
                # it is provable, per-pass, whether the 0.1 floor / 0.8 cap are
                # active (they are OFF unless the flags were passed).
                _print_opacity_bundle(f"structure pass @ iter {it}")
                stats = pipeline.densify_structure_aware(structure_cfg)
            elif args.structure_densify:
                # Interleaved AbsGS gradient densify (dual-schedule filler). Same
                # gentle gradient clone/split as --simple-densify; scale prune
                # gated on step > reset_every so kNN-init blobs aren't culled
                # before structure densify can subdivide them.
                allow_scale_prune = it > args.reset_every
                stats = pipeline.densify_simple(
                    densify_cfg,
                    grad_thresh=args.grad_thresh,
                    scale_thresh=extent * args.grow_scale3d,
                    prune_big_scale=extent * args.prune_scale3d,
                    allow_scale_prune=allow_scale_prune,
                    carry_eta=True,
                )
            elif args.simple_densify:
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
            elif args.importance_score_densify:
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
            # SAD-GS eta window handling. clear_tracking above only zeros the
            # grad/visibility window (never eta), so the eta buffers are intact
            # here regardless of the branch.
            #   STRUCTURE tick: densify_structure_aware used the plain scatter
            #     (eta NOT carried), so eta is now stale on the new slot set.
            #     Zero it to open a fresh window (SADGS zeroes eta right after a
            #     normal densification).
            #   FILLER tick (sad_carry_eta): densify_simple(carry_eta=True)
            #     already permuted eta through the compaction, so DON'T clear —
            #     the window keeps accumulating across the whole structure
            #     interval. The Morton sort below must then also carry eta.
            if args.structure_densify and not sad_carry_eta:
                pipeline.clear_structure_scores()
            # Morton re-sort after every densify: the fresh gaussians live at
            # the tail of the buffer (appended by split_and_clone), so a
            # Z-order pass restores spatial locality before the next window
            # of forwards. Order matters: it MUST follow the clears above,
            # since Morton permutes slots and any non-zero slot-indexed
            # accumulator (grad/visibility) would then reference the wrong
            # gaussian. On a filler tick we carry eta through the sort too so
            # the just-preserved window isn't scrambled.
            if True:
                pipeline.morton_sort(morton_world_min, morton_world_max,
                                     carry_eta=sad_carry_eta)
                if pipeline.last_submit_id:
                    submit_id_ring.append(pipeline.last_submit_id)
            # Mip-Splatting: recompute the 3D smoothing filter now that the means
            # + count changed (mip-splatting recomputes ~every 100 iters). New
            # children get a valid filter for the next window's forwards.
            if args.mip_splatting:
                pipeline.compute_3d_filter()
                _log_filter3d("post-densify", it)
            # densify_simple runs wait-free and returns None for counts
            # (they live GPU-side); the log-every branch below reads the
            # counter on its own cadence. densify()'s stats are still
            # populated synchronously.
            if stats.get("final_count") is not None:
                cur_n_gaussians = stats["final_count"]
            kind = "structure" if do_structure_step else (
                "absgs" if args.structure_densify else "densify")
            densify_json.write(json.dumps(
                {"iter": it, "event": "densify", "kind": kind, **stats}) + "\n")

            # POPSTAT post-densify snapshot (after densify + morton settled).
            if popstat is not None:
                popstat.snapshot(pipeline, it, "post", _popstat_kind,
                                 extent=extent, structure_cfg=structure_cfg,
                                 split_ratio_threshold=args.sad_tau_split)

            # DIAGNOSTIC post-capture: realized growth + post-densify spatial
            # concentration, after densify + morton have settled.
            if _diag_rec is not None:
                _diag_rec = _diag.capture_post(pipeline, _diag_rec,
                                               structure=args.structure_densify)
                _diag_rec["iter"] = it
                _diag_rec["kind"] = kind
                densify_diag_json.write(json.dumps(_diag_rec) + "\n")
                fo = _diag_rec.get("fanout", {})
                gate = _diag_rec.get("eta_gate", {})
                sp = _diag_rec.get("spatial_post", {})
                print(f"  [diag] {kind:9s} {_diag_rec['pre_count']}->{_diag_rec['post_count']} "
                      f"(x{_diag_rec['growth_factor']}) | "
                      f"trig={gate.get('frac_triggered_of_seen')} "
                      f"prune={gate.get('frac_prune_of_seen')} "
                      f"prod_p90={fo.get('prod_p90')} prod_max={fo.get('prod_max')} "
                      f"| gini={sp.get('gini')} top1%={sp.get('top1pct_cell_share')}")
                # Opacity-bundle assertions (diagnostic mode only): verify the
                # shader actually honored the RESOLVED floor/cap on THIS pass.
                # Asserted against the configured values (not the reference
                # 0.1/0.8), so a default run (floor 0.005, cap off) never
                # false-fails; the assert only bites if the bundle was
                # configured but did not apply on the compacted (active) buffer.
                if kind == "structure":
                    _amin, _amax, _abmsgs = _diag.assert_opacity_bundle(
                        pipeline,
                        floor=structure_cfg.t_opacity,
                        cap_enabled=bool(structure_cfg.opacity_cap
                                         and structure_cfg.opacity_cap > 0.0),
                        cap=structure_cfg.opacity_cap)
                    print(f"  [opacity-bundle assert] iter {it}: min={_amin:.4f} "
                          f"max={_amax:.4f} | " + " | ".join(_abmsgs))
            # Only print a per-densify line when we have real counts — i.e. the
            # synchronous score-ranked densify() path. Wait-free paths (AbsGS
            # + structure) report None and read the counter on log ticks
            # instead, matching densify_simple's discipline.
            pre_s = stats.get("pre_count")
            final_s = stats.get("final_count")
            if pre_s is not None and final_s is not None:
                print(f"iter {it}: {kind:9s}| pre={pre_s} -> post={final_s}")

        # Opacity reset AFTER densify (gsplat resets at the end of
        # step_post_backward). Skipped on densify steps above, so they never
        # collide within one iteration.
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
        if (args.morton_every > 0
                and it % args.morton_every == 0
                and not do_densify):
            pipeline.morton_sort(morton_world_min, morton_world_max,
                                 carry_eta=args.structure_densify)
            pipeline.clear_tracking()
            if pipeline.last_submit_id:
                submit_id_ring.append(pipeline.last_submit_id)
            densify_json.write(json.dumps({"iter": it, "event": "morton_sort"}) + "\n")

        # Mip-Splatting: keep the 3D filter fresh AFTER densification ends.
        # While densifying (it <= densify_to) the post-densify recompute already
        # covers it; once densify stops, the means keep drifting under the
        # position LR, so the filter would freeze at its densify_to positions.
        # Recompute every 100 iters past densify_to (matching mip-splatting's
        # cadence; it skips the last ~100 iters where the filter barely moves).
        if (args.mip_splatting and it > args.densify_to
                and it % 100 == 0 and it < args.iters - 100):
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

        # --- Pre-densify PSNR curve (--psnr-curve): sample TEST-set PSNR one
        # iter before each densify step (it+1 is the densify boundary), so the
        # reading reflects the settled state rather than a just-perturbed one.
        # Kept separate from the --eval-every sweep so it runs at the densify
        # cadence without the bin-viz / SSIM overhead. Its time is folded into
        # eval_time_total so the eval-curve's elapsed_s stays a fair train clock.
        if psnr_curve_json is not None and (it + 1) % max(1, args.psnr_curve_every) == 0:
            _pc_t0 = time.time()
            _pc_psnr, _pc_ng = _eval_test_psnr_srgb()
            _pc_rec = {"iter": it, "psnr_srgb": _pc_psnr, "n_gaussians": _pc_ng}
            psnr_curve_records.append(_pc_rec)
            psnr_curve_json.write(json.dumps(_pc_rec) + "\n")
            eval_time_total += time.time() - _pc_t0
            print(f"[psnr-curve] iter {it:6d}  psnr_srgb={_pc_psnr:5.2f}  "
                  f"n_g={_pc_ng} (pre-densify)")

        # --- PSNR eval sweep (--eval-every). Renders the held-out TEST cameras
        # and reports the paper metric: mean of per-image PSNR (NOT dB-of-mean-
        # MSE). Logs (iter, elapsed_s, psnr_*) to eval_curve.jsonl. Only runs
        # when --eval-every > 0. Time spent here is excluded from `elapsed_s` so
        # the convergence-vs-time plot is a fair comparison of training speed. ---
        if eval_curve_json is not None and it % args.eval_every == 0:
            eval_t0 = time.time()
            # Drain GPU queue so num_gaussians_counter is accurate and no
            # in-flight work skews the per-cam renders.
            pipeline.device.wait_for_idle()
            eval_n_g = int(
                pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0]
            )
            # Accumulate PER-IMAGE PSNR (dB) and average the dB values — the
            # paper's mean-of-per-image metric. Scored on the held-out TEST
            # split (all cams when --eval is off, since test_indices == all).
            psnr_srgb_frames = []
            psnr_lin_frames = []
            psnr_srgb_mask_frames = []
            ssim_srgb_frames = []
            for cam_i in test_indices:
                entry = scene.images[cam_i]
                pipeline.render(entry.camera)
                # Binning/overlap debug viz: read the buffers this forward just
                # filled (tile ranges + per-pixel blended count) and dump the
                # per-view heatmaps + histogram. render() already drained the
                # queue, so the buffers are complete.
                if bin_viz_json is not None:
                    tile_counts, overlap, ntx, nty = pipeline.read_bin_debug()
                    write_bin_viz(
                        cam_dir=cams_dir / f"cam_{cam_i:02d}",
                        iter_idx=it, cam_idx=cam_i,
                        image_name=entry.image_name,
                        tile_counts=tile_counts,
                        num_tiles_x=ntx, num_tiles_y=nty,
                        overlap=overlap,
                        jsonl_handle=bin_viz_json,
                    )
                r_srgb = np.clip(pipeline.read_output_image()[..., :3], 0.0, 1.0)
                t_srgb = targets_srgb[cam_i][..., :3]
                psnr_srgb_frames.append(_psnr_image(r_srgb, t_srgb))
                ssim_srgb_frames.append(_ssim_image(r_srgb, t_srgb))
                r_lin = _srgb_to_linear(r_srgb)
                t_lin = targets_linear(cam_i)
                psnr_lin_frames.append(_psnr_image(r_lin, t_lin))
                mask = (t_srgb.sum(-1) > 0.05) | (r_srgb.sum(-1) > 0.05)
                if mask.sum() > 0:
                    psnr_srgb_mask_frames.append(_psnr_image(r_srgb[mask], t_srgb[mask]))

            def _mean(values: list[float]) -> float:
                return float(np.mean(values)) if values else 0.0

            psnr_srgb = _mean(psnr_srgb_frames)
            ssim_srgb = _mean(ssim_srgb_frames)
            psnr_lin = _mean(psnr_lin_frames)
            psnr_srgb_mask = _mean(psnr_srgb_mask_frames)
            eval_dt = time.time() - eval_t0
            eval_time_total += eval_dt
            elapsed_train = time.time() - t_start - eval_time_total
            eval_curve_json.write(json.dumps({
                "iter": it,
                "elapsed_s": elapsed_train,
                "n_gaussians": eval_n_g,
                "n_eval_cams": len(test_indices),
                "eval_holdout": bool(args.eval_holdout),
                # Paper metric: mean of per-image PSNR/SSIM over the test split.
                "psnr_srgb": psnr_srgb,
                "ssim_srgb": ssim_srgb,
                "psnr_linear": psnr_lin,
                "psnr_srgb_masked": psnr_srgb_mask,
                "eval_dt_s": eval_dt,
            }) + "\n")
            print(f"[eval] iter {it:6d}  "
                  f"psnr_srgb={psnr_srgb:5.2f}  ssim={ssim_srgb:.4f} (test, mean-of-frames)  "
                  f"n_g={eval_n_g:>7d}  train_elapsed={elapsed_train:.1f}s  "
                  f"(eval took {eval_dt:.1f}s)")

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

    # Lightweight held-out TEST PSNR (paper metric, same as sadgs_vs_slang):
    # renders ONLY the test cams — no full per-camera PNG dump / PLY. Runs
    # AFTER train_wall_s is captured, so it never pollutes the timing.
    if args.final_test_psnr and not args.profile_run:
        _tp_psnr, _tp_ng = _eval_test_psnr_srgb()
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
    if densify_diag_json is not None:
        densify_diag_json.close()
    if popstat is not None:
        popstat.close()
    if eval_curve_json is not None:
        eval_curve_json.close()
    if bin_viz_json is not None:
        bin_viz_json.close()

    # --- PSNR curve (--psnr-curve): write the per-run psnr_curve.json (arrays
    # for cross-run averaging) and render psnr_curve.png (PSNR vs iteration).
    # Done before the --profile-run early return so it always saves. ---
    if args.psnr_curve:
        if psnr_curve_json is not None:
            psnr_curve_json.close()
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

    # Under --profile-run we skip the final I/O-heavy metrics pass:
    # per-camera renders + PSNR + PLY export pollute the capture window
    # cadence and add minutes of stalled I/O. The profile is done when
    # the user closes the window (or iters completes).
    if args.profile_run:
        print("[profile-run] finished; skipping final metrics / PLY export.")
        return

    #TODO: Just make stuff under here into a cleaner post-training analysis routine

    # Mip-Splatting: refresh the 3D filter one last time so the final metrics
    # render reflects the FINAL gaussian positions. The periodic recompute above
    # skips the last ~100 iters, so without this the reported PSNR would use a
    # filter ~100-200 iters stale.
    if args.mip_splatting:
        pipeline.compute_3d_filter()
        _log_filter3d("final")

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
    if args.compute_final_metrics:
        _render_all_cameras(pipeline, scene, args.iters, cams_dir)

        for cam_i, entry in enumerate(scene.images):
            pipeline.render(entry.camera)
            # Render is sRGB now (INRIA/gsplat/SADGS convention). The sRGB PSNR
            # (this is the paper-reported metric) is a direct compare vs t_srgb;
            # the linear PSNR is kept for continuity — both are linearized.
            r_srgb = np.clip(pipeline.read_output_image()[..., :3], 0.0, 1.0)
            t_srgb = targets_srgb[cam_i][..., :3]
            mse_srgb = float(np.mean((r_srgb - t_srgb) ** 2))

            r_lin = _srgb_to_linear(r_srgb)
            t_lin = targets_linear(cam_i)
            mse_lin = float(np.mean((r_lin - t_lin) ** 2))

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
                "in_test_split": cam_i in set(test_indices),
                "mse_linear": mse_lin,
                "psnr_linear": _psnr_image(r_lin, t_lin),
                "mse_srgb": mse_srgb,
                "psnr_srgb": _psnr_image(r_srgb, t_srgb),
                "ssim_srgb": _ssim_image(r_srgb, t_srgb),
                "mse_srgb_masked": mse_srgb_mask,
                "psnr_srgb_masked": (_psnr_image(r_srgb[mask], t_srgb[mask])
                                    if mask.sum() > 0 else 0.0),
            })

        # --- Reported metric = paper metric everywhere ---
        # PSNR is ALWAYS the arithmetic mean of per-image PSNR (3DGS/SAD-GS
        # convention; see _psnr_image + scripts/image_metrics.py). The headline
        # `psnr_srgb` is that mean over the held-out TEST split — THE number to
        # compare against the SAD-GS / 3DGS tables. We also break it out by
        # train/all and keep the old dB-of-mean-MSE values only as explicitly
        # suffixed `*_db_of_mean_mse` fields, so nothing silently reports the
        # pessimistic aggregation.
        test_set = set(test_indices)
        test_cams = [c for c in per_cam if c["cam_idx"] in test_set]
        train_cams = [c for c in per_cam if c["cam_idx"] not in test_set]

        def mean_metric(cams, key):
            return float(np.mean([c[key] for c in cams])) if cams else 0.0

        def db_of_mean_mse(total_mse, n): return -10.0 * math.log10(max(total_mse / max(n, 1), 1e-12))

        summary = {
            # === HEADLINE: paper metrics (mean of per-image PSNR/SSIM, test split) ===
            "psnr_srgb": mean_metric(test_cams, "psnr_srgb"),
            "ssim_srgb": mean_metric(test_cams, "ssim_srgb"),
            "psnr_linear": mean_metric(test_cams, "psnr_linear"),
            "psnr_srgb_masked": mean_metric(test_cams, "psnr_srgb_masked"),
            "eval_holdout": bool(args.eval_holdout),
            "eval_holdout_stride": stride,
            "n_test_cams": len(test_cams),
            "n_train_cams": len(train_cams),
            # Same paper metric, other splits (train / all cams).
            "train_psnr_srgb": mean_metric(train_cams, "psnr_srgb"),
            "train_ssim_srgb": mean_metric(train_cams, "ssim_srgb"),
            "all_psnr_srgb": mean_metric(per_cam, "psnr_srgb"),
            "all_ssim_srgb": mean_metric(per_cam, "ssim_srgb"),
            # Legacy dB-of-mean-MSE (pessimistic pooling) — NOT the paper metric,
            # kept only for continuity with old runs. Suffixed so it can't be
            # mistaken for the headline.
            "test_psnr_srgb_db_of_mean_mse": db_of_mean_mse(
                sum(c["mse_srgb"] for c in test_cams), len(test_cams)),
            "all_psnr_srgb_db_of_mean_mse": db_of_mean_mse(total_mse_srgb, n_cams),
            "all_psnr_linear_db_of_mean_mse": db_of_mean_mse(total_mse_lin, n_cams),
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
        if args.eval_holdout:
            print(f"TEST PSNR sRGB (paper):  {summary['psnr_srgb']:6.2f} dB  "
                f"<- mean of per-image dB over {summary['n_test_cams']} held-out views "
                f"(compare to SAD-GS/3DGS table)")
            print(f"TEST SSIM sRGB (paper):  {summary['ssim_srgb']:6.4f}     "
                f"<- mean of per-image SSIM, {summary['n_test_cams']} held-out views")
            print(f"train PSNR/SSIM sRGB:    {summary['train_psnr_srgb']:6.2f} dB / "
                f"{summary['train_ssim_srgb']:.4f}  (mean per-image, {summary['n_train_cams']} train views)")
        else:
            print(f"PSNR sRGB (paper):       {summary['psnr_srgb']:6.2f} dB  "
                f"<- mean of per-image dB over all {n_cams} views (no holdout)")
            print(f"SSIM sRGB (paper):       {summary['ssim_srgb']:6.4f}     "
                f"<- mean of per-image SSIM over all {n_cams} views")
        print(f"PSNR sRGB masked:        {summary['psnr_srgb_masked']:6.2f} dB  <- honest silhouette (test split)")
        print(f"PSNR linear:             {summary['psnr_linear']:6.2f} dB  (test split, mean per-image)")
        print(f"  [legacy dB-of-mean-MSE, all cams: {summary['all_psnr_srgb_db_of_mean_mse']:.2f} dB — NOT the paper metric]")
        print(f"final gaussian count:    {summary['final_n_gaussians']}")
        print(f"saved ply:               {ply_path}")
        print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
