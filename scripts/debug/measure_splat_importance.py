# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""How many splats does a trained model actually need?

Trains with the normal protocol up to ``--probe-iter``, then measures:
  * importance: the total blend weight (sum of alpha * T) each splat contributes
    over every training view. Colour is affine in sh0 (0.5 + C0 * sh0), so the
    existing backward under a unit red image gradient yields C0 * importance
    in the red colour gradient - no extra shader needed;
  * prune curves: held-out PSNR after virtually removing the least important
    k% of splats (opacity forced to ~0), without any retraining;
  * optionally (--time-steps), the training-step time before and after really
    removing the least important --time-prune-fraction of the splats.

Usage (any v2_train.py flag is accepted):
    .venv/Scripts/python.exe scripts/debug/measure_splat_importance.py \\
        --scene datasets/360_v2/bicycle --iters 3000 --probe-iter 3000 \\
        --densify-mode structure_aware --mip-splatting --seed 42 --no-viewer \\
        --out outputs/importance/bicycle_gaussian_3000.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent
for _sub in ("running", "debug", "plots"):
    _path = str(_SCRIPTS / _sub)
    if _path not in sys.path:
        sys.path.append(_path)

from trainer.backend.densify.densify import DensifyConfig
from trainer.config.config import AdamLRs, IterationContextualVariables
from trainer.contracts.codegen import gaussian_numpy_dtype
from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import build_parser
from trainer.frontend.evaluation.evaluation import eval_test_psnr_srgb

SH_C0 = 0.28209479177387814
FIXED_POINT_SCALE = 65536.0
PRUNED_LOGIT = -30.0
DEFAULT_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)


def parse_args():
    parser = build_parser()
    parser.add_argument("--probe-iter", type=int, default=0,
                        help="iteration to measure at (0 = --iters)")
    parser.add_argument("--prune-fractions", type=float, nargs="+",
                        default=list(DEFAULT_FRACTIONS))
    parser.add_argument("--out", type=str, default=None, help="result JSON path")
    parser.add_argument("--time-steps", type=int, default=0,
                        help="time this many training steps before and after removal")
    parser.add_argument("--time-prune-fraction", type=float, default=0.5)
    args = parser.parse_args()
    args.no_viewer = True
    args.profile_run = False
    args.capture_external = False
    args.preview_every = 0
    args.log_every = 0
    args.eval_every = 0
    if args.probe_iter <= 0:
        args.probe_iter = args.iters
    args.iters = max(args.iters, args.probe_iter)
    return args


def read_gaussians(pipeline) -> np.ndarray:
    n = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    raw = pipeline.buffers.gaussians.to_numpy().tobytes()
    return np.frombuffer(raw, dtype=gaussian_numpy_dtype())[:n].copy()


def write_gaussians(pipeline, gaussians: np.ndarray) -> None:
    pipeline.buffers.gaussians.copy_from_numpy(
        np.ascontiguousarray(gaussians).view(np.float32).reshape(-1))


def splat_importance(pipeline, cameras, contextual, n: int) -> np.ndarray:
    """Sum over the given views of each splat's blend weight alpha * T."""
    pipeline.clear_gradients()
    unit_red = np.zeros((pipeline.height, pipeline.width, 4), dtype=np.float32)
    unit_red[..., 0] = 1.0
    pipeline.buffers.dL_dC_texture.copy_from_numpy(unit_red)
    for camera in cameras:
        with pipeline.recording("importance"):
            pipeline.rasterize(camera, contextual)
            pipeline.record_backward(contextual)
    pipeline.runtime.wait_for_idle()
    raw = pipeline.buffers.grad_buffers["color_r"].to_numpy().tobytes()
    grad = np.frombuffer(raw, dtype=np.int64)[:n].astype(np.float64)
    return grad / FIXED_POINT_SCALE / SH_C0


def importance_stats(importance: np.ndarray, opacity_logit: np.ndarray) -> dict:
    n = importance.size
    order = np.sort(importance)[::-1]
    cumulative = np.cumsum(order) / max(order.sum(), 1e-30)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))
    bins = {"translucent(<0.1)": opacity < 0.1,
            "mid(0.1-0.9)": (opacity >= 0.1) & (opacity < 0.9),
            "opaque(>=0.9)": opacity >= 0.9}
    return {
        "splats": n,
        "zero_importance": int(np.count_nonzero(importance <= 0.0)),
        "share_holding_99pct": float(np.searchsorted(cumulative, 0.99) + 1) / n,
        "share_holding_999pct": float(np.searchsorted(cumulative, 0.999) + 1) / n,
        "percentiles": {str(q): float(np.percentile(importance, q))
                        for q in (10, 25, 50, 75, 90, 99)},
        "by_opacity": {name: {"splats": int(mask.sum()),
                              "zero_importance": int((mask & (importance <= 0)).sum()),
                              "importance_share": float(importance[mask].sum()
                                                        / max(importance.sum(), 1e-30))}
                       for name, mask in bins.items()},
    }


def prune_curve(pipeline, gaussians: np.ndarray, importance: np.ndarray,
                fractions, evaluate) -> list[dict]:
    """Held-out PSNR after zeroing the opacity of the least important splats."""
    order = np.argsort(importance, kind="stable")
    candidates = [("zero_importance", np.flatnonzero(importance <= 0.0))]
    candidates += [(f"lowest_{fraction:g}", order[:round(fraction * importance.size)])
                   for fraction in fractions]
    rows = []
    for label, pruned in candidates:
        trial = gaussians.copy()
        trial["opacity"][pruned] = PRUNED_LOGIT
        write_gaussians(pipeline, trial)
        psnr = evaluate()
        rows.append({"label": label, "pruned": int(pruned.size),
                     "kept": int(importance.size - pruned.size),
                     "importance_removed": float(importance[pruned].sum()
                                                 / max(importance.sum(), 1e-30)),
                     "psnr_srgb": psnr})
        print(f"[importance] {label:>18}: pruned {pruned.size:>9,}  "
              f"kept {importance.size - pruned.size:>9,}  psnr {psnr:.3f}", flush=True)
    write_gaussians(pipeline, gaussians)
    return rows


def remove_least_important(pipeline, gaussians: np.ndarray, importance: np.ndarray,
                           fraction: float) -> int:
    """Really drop the least important splats via the densify prune + compaction."""
    trial = gaussians.copy()
    trial["opacity"][np.argsort(importance)[:round(fraction * importance.size)]] = PRUNED_LOGIT
    write_gaussians(pipeline, trial)
    pipeline.densify_simple(DensifyConfig(), grad_thresh=float("inf"),
                            allow_scale_prune=False)
    pipeline.clear_tracking()
    pipeline.compute_3d_filter()
    pipeline.runtime.wait_for_idle()
    return int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])


def training_step_ms(app, steps: int, warmup: int = 20) -> float:
    """Wall-clock ms per normal training step, cycling over the train views."""
    pipeline = app.pipeline
    scene = app.scene_context
    limiter = app.runtime.queue_limiter
    lrs = AdamLRs()
    started = 0.0
    for step in range(warmup + steps):
        if step == warmup:
            pipeline.runtime.wait_for_idle()
            started = time.perf_counter()
        index = scene.train_indices[step % len(scene.train_indices)]
        pipeline.set_target_texture(app.target_textures[index])
        limiter.before_submit()
        pipeline.train_step(scene.colmapScene.images[index].camera, lrs=lrs)
        limiter.record(pipeline.last_submit_id)
    pipeline.runtime.wait_for_idle()
    return 1e3 * (time.perf_counter() - started) / steps


def main() -> int:
    args = parse_args()
    app = TrainingApplication(args)
    app.prepare_training_context()
    app.build_training_loop().advance_to_capture(args.probe_iter)

    pipeline = app.pipeline
    scene = app.scene_context
    images = scene.colmapScene.images
    contextual = IterationContextualVariables(
        hardness_value=float(args.extinction_hardness), sh_degree=int(args.sh_degree))

    gaussians = read_gaussians(pipeline)
    importance = splat_importance(
        pipeline, [images[i].camera for i in scene.train_indices], contextual,
        gaussians.size)
    stats = importance_stats(importance, gaussians["opacity"])
    print(f"[importance] iter {args.probe_iter}: {stats['splats']:,} splats, "
          f"{stats['zero_importance']:,} never visible, 99% of the blend weight in "
          f"{100 * stats['share_holding_99pct']:.1f}% of the splats", flush=True)

    def evaluate() -> float:
        return eval_test_psnr_srgb(pipeline, scene.colmapScene, scene.test_indices,
                                   app.targets.srgb)[0]

    curve = prune_curve(pipeline, gaussians, importance, args.prune_fractions, evaluate)
    result = {"scene": Path(args.scene).name, "probe_iter": args.probe_iter,
              "splat_kernel": args.splat_kernel,
              "splat_profile_power": args.splat_profile_power,
              "train_views": len(scene.train_indices),
              "test_views": len(scene.test_indices),
              "stats": stats, "prune_curve": curve}
    if args.time_steps > 0:
        full_ms = training_step_ms(app, args.time_steps)
        kept = remove_least_important(pipeline, gaussians, importance,
                                      args.time_prune_fraction)
        pruned_ms = training_step_ms(app, args.time_steps)
        result["step_timing"] = {"steps": args.time_steps, "full_splats": int(gaussians.size),
                                 "full_ms": full_ms, "kept_splats": kept,
                                 "pruned_ms": pruned_ms}
        print(f"[importance] training step: {gaussians.size:,} splats {full_ms:.2f} ms -> "
              f"{kept:,} splats {pruned_ms:.2f} ms "
              f"({100 * (1 - pruned_ms / full_ms):.1f}% faster)", flush=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"[importance] wrote {out}")
    pipeline.runtime.wait_for_idle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
