# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Deterministic per-phase (fwd/bwd/adam) GPU-ms benchmark at a pinned iter.

Trains normally to --capture-iter, then times the frozen train_step (pinned camera, LRs 0) without Nsight.
Usage: python scripts/running/bench_iter.py --capture-iter 1560 --densify-mode structure_aware --sh-degree 3 --mip-splatting --seed 42
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
for _sub in ("running", "debug", "plots"):
    _p = str(_SCRIPTS / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

from trainer.frontend.app import TrainingApplication
from trainer.frontend.cli import build_parser


def _build_args():
    parser = build_parser()
    # Benchmark-only controls layered on top of the frozen CLI contract.
    parser.add_argument("--bench-warmup", type=int, default=16,
                        help="untimed frozen steps before timing (steady state)")
    parser.add_argument("--bench-iters", type=int, default=64,
                        help="timed frozen steps to aggregate")
    parser.add_argument("--bench-json", type=str, default=None,
                        help="write the timing summary JSON to this path")
    args = parser.parse_args()

    # Force a headless, non-profiling run; --capture-iter is the pin point, defaulting to the launch-config capture.
    args.no_viewer = True
    args.profile_run = False
    args.capture_external = False
    args.preview_every = 0
    args.log_every = 0
    args.eval_every = 0
    args.psnr_curve = False
    if args.capture_iter <= 0:
        args.capture_iter = 1560
    # The pin point must be reachable within the run.
    if args.iters < args.capture_iter:
        args.iters = args.capture_iter
    return args


def main() -> int:
    args = _build_args()
    capture_step = args.capture_iter

    app = TrainingApplication(args)
    app.prepare_training_context()
    loop = app.build_training_loop()

    print(f"[bench] advancing to iter {capture_step} "
          f"(scene={Path(args.scene).name}, densify={args.densify_mode}, "
          f"sh={args.sh_degree}, mip={bool(args.mip_splatting)}, seed={args.seed})...")
    context = loop.advance_to_capture(capture_step)

    pipeline = app._pipeline
    try:
        n_gaussians = int(
            pipeline.buffers.primitive.active_count.to_numpy().view("uint32")[0])
    except Exception as exc:  # informational readback only; never abort timing
        print(f"[bench] warning: n_gaussians readback failed ({exc}); continuing")
        n_gaussians = -1
    print(f"[bench] frozen state @ iter {capture_step}: "
          f"cam_idx={context.camera_index}  n_gaussians={n_gaussians}")

    # Fixed-seed runs are deterministic, so a bit-identical optimization must reproduce this fingerprint exactly.
    import numpy as _np
    pipeline.render(context.camera_entry.camera)
    img = pipeline.read_output_image()[..., :3]
    fp = {
        "n_gaussians": n_gaussians,
        "img_sum": float(_np.sum(img.astype(_np.float64))),
        "img_mean": float(_np.mean(img)),
        "img_absmax": float(_np.max(_np.abs(img))),
        "img_finite": bool(_np.all(_np.isfinite(img))),
    }
    print(f"[bench] fingerprint: n_g={fp['n_gaussians']} "
          f"img_sum={fp['img_sum']:.6f} img_mean={fp['img_mean']:.8f} "
          f"absmax={fp['img_absmax']:.6f} finite={fp['img_finite']}")

    result = pipeline.benchmark_frozen(
        context.camera_entry.camera,
        warmup=args.bench_warmup,
        iters=args.bench_iters,
    )
    result["iter"] = capture_step
    result["n_gaussians"] = n_gaussians
    result["scene"] = Path(args.scene).name
    result["cam_idx"] = context.camera_index
    result["fingerprint"] = fp

    markers = result["markers"]
    top = result["top_phases"]
    print()
    print(f"=== TOP-LEVEL phases @ iter {capture_step} "
          f"(median of {result['iters']} frozen steps, {n_gaussians} gaussians) ===")
    print(f"{'phase':<22}{'median':>10}{'mean':>10}{'min':>10}{'max':>10}   (ms)")
    for label in top:
        if label not in markers:
            continue
        s = markers[label]
        print(f"{label:<22}{s['median']:>10.4f}{s['mean']:>10.4f}"
              f"{s['min']:>10.4f}{s['max']:>10.4f}")
    print(f"{'TOTAL(median)':<22}{result['total_ms']:>10.4f}")

    # Nested markers are per-kernel, so this ranking pinpoints a hotspot to its dispatch.
    print()
    print(f"=== ALL markers by median cost @ iter {capture_step} ===")
    print(f"{'marker':<26}{'median':>10}{'mean':>10}{'max':>10}   (ms)")
    ranked = sorted(markers.items(), key=lambda kv: kv[1]["median"], reverse=True)
    for label, s in ranked:
        print(f"{label:<26}{s['median']:>10.4f}{s['mean']:>10.4f}{s['max']:>10.4f}")
    print()
    print(f"[bench] fwd={result['forward_ms']:.4f}ms  "
          f"bwd={result['backward_ms']:.4f}ms  "
          f"adam={result['adam_ms']:.4f}ms  (median)")

    if args.bench_json:
        out = Path(args.bench_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"[bench] wrote {out}")

    pipeline.runtime.wait_for_idle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
