# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Headless GPU Trace capture via ngfx.exe that auto-saves the .ngfx-gputrace (self-injection cannot).

The trainer runs in --capture-external mode, so every traced frame is a frozen, bit-identical workload.
Usage: python scripts/running/capture_gpu_trace.py --project capture.ngfx-proj --scene datasets/360_v2/bicycle --capture-iter 560 -- --densify-mode structure_aware --seed 42
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from nsight_tools import find_nsight_tool, open_in_nsight

# Steady state starts after capture_iter + 7 presents (8 warm-up frames); start the trace a bit past that.
_WARMUP_MARGIN = 15


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Headless GPU Trace capture with auto-save (ngfx.exe).")
    ap.add_argument("--scene", default=None,
                    help="Scene/dataset path passed to v2_train.py. If omitted, "
                         "v2_train.py uses its own default scene.")
    ap.add_argument("--capture-iter", type=int, default=560,
                    help="Iteration whose (frozen) workload is traced.")
    ap.add_argument("--project", required=True,
                    help="Nsight Graphics project (.ngfx-proj) with the capture settings.")
    ap.add_argument("--output-dir", type=str, default="nsight/gpu_trace/out",
                    help="Folder where ngfx.exe writes the .ngfx-gputrace. A "
                         "timestamped subfolder is created inside it.")
    ap.add_argument("--max-duration-ms", type=int, default=1000,
                    help="GPU Trace collection window (ms).")
    ap.add_argument("--limit-to-frames", type=int, default=0,
                    help="Optional hard cap on traced frames (0 = duration only).")
    ap.add_argument("--capture-frames", type=int, default=0,
                    help="Frozen frames the trainer presents (0 = auto: the "
                         "frames before the trace starts + 1500 margin).")
    ap.add_argument("--python", type=str, default=sys.executable,
                    help="Python interpreter to run v2_train.py with "
                         "(default: this venv's python).")
    ap.add_argument("--multi-pass-metrics",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Collect extra hardware counters over MULTIPLE frames "
                         "(our frozen loop presents bit-identical frames, so "
                         "this replays cleanly). Required for the Trace "
                         "Analysis view's aggregated-frame bottleneck / "
                         "expert-system advice. Pass --no-multi-pass-metrics "
                         "for a faster single-pass timeline-only trace.")
    ap.add_argument("--architecture", type=str, default="Ada",
                    help="GPU architecture the --metric-set-name applies to "
                         "(default: Ada = RTX 40-series). Set to your GPU's "
                         "arch if different, e.g. 'Ampere GA10x' (RTX "
                         "30-series), 'Blackwell GB20x'.")
    ap.add_argument("--metric-set-name", type=str, default="Throughput Metrics",
                    help="GPU Trace metric set to collect (case sensitive). "
                         "'Throughput Metrics' feeds the Trace Analysis "
                         "bottleneck breakdown. Empty string keeps the "
                         "project's own metric set.")
    ap.add_argument("--open", action=argparse.BooleanOptionalAction, default=True,
                    help="Open the saved trace in Nsight Graphics when done.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the ngfx.exe command and exit.")
    ap.add_argument("trainer_args", nargs=argparse.REMAINDER,
                    help="Args after `--` forwarded verbatim to v2_train.py.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent.parent
    ngfx = find_nsight_tool("ngfx.exe", "NGFX_EXE")
    if ngfx is None:
        print("[capture] ngfx.exe not found. Set NGFX_EXE or install Nsight "
              "Graphics.", file=sys.stderr)
        return 2

    project = (repo / args.project).resolve()
    if not project.is_file():
        print(f"[capture] project file not found: {project}", file=sys.stderr)
        return 2

    start_after = args.capture_iter + _WARMUP_MARGIN
    # start_after counts every present; capture_iter + 7 of them come before the frozen loop.
    frozen_before_trace = start_after - (args.capture_iter + 7)
    capture_frames = args.capture_frames or (frozen_before_trace + 1500)
    if capture_frames <= frozen_before_trace + args.limit_to_frames:
        print("[capture] WARNING: --capture-frames may be too small; the "
              "trainer could exit before ngfx.exe finishes the trace.",
              file=sys.stderr)

    out_root = (repo / args.output_dir).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_name = (Path(args.scene).name if args.scene else "default") or "scene"
    out_dir = out_root / f"{stamp}_{scene_name}_it{args.capture_iter}"
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Strip a leading "--" separator argparse leaves in REMAINDER.
    fwd = list(args.trainer_args)
    if fwd and fwd[0] == "--":
        fwd = fwd[1:]

    # ngfx --args takes the trainer command line as a single string.
    trainer = Path(__file__).resolve().parent / "v2_train.py"
    app_argv = [
        str(trainer),
        "--capture-external",
        "--capture-iter", str(args.capture_iter),
        "--capture-frames", str(capture_frames),
        *(["--scene", str(args.scene)] if args.scene else []),
        *fwd,
    ]
    app_args_str = subprocess.list2cmdline(app_argv)

    cmd = [
        str(ngfx),
        "--activity", "GPU Trace Profiler",
        "--project", str(project),
        "--output-dir", str(out_dir),
        "--exe", str(args.python),
        "--dir", str(repo),
        "--args", app_args_str,
        "--start-after-frames", str(start_after),
        "--max-duration-ms", str(args.max_duration_ms),
        "--auto-export",
        "--no-timeout",
        # ngfx.exe drops this project setting on CLI launch (no Slang source correlation without it).
        "--real-time-shader-profiler",
    ]
    # The metric set must be paired with --architecture; an empty name keeps the project's set.
    if args.metric_set_name:
        cmd += ["--architecture", args.architecture,
                "--metric-set-name", args.metric_set_name]
    if args.multi_pass_metrics:
        cmd += ["--multi-pass-metrics"]
    if args.limit_to_frames > 0:
        cmd += ["--limit-to-frames", str(args.limit_to_frames)]

    print("[capture] ngfx.exe   :", ngfx)
    print("[capture] project    :", project)
    print("[capture] output dir :", out_dir)
    print(f"[capture] trace start: after {start_after} presents "
          f"(capture-iter {args.capture_iter} + {_WARMUP_MARGIN} margin)")
    print(f"[capture] duration   : {args.max_duration_ms} ms"
          + (f", <= {args.limit_to_frames} frames" if args.limit_to_frames else ""))
    print("[capture] metric set : "
          + (f"{args.architecture!r} / {args.metric_set_name!r}"
             if args.metric_set_name else "(project default)")
          + (", multi-pass" if args.multi_pass_metrics else ", single-pass"))
    print(f"[capture] frozen pres: {capture_frames}")
    print("[capture] app args   :", app_args_str)
    print("[capture] full cmd   :", subprocess.list2cmdline(cmd))

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=str(repo))

    traces = sorted(out_dir.glob("*.ngfx-gputrace"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if traces:
        print(f"[capture] DONE. Trace saved: {traces[0]}")
        if args.open:
            open_in_nsight(traces[0], ngfx)
            print("[capture] Opened in Nsight Graphics.")
        else:
            print("[capture] Open it in standalone Nsight Graphics "
                  "(File > Open) — no live process needed.")
    else:
        print(f"[capture] ngfx.exe exited ({proc.returncode}) but no "
              f".ngfx-gputrace was written to {out_dir}.", file=sys.stderr)
        print("[capture] Check the console above for ngfx errors (metric-set / "
              "permissions / device-lost).", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
