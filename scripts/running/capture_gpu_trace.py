"""Headless GPU Trace capture that AUTO-SAVES the .ngfx-gputrace to disk.

Why this exists
---------------
The NGFX_Injection API (self-injection, used by --capture-iter in v2_train.py)
has NO function to persist a capture: the collected trace lives only inside the
in-process HUD and must be written out with the HUD's *Save* button. If anything
disturbs that HUD (e.g. pressing *Open* on the capture, which tears down the
injected host and takes Nsight with it), you lose the trace.

The Nsight Graphics command-line launcher (ngfx.exe) has no such limitation. It
spawns the target with Nsight injected, starts/stops the GPU Trace itself, and
AUTO-WRITES the .ngfx-gputrace into --output-dir. No HUD interaction, no Save,
no Open, nothing to crash.

How it stays deterministic
--------------------------
In --profile-run the trainer Presents exactly one frame per train_step, so frame
counts are predictable. This launcher runs v2_train.py in --capture-external
mode: it trains normally up to --capture-iter, then Presents a *frozen*,
bit-identical workload (all LRs = 0, camera pinned) for --capture-frames frames
and exits. We tell ngfx.exe to:
    * start the trace after (capture_iter + WARMUP_MARGIN) presents  -> we are
      safely inside the frozen steady state,
    * stop after --max-duration-ms (and/or --limit-to-frames),
so every sampled frame is identical and reproducible across runs.

Usage
-----
    .venv\\Scripts\\python.exe scripts/capture_gpu_trace.py \\
        --scene datasets/360_v2/bicycle \\
        --capture-iter 560 \\
        --project nsight/gpu_trace/gpu_trace_base/gpu_trace_base.ngfx-proj \\
        --output-dir nsight/gpu_trace/out \\
        -- --densify-mode structure_aware --seed 42

Everything after ``--`` is forwarded verbatim to v2_train.py.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Frames the trainer presents (main loop) before the frozen loop reaches steady
# state: iters 1..(capture_iter-1) present once each, then 8 warm-up presents in
# the capture block => capture_iter + 7. Start the trace a bit past that.
_WARMUP_MARGIN = 15


def _find_ngfx_exe() -> Path | None:
    """Locate ngfx.exe from the newest Nsight Graphics install.
    Override with the NGFX_EXE environment variable."""
    override = os.environ.get("NGFX_EXE")
    if override:
        p = Path(override)
        return p if p.is_file() else None

    nvidia_root = Path(r"C:\Program Files\NVIDIA Corporation")
    if not nvidia_root.is_dir():
        return None

    def _ver_key(p: Path) -> list[int]:
        return [int(x) for x in re.findall(r"\d+", p.name)] or [-1]

    installs = sorted(nvidia_root.glob("Nsight Graphics *"),
                      key=_ver_key, reverse=True)
    for install in installs:
        for exe in install.glob("host/*/ngfx.exe"):
            if exe.is_file():
                return exe
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Headless GPU Trace capture with auto-save (ngfx.exe).")
    ap.add_argument("--scene", default=None,
                    help="Scene/dataset path passed to v2_train.py. If omitted, "
                         "v2_train.py uses its own default scene.")
    ap.add_argument("--capture-iter", type=int, default=560,
                    help="Iteration whose (frozen) workload is traced.")
    ap.add_argument("--project", type=str,
                    default="nsight/gpu_trace/gpu_trace_base/gpu_trace_base.ngfx-proj",
                    help="Path to the .ngfx-proj (metric set + settings).")
    ap.add_argument("--output-dir", type=str, default="nsight/gpu_trace/out",
                    help="Folder where ngfx.exe writes the .ngfx-gputrace. A "
                         "timestamped subfolder is created inside it.")
    ap.add_argument("--max-duration-ms", type=int, default=1000,
                    help="GPU Trace collection window (ms).")
    ap.add_argument("--limit-to-frames", type=int, default=0,
                    help="Optional hard cap on traced frames (0 = duration only).")
    ap.add_argument("--capture-frames", type=int, default=0,
                    help="Frozen frames the trainer presents (0 = auto: "
                         "start-after + 1500 margin).")
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
    ap.add_argument("--architecture", type=str, default="Ampere GA10x",
                    help="GPU architecture the --metric-set-name applies to "
                         "(default: Ampere GA10x = RTX 30-series). Set to your "
                         "GPU's arch if different, e.g. 'Ada', 'Blackwell "
                         "GB20x'.")
    ap.add_argument("--metric-set-name", type=str, default="Throughput Metrics",
                    help="GPU Trace metric set to collect (case sensitive). "
                         "'Throughput Metrics' feeds the Trace Analysis "
                         "bottleneck breakdown. Empty string keeps the "
                         "project's own metric set.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the ngfx.exe command and exit.")
    ap.add_argument("trainer_args", nargs=argparse.REMAINDER,
                    help="Args after `--` forwarded verbatim to v2_train.py.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent.parent
    ngfx = _find_ngfx_exe()
    if ngfx is None:
        print("[capture] ngfx.exe not found. Set NGFX_EXE or install Nsight "
              "Graphics.", file=sys.stderr)
        return 2

    project = (repo / args.project).resolve()
    if not project.is_file():
        print(f"[capture] project file not found: {project}", file=sys.stderr)
        return 2

    start_after = args.capture_iter + _WARMUP_MARGIN
    capture_frames = args.capture_frames or (start_after + 1500)
    if capture_frames <= start_after + args.limit_to_frames:
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

    # Command line for the launched trainer (single string for ngfx --args).
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
    ]
    # Metric set drives which counters (and thus which Trace Analysis bottleneck
    # rules) are available; must be paired with --architecture. Empty name =
    # keep the project's set.
    if args.metric_set_name:
        cmd += ["--architecture", args.architecture,
                "--metric-set-name", args.metric_set_name]
    # Multi-pass = the aggregated-frame counters + expert-system advice in the
    # Trace Analysis view (single-pass gives only the timeline).
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
    print(f"[capture] metric set : "
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
