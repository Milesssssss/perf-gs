"""Headless Graphics Capture that AUTO-SAVES an .ngfx-capture to disk.

Why this exists (vs. capture_gpu_trace.py)
------------------------------------------
``capture_gpu_trace.py`` drives ngfx.exe's "GPU Trace Profiler" activity: a
continuous HARDWARE-COUNTER profiler that samples many CONTINUOUSLY-PRESENTED,
bit-identical frames over a ~1s window. That is the right tool for *performance*
(bottlenecks, throughput, shader timing) and is intentionally kept as-is.

It is the WRONG tool for the "where are my ~20 GB of VRAM going" question. That
needs a *Graphics Capture*: a single-frame snapshot of API calls AND application
resources, which you open in Nsight's Graphics Debugger and inspect under
    Graphics Debugger -> Memory      (VkDeviceMemory objects, sizes, aliasing)
    Graphics Debugger -> Object Browser
A separate executable produces those:
    ngfx-capture.exe        (NOT ngfx.exe --activity "GPU Trace Profiler")

It spawns the target with Nsight injected, triggers a capture at a specific
presented frame, AUTO-WRITES the capture into --output-dir, and (with
--terminate-after-capture) kills the app afterwards. No HUD, no Save, no Open.

How it stays deterministic
--------------------------
We reuse the trainer's existing ``--capture-external`` frozen mode unchanged, so
the Graphics Capture inspects the SAME trainer state the GPU Trace does. In that
mode v2_train.py trains normally to --capture-iter, then Presents a *frozen*,
bit-identical workload (all LRs = 0, camera pinned, no densify) for
--capture-frames frames and exits. The trainer's present accounting is:

    frames 1 .. (capture_iter-1)         main loop (one Present per train_step)
    frames capture_iter .. capture_iter+7    8 warm-up frozen frames
    frames capture_iter+8 ..             the bounded frozen capture loop

so a 1-based capture target of ``capture_iter + WARMUP_MARGIN`` (the same margin
capture_gpu_trace.py uses to arm its trace) lands a few frames INTO the frozen
steady state — after warm-up, before the loop ends. We tell ngfx-capture.exe to
capture exactly that one presented frame and then terminate the trainer.

Unlike the GPU Trace path we do NOT need the 1500-frame loop: a single frame is
enough for a resource snapshot. We still present a few hundred frozen frames so
the trainer is guaranteed to still be presenting when Nsight reaches the target
frame; --terminate-after-capture stops it the instant the capture completes.

Usage
-----
    .venv\\Scripts\\python.exe scripts/running/capture_graphics.py \\
        --scene datasets/360_v2/bicycle \\
        --capture-iter 560 \\
        --output-dir nsight/graphics/out \\
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

# Kept identical to capture_gpu_trace.py so the Graphics Capture target frame
# sits in the same frozen steady state the GPU Trace arms in: main iterations
# 1..(capture_iter-1), then 8 warm-up frozen frames = capture_iter+7 presents;
# a 1-based target of capture_iter+15 is ~8 frames into the frozen loop.
_WARMUP_MARGIN = 15

# ngfx-capture.exe --frame-count is documented as [1 - 600]; a resource snapshot
# only needs one frame.
_FRAME_COUNT = 1


def _find_ngfx_capture_exe() -> Path | None:
    """Locate ngfx-capture.exe from the newest Nsight Graphics install.
    Override with the NGFX_CAPTURE_EXE environment variable."""
    override = os.environ.get("NGFX_CAPTURE_EXE")
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
        for exe in install.glob("host/*/ngfx-capture.exe"):
            if exe.is_file():
                return exe
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Headless single-frame Graphics Capture with auto-save "
                    "(ngfx-capture.exe) for VRAM / resource inspection.")
    ap.add_argument("--scene", default=None,
                    help="Scene/dataset path passed to v2_train.py. If omitted, "
                         "v2_train.py uses its own default scene.")
    ap.add_argument("--capture-iter", type=int, default=560,
                    help="Iteration whose (frozen) workload is captured.")
    ap.add_argument("--output-dir", type=str, default="nsight/graphics/out",
                    help="Folder where ngfx-capture.exe writes the capture. A "
                         "timestamped subfolder is created inside it.")
    ap.add_argument("--capture-frame", type=int, default=0,
                    help="1-based presented frame to capture (0 = auto: "
                         "capture_iter + %d, i.e. inside the frozen steady "
                         "state). Must be > 1." % _WARMUP_MARGIN)
    ap.add_argument("--capture-frames", type=int, default=0,
                    help="Frozen frames the trainer presents (0 = auto: enough "
                         "to comfortably outlast the capture-frame target). We "
                         "do NOT need the GPU-Trace-sized loop here.")
    ap.add_argument("--python", type=str, default=sys.executable,
                    help="Python interpreter to run v2_train.py with "
                         "(default: this venv's python).")
    ap.add_argument("--full-gpu-allocs",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Pass --capture-full-gpu-allocs: capture whole "
                         "VkDeviceMemory allocations (not just individual "
                         "VkBuffer/VkImage contents). Increases capture "
                         "overhead + file size but is what you want for "
                         "'where is my 20 GB going' — it exposes the full "
                         "device-memory objects the Memory view groups "
                         "resources under. Pass --no-full-gpu-allocs to skip.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the ngfx-capture.exe command and exit.")
    ap.add_argument("trainer_args", nargs=argparse.REMAINDER,
                    help="Args after `--` forwarded verbatim to v2_train.py.")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent.parent
    ngfx_capture = _find_ngfx_capture_exe()
    if ngfx_capture is None:
        print("[capture] ngfx-capture.exe not found. Set NGFX_CAPTURE_EXE or "
              "install Nsight Graphics.", file=sys.stderr)
        return 2

    capture_frame = args.capture_frame or (args.capture_iter + _WARMUP_MARGIN)
    if capture_frame <= 1:
        # ngfx-capture.exe rejects --capture-frame <= 1 ("must be greater
        # than 1"). Only reachable with a tiny/explicit --capture-iter.
        print(f"[capture] --capture-frame must be > 1 (got {capture_frame}). "
              "Raise --capture-iter or pass --capture-frame.", file=sys.stderr)
        return 2

    # Trainer must still be presenting when Nsight reaches the target frame. The
    # frozen loop begins at present (capture_iter + 8); target is capture_frame.
    # Give a generous tail so the app never exits first (terminate-after-capture
    # stops it the moment the capture lands anyway).
    frozen_needed = max(1, capture_frame - (args.capture_iter + 8))
    capture_frames = args.capture_frames or (frozen_needed + 300)
    if capture_frames < frozen_needed:
        print("[capture] WARNING: --capture-frames may be too small; the "
              "trainer could exit before ngfx-capture.exe reaches "
              f"--capture-frame {capture_frame}.", file=sys.stderr)

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

    # Reuse the trainer's frozen external-capture mode unchanged so this snapshot
    # matches the GPU Trace's trainer state. --capture-external implies
    # --profile-run (opens the presenting Vulkan window Nsight delimits by).
    trainer = Path(__file__).resolve().parent / "v2_train.py"
    app_argv = [
        str(trainer),
        "--capture-external",
        "--capture-iter", str(args.capture_iter),
        "--capture-frames", str(capture_frames),
        *(["--scene", str(args.scene)] if args.scene else []),
        *fwd,
    ]

    # ngfx-capture.exe's --args is documented as variadic ("TEXT ..."), but its
    # CLI11 parser still treats any following token beginning with "--" (e.g.
    # --capture-external) as an unknown option FOR ITSELF, not as a value:
    #   "The following arguments were not expected: --capture-external ..."
    # So we pass the whole trainer command line as ONE quoted string, exactly
    # like capture_gpu_trace.py does for ngfx.exe. Keep it last regardless.
    app_args_str = subprocess.list2cmdline(app_argv)
    cmd = [
        str(ngfx_capture),
        "--exe", str(args.python),
        "--working-dir", str(repo),
        "--output-dir", str(out_dir),
        "--capture-frame", str(capture_frame),
        "--frame-count", str(_FRAME_COUNT),
        "--terminate-after-capture",
    ]
    if args.full_gpu_allocs:
        cmd += ["--capture-full-gpu-allocs"]
    cmd += ["--args", app_args_str]

    print("[capture] ngfx-capture :", ngfx_capture)
    print("[capture] output dir   :", out_dir)
    print(f"[capture] capture frame: {capture_frame} (1-based present; "
          f"capture-iter {args.capture_iter} + {_WARMUP_MARGIN} margin)")
    print(f"[capture] frame count  : {_FRAME_COUNT}")
    print("[capture] full allocs  : "
          + ("yes (--capture-full-gpu-allocs)" if args.full_gpu_allocs
             else "no"))
    print(f"[capture] frozen pres  : {capture_frames}")
    print("[capture] app args     :", app_args_str)
    print("[capture] full cmd     :", subprocess.list2cmdline(cmd))

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=str(repo))

    # Graphics captures land as .ngfx-capture; glob broadly in case the suffix
    # varies by version, newest first.
    caps = sorted(
        (p for pat in ("*.ngfx-capture", "*.ngfx-frame", "*.nsight-gfxc")
         for p in out_dir.glob(pat)),
        key=lambda p: p.stat().st_mtime, reverse=True)
    if caps:
        print(f"[capture] DONE. Capture saved: {caps[0]}")
        print("[capture] Open it in Nsight Graphics, then Graphics Debugger -> "
              "Memory to inspect VkDeviceMemory sizes / aliasing.")
    else:
        print(f"[capture] ngfx-capture.exe exited ({proc.returncode}) but no "
              f"capture file was written to {out_dir}.", file=sys.stderr)
        print("[capture] Check the console above for errors (bad launch args / "
              "frame never reached / device-lost). Try --dry-run to inspect "
              "the command.", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
