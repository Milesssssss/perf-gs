# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Headless single-frame Graphics Capture that auto-saves an .ngfx-capture for VRAM/resource inspection.

Uses ngfx-capture.exe on the same frozen --capture-external trainer state as capture_gpu_trace.py.
Usage: python scripts/running/capture_graphics.py --scene datasets/360_v2/bicycle --capture-iter 560 -- --densify-mode structure_aware --seed 42
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Same as capture_gpu_trace.py: lands ~8 frames into the frozen steady state (after capture_iter + 7 presents).
_WARMUP_MARGIN = 15

# A resource snapshot needs one frame (--frame-count accepts 1-600).
_FRAME_COUNT = 1


def _find_ngfx_capture_exe() -> Path | None:
    """Locate ngfx-capture.exe in the newest Nsight Graphics install; NGFX_CAPTURE_EXE overrides."""
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
        # ngfx-capture.exe rejects --capture-frame <= 1; only reachable with a tiny --capture-iter.
        print(f"[capture] --capture-frame must be > 1 (got {capture_frame}). "
              "Raise --capture-iter or pass --capture-frame.", file=sys.stderr)
        return 2

    # The frozen loop starts at present capture_iter + 8; a generous tail keeps it presenting until the target frame.
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

    # --capture-external implies --profile-run, which opens the presenting window Nsight delimits frames by.
    trainer = Path(__file__).resolve().parent / "v2_train.py"
    app_argv = [
        str(trainer),
        "--capture-external",
        "--capture-iter", str(args.capture_iter),
        "--capture-frames", str(capture_frames),
        *(["--scene", str(args.scene)] if args.scene else []),
        *fwd,
    ]

    # ngfx-capture parses "--x" tokens after --args as its own options; pass one quoted string, kept last.
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

    # Glob broadly in case the capture suffix varies by Nsight version.
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
