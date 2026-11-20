# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Sweep one training flag across the Mip-NeRF 360 scenes; collect PSNR + time.

Each cell is a cached fresh subprocess; run on an idle GPU, since train_wall_s is wall-clock.
Mip-NeRF 360 training and evaluation both use half resolution indoors and quarter resolution outdoors.
Usage: python scripts/running/eval_sweep.py --axis stbn-pixel-fraction --values 1,0.75,0.5,0.25,0.1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# Outdoor then indoor; flowers/treehill live in 360_extra_scenes (see _scene_dir).
DEFAULT_SCENES = ["bicycle", "flowers", "garden", "stump", "treehill",
                  "bonsai", "counter", "kitchen", "room"]
DATASET_ROOTS = ["360_v2", "360_extra_scenes", "tandt", "db"]


# The sweep opts into evaluation and records held-out PSNR.
BASE_ARGS = [
    "--seed", "42",
    "--no-viewer",
    "--log-every", "0",
    "--evaluation",
    "--final-test-psnr",
]


def scene_downsample(scene: str) -> int:
    if scene in ("drjohnson", "playroom"):
        return 1
    if scene in ("bonsai", "counter", "kitchen", "room", "train", "truck"):
        return 2
    return 4


def _scene_dir(scene: str) -> Path | None:
    for root in DATASET_ROOTS:
        candidate = REPO / "datasets" / root / scene
        if candidate.is_dir():
            return candidate
    return None


def _tag(value: str) -> str:
    """Filesystem-safe form of a swept value ('0.75' -> '0p75')."""
    return value.replace(".", "p").replace("-", "neg").replace("/", "_")


def _gpu_temperature() -> float | None:
    """The GPU's temperature in Celsius from nvidia-smi; None where that is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=False)
        return float(out.stdout.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def wait_for_cool_gpu(max_celsius: float, timeout_s: float = 300.0) -> None:
    """Block until the GPU is at most ``max_celsius``, so timed runs start from the same thermal state."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        temperature = _gpu_temperature()
        if temperature is None or temperature <= max_celsius:
            return
        time.sleep(5.0)
    print(f"[sweep] GPU still above {max_celsius:g} C after {timeout_s:.0f} s; "
          f"timing the next run anyway", flush=True)


def train_run(scene: str, run_name: str, flags: list[str], *, iters: int,
              force: bool, cooldown_c: float | None = None) -> dict | None:
    """Train ``scene`` with BASE_ARGS + ``flags`` in a fresh subprocess; cached unless ``force``, None on failure."""
    scene_dir = _scene_dir(scene)
    if scene_dir is None:
        print(f"[sweep] SKIP {scene}: not found under "
              f"{', '.join('datasets/' + r for r in DATASET_ROOTS)}", flush=True)
        return None

    result_json = REPO / "outputs" / "train" / run_name / "test_psnr.json"
    if result_json.exists() and not force:
        print(f"[sweep] cached {run_name}", flush=True)
    else:
        cmd = [
            sys.executable, str(REPO / "scripts" / "running" / "v2_train.py"),
            "--scene", str(scene_dir),
            "--iters", str(iters),
            # An explicit --downsample in flags comes later and wins.
            "--downsample", str(scene_downsample(scene)),
            *BASE_ARGS,
            *flags,
            "--run-name", run_name,
        ]
        if cooldown_c is not None:
            wait_for_cool_gpu(cooldown_c)
        print(f"[sweep] TRAIN {run_name} ...", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO), check=False)
        if proc.returncode != 0 or not result_json.exists():
            print(f"[sweep] FAILED {run_name} (rc={proc.returncode})", flush=True)
            return None

    data = json.loads(result_json.read_text())
    return {
        "scene": scene,
        "psnr_srgb": data.get("psnr_srgb"),
        "train_wall_s": data.get("train_wall_s"),
        "n_gaussians": data.get("n_gaussians"),
        "iters": data.get("iters"),
        **{key: data.get(key) for key in (
            "max_splats", "max_sort_pairs", "peak_sort_pairs", "sort_overflows",
            "targets_streamed", "allocated_bytes", "vram_budget_bytes", "vram_usage_bytes")},
    }


def run_cell(scene: str, axis: str, value: str, *, iters: int, force: bool,
             extra: list[str], sweep_name: str,
             cooldown_c: float | None = None) -> dict | None:
    record = train_run(scene, f"sweep_{sweep_name}_{scene}_{_tag(value)}",
                       [f"--{axis}", value, *extra], iters=iters, force=force,
                       cooldown_c=cooldown_c)
    if record is None:
        return None
    return {"scene": scene, "axis": axis, "value": value, **record}


def _publish(out_dir: Path, results: list[dict], axis: str, iters: int,
             plot: bool) -> None:
    """Persist results and refresh the CSV + plots; called after every cell so a sweep is readable mid-flight."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    if not plot or not results:
        return
    cmd = [sys.executable, str(REPO / "scripts" / "plots" / "plot_sweep.py"),
           "--results", str(out_dir / "results.json"),
           "--axis-label", axis, "--iters", str(iters)]
    # Quiet on success (runs after every cell) but never silent on failure.
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(f"[sweep] WARNING: plotting failed (rc={proc.returncode}); results "
              f"are safe in results.json\n{proc.stderr.strip()[-600:]}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep one trainer flag over the Mip-NeRF 360 scenes.")
    ap.add_argument("--axis", required=True,
                    help="the flag to sweep, WITHOUT the leading '--' "
                         "(e.g. stbn-pixel-fraction, sh-degree)")
    ap.add_argument("--values", required=True,
                    help="comma-separated values for --axis")
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES),
                    help="comma-separated scene names")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--name", default=None,
                    help="output folder name under outputs/eval_sweep "
                         "(default: the axis name)")
    ap.add_argument("--force", action="store_true",
                    help="retrain cells that already have results")
    ap.add_argument("--no-plot", action="store_true",
                    help="collect results only; skip the charts + CSV")
    ap.add_argument("--cooldown-c", type=float, default=None,
                    help="before each training run, wait (up to 5 min) until the GPU is at "
                         "most this many degrees Celsius (nvidia-smi), so wall-clock "
                         "times start from the same thermal state")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is passed through to the "
                         "trainer verbatim (must be LAST)")
    args = ap.parse_args()

    axis = args.axis.lstrip("-")
    values = [v.strip() for v in args.values.split(",") if v.strip()]
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    sweep_name = (args.name or axis).replace("-", "_")
    out_dir = REPO / "outputs" / "eval_sweep" / sweep_name

    # Fail fast on a mistyped axis rather than N identical subprocess failures.
    sys.path.insert(0, str(REPO / "src"))
    from trainer.frontend.cli import build_parser
    known = set(vars(build_parser().parse_args([])))
    if axis.replace("-", "_") not in known:
        print(f"error: --axis {axis} is not a trainer flag", file=sys.stderr)
        return 2

    print(f"[sweep] {len(scenes)} scenes x {len(values)} values = "
          f"{len(scenes) * len(values)} runs at {args.iters} iters -> {out_dir}")

    results: list[dict] = []
    for value in values:
        for scene in scenes:
            record = run_cell(scene, axis, value, iters=args.iters,
                              force=args.force, extra=list(args.extra),
                              sweep_name=sweep_name, cooldown_c=args.cooldown_c)
            if record is None:
                continue
            results.append(record)
            psnr = record["psnr_srgb"]
            wall = record["train_wall_s"]
            print(f"[sweep]   {scene} {axis}={value}: "
                  f"PSNR={psnr:.3f} dB  {wall:.1f}s  n_g={record['n_gaussians']}",
                  flush=True)
            _publish(out_dir, results, axis, args.iters, not args.no_plot)

    _publish(out_dir, results, axis, args.iters, not args.no_plot)
    print(f"\n[sweep] {len(results)} runs -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
