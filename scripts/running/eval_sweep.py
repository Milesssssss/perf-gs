"""Sweep ONE training flag across the Mip-NeRF 360 scenes; collect PSNR + time.

Generalizes ``eval_mipnerf_sweep.py``, which could only sweep ``--sh-degree``.
Here the swept axis is named on the command line, so any knob the trainer accepts
can be swept without editing a script:

    # the STBN pixel-density sweep
    .venv/Scripts/python.exe scripts/running/eval_sweep.py \
        --axis stbn-pixel-fraction --values 1,0.75,0.5,0.25,0.1

    # one scene, quick
    .venv/Scripts/python.exe scripts/running/eval_sweep.py \
        --axis stbn-pixel-fraction --values 1,0.25 --scenes bicycle --iters 500

    # anything else, same machinery
    .venv/Scripts/python.exe scripts/running/eval_sweep.py --axis sh-degree --values 0,3
    .venv/Scripts/python.exe scripts/running/eval_sweep.py \
        --axis extinction-hardness --values 1,2,4 --extra --lambda-ssim 0.0

Each cell is a fresh subprocess, so every run gets its own GPU device and a clean
densification state — the isolation the published numbers were produced under.
Only two numbers are collected, both from the run's ``test_psnr.json``:
held-out TEST-set PSNR (mean per-image sRGB, the paper metric) and
``train_wall_s`` (the timed loop, with eval excluded), plus the final gaussian
count because it is free and explains most PSNR differences.

Idempotent: a cell whose ``test_psnr.json`` already exists is reused unless
``--force``, so an interrupted sweep resumes for free. Results are written after
every cell, and the plots + CSV are regenerated each time, so a sweep is readable
while it is still running.

RUN IT ON AN IDLE GPU. ``train_wall_s`` is wall-clock, so anything else using the
device — a profiler attached (Nsight, PIX), another trainer, even a live viewer —
inflates it, and the PSNR column will look fine while every timing is quietly
wrong. Re-run with ``--force`` if a sweep shared the GPU; there is no way to
salvage the numbers after the fact.

Outputs, under ``outputs/eval_sweep/<name>/``:
    results.json          one record per cell
    sweep_summary.csv     the table view (scene x value, plus per-value averages)
    psnr_sweep.png        held-out PSNR
    timing_sweep.png      train wall time
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# Canonical Mip-NeRF 360 order: outdoor scenes, then indoor. flowers/treehill are
# the "extra" scenes and live in a different folder (see _scene_dir) — the older
# sweep script hardcoded 360_v2 and silently skipped them.
DEFAULT_SCENES = ["bicycle", "flowers", "garden", "stump", "treehill",
                  "bonsai", "counter", "kitchen", "room"]
DATASET_ROOTS = ["360_v2", "360_extra_scenes"]

# The eval config the published numbers use: quarter res (the trainer's default
# --downsample 4), 3k iters, structure-aware densify, mip-splatting, the paper
# holdout protocol. --evaluation is required because it vetoes --final-test-psnr.
BASE_ARGS = [
    "--densify-mode", "structure_aware",
    "--mip-splatting",
    "--seed", "42",
    "--no-viewer",
    "--log-every", "0",
    "--evaluation",
    "--final-test-psnr",
]


def _scene_dir(scene: str) -> Path | None:
    for root in DATASET_ROOTS:
        candidate = REPO / "datasets" / root / scene
        if candidate.is_dir():
            return candidate
    return None


def _tag(value: str) -> str:
    """Filesystem-safe form of a swept value ('0.75' -> '0p75')."""
    return value.replace(".", "p").replace("-", "neg").replace("/", "_")


def run_cell(scene: str, axis: str, value: str, *, iters: int, force: bool,
             extra: list[str], sweep_name: str) -> dict | None:
    scene_dir = _scene_dir(scene)
    if scene_dir is None:
        print(f"[sweep] SKIP {scene}: not found under "
              f"{', '.join('datasets/' + r for r in DATASET_ROOTS)}", flush=True)
        return None

    run_name = f"sweep_{sweep_name}_{scene}_{_tag(value)}"
    result_json = REPO / "outputs" / "train" / run_name / "test_psnr.json"

    if result_json.exists() and not force:
        print(f"[sweep] cached {scene} {axis}={value}", flush=True)
    else:
        cmd = [
            sys.executable, str(REPO / "scripts" / "running" / "v2_train.py"),
            "--scene", str(scene_dir),
            "--iters", str(iters),
            *BASE_ARGS,
            f"--{axis}", value,
            *extra,
            "--run-name", run_name,
        ]
        print(f"[sweep] TRAIN {scene} {axis}={value} ...", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO))
        if proc.returncode != 0 or not result_json.exists():
            print(f"[sweep] FAILED {scene} {axis}={value} (rc={proc.returncode})",
                  flush=True)
            return None

    data = json.loads(result_json.read_text())
    return {
        "scene": scene,
        "axis": axis,
        "value": value,
        "psnr_srgb": data.get("psnr_srgb"),
        "train_wall_s": data.get("train_wall_s"),
        "n_gaussians": data.get("n_gaussians"),
        "iters": data.get("iters"),
    }


def _publish(out_dir: Path, results: list[dict], axis: str, iters: int,
             plot: bool) -> None:
    """Persist results and refresh the CSV + plots. Called after every cell so a
    long sweep can be read (and a crashed one salvaged) mid-flight."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    if not plot or not results:
        return
    cmd = [sys.executable, str(REPO / "scripts" / "plots" / "plot_sweep.py"),
           "--results", str(out_dir / "results.json"),
           "--axis-label", axis, "--iters", str(iters)]
    # Quiet on success (this runs after every cell), but never SILENT on failure:
    # a broken plotter used to swallow itself here and the sweep looked healthy
    # right up until there were no charts at the end.
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
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
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is passed through to the "
                         "trainer verbatim (must be LAST)")
    args = ap.parse_args()

    axis = args.axis.lstrip("-")
    values = [v.strip() for v in args.values.split(",") if v.strip()]
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    sweep_name = (args.name or axis).replace("-", "_")
    out_dir = REPO / "outputs" / "eval_sweep" / sweep_name

    # Fail fast on a mistyped axis: the trainer's parser is the authority, and a
    # bad flag would otherwise surface as N identical subprocess failures.
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
                              sweep_name=sweep_name)
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
