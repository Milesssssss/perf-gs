# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Speed/quality frontier over the knobs that trade PSNR for training time.

Cached per-run results go to ``outputs/eval_sweep/<name>/``; run on an idle GPU since the time axis is wall-clock.
Usage: python scripts/running/eval_pareto.py --configs baseline,bf16 --scenes room
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from eval_sweep import REPO, train_run

# README protocol at SH0, with SSIM/LPIPS in the run report.
COMMON = ["--sh-degree", "0", "--eval-every", "0", "--preview-every", "0",
          "--no-write-ground-truths", "--run-report", "--eval-lpips"]

BF16 = ["--adam-moments", "bf16"]
PRUNE = ["--prune-blend-weight", "0.2", "--prune-blend-from", "1900"]
PRUNE_EARLY = ["--prune-blend-weight", "0.3", "--prune-blend-from", "500"]
OPAQUE_CORE = ["--splat-kernel", "opaque_core", "--splat-profile-power", "3",
               "--splat-profile-onset", "2"]


def stbn(fraction: float) -> list[str]:
    return ["--stbn-pixel-fraction", str(fraction)]


# tag -> (chart label, trainer flags)
CONFIGS: dict[str, tuple[str, list[str]]] = {
    "baseline": ("baseline", []),
    "bf16": ("bf16", BF16),
    "stbn50": ("STBN 50%", stbn(0.5)),
    "stbn25": ("STBN 25%", stbn(0.25)),
    "stbn10": ("STBN 10%", stbn(0.1)),
    "prune": ("prune", PRUNE),
    "prune_early": ("prune early", PRUNE_EARLY),
    "opaque": ("opaque core", OPAQUE_CORE),
    "bf16_prune": ("bf16+prune", BF16 + PRUNE),
    "bf16_prune_stbn50": ("bf16+prune+STBN 50%", BF16 + PRUNE + stbn(0.5)),
    "bf16_prune_stbn25": ("bf16+prune+STBN 25%", BF16 + PRUNE + stbn(0.25)),
    "bf16_prune_opaque": ("bf16+prune+opaque", BF16 + PRUNE + OPAQUE_CORE),
    "fastest": ("bf16+prune early+STBN 50%+opaque",
                BF16 + PRUNE_EARLY + stbn(0.5) + OPAQUE_CORE),
}

# Room's PSNR moves ~0.1 dB between seeds, bicycle's ~0.005 dB.
SEEDS = {"bicycle": [42], "room": [42, 1, 2]}


def _quality(run_name: str) -> dict:
    """SSIM / LPIPS from the run report (absent when the run had no report)."""
    report = REPO / "outputs" / "train" / run_name / "run_report.json"
    if not report.exists():
        return {}
    quality = json.loads(report.read_text()).get("quality", {})
    return {"ssim_srgb": quality.get("ssim_srgb"), "lpips": quality.get("lpips")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--configs", default=",".join(CONFIGS),
                    help="comma-separated config tags (default: all)")
    ap.add_argument("--scenes", default=",".join(SEEDS))
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seeds for every scene "
                         "(default: 3 for room, 1 for bicycle)")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--name", default="pareto")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tags = [t.strip() for t in args.configs.split(",") if t.strip()]
    unknown = [t for t in tags if t not in CONFIGS]
    if unknown:
        print(f"error: unknown configs {unknown}; choose from {list(CONFIGS)}",
              file=sys.stderr)
        return 2
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    out_dir = REPO / "outputs" / "eval_sweep" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for tag in tags:
        label, flags = CONFIGS[tag]
        for scene in scenes:
            seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
                     else SEEDS.get(scene, [42]))
            for seed in seeds:
                run_name = f"{args.name}_{tag}_{scene}_s{seed}"
                record = train_run(scene, run_name,
                                   [*COMMON, *flags, "--seed", str(seed)],
                                   iters=args.iters, force=args.force)
                if record is None:
                    continue
                results.append({"config": tag, "label": label, "seed": seed,
                                "flags": " ".join(flags), **record,
                                **_quality(run_name)})
                print(f"[pareto] {label:<36} {scene:<8} seed {seed}: "
                      f"{record['psnr_srgb']:.3f} dB {record['train_wall_s']:.1f} s",
                      flush=True)
                (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    proc = subprocess.run([sys.executable, str(REPO / "scripts" / "plots" / "plot_pareto.py"),
                           "--results", str(out_dir / "results.json")],
                          cwd=str(REPO), check=False)
    print(f"[pareto] {len(results)} runs -> {out_dir}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
