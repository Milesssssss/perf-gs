# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Train every Mip-NeRF 360 scene at SH degree 0 and 3, collect PSNR + wall time.

Each run is a fresh subprocess aggregated into outputs/eval_sweep/results.json; cached runs are skipped unless --force.
Usage: python scripts/running/eval_mipnerf_sweep.py --scenes bicycle,garden --sh 0,3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
# Public Mip-NeRF 360 "released 7" (flowers/treehill are extra), outdoor-then-indoor order.
DEFAULT_SCENES = ["bicycle", "garden", "stump", "bonsai", "counter", "kitchen", "room"]
OUT_ROOT = REPO / "outputs" / "eval_sweep"


def run_one(scene: str, sh: int, iters: int, force: bool) -> dict | None:
    scene_dir = REPO / "datasets" / "360_v2" / scene
    if not scene_dir.is_dir():
        print(f"[sweep] SKIP {scene} sh{sh}: {scene_dir} not found", flush=True)
        return None
    run_name = f"sweep_{scene}_sh{sh}"
    run_dir = REPO / "outputs" / "train" / run_name
    result_json = run_dir / "test_psnr.json"

    if result_json.exists() and not force:
        print(f"[sweep] cached {scene} sh{sh}", flush=True)
    else:
        cmd = [
            sys.executable, str(REPO / "scripts" / "running" / "v2_train.py"),
            "--scene", str(scene_dir),
            "--iters", str(iters),
            "--densify-mode", "structure_aware",
            "--mip-splatting",
            "--sh-degree", str(sh),
            "--seed", "42",
            "--no-viewer",
            "--final-test-psnr",
            "--run-name", run_name,
        ]
        print(f"[sweep] TRAIN {scene} sh{sh} ...", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO))
        if proc.returncode != 0 or not result_json.exists():
            print(f"[sweep] FAILED {scene} sh{sh} (rc={proc.returncode})", flush=True)
            return None

    data = json.loads(result_json.read_text())
    return {
        "scene": scene,
        "sh": sh,
        "psnr_srgb": data.get("psnr_srgb"),
        "n_gaussians": data.get("n_gaussians"),
        "train_wall_s": data.get("train_wall_s"),
        "iters": data.get("iters"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES),
                    help="comma-separated scene names")
    ap.add_argument("--sh", default="0,3", help="comma-separated SH degrees")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--force", action="store_true", help="retrain even if cached")
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    shs = [int(s) for s in args.sh.split(",") if s.strip()]

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for sh in shs:
        for scene in scenes:
            r = run_one(scene, sh, args.iters, args.force)
            if r is not None:
                results.append(r)
                print(f"[sweep]   {scene} sh{sh}: PSNR={r['psnr_srgb']:.3f} dB  "
                      f"n_g={r['n_gaussians']}  {r['train_wall_s']:.1f}s", flush=True)
                # Persist incrementally so a crash mid-sweep keeps partial results.
                (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))

    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[sweep] wrote {OUT_ROOT / 'results.json'}  ({len(results)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
