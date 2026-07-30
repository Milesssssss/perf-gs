"""Precompute and cache the multi-scale structure-tensor fields for a scene.

Structure-aware densification (``v2_train.py --densify-mode structure_aware``) needs a
per-view structure-tensor field S̄, and computing them takes several seconds
per image. This script does that once and writes them to a cache under
``<scene>/structure_cache/`` so training runs load them instantly instead of
recomputing every launch.

The cache key encodes the scene resolution and the analysis parameters (see
``structure_analysis.cache_path``), so a cache built here is only reused by a
training run with matching resolution / params. v2_train falls back to
computing on the fly (and warns) if no matching cache is found.

Usage:
    .venv/Scripts/python.exe scripts/preprocess_structure.py \\
        --scene datasets/synthetic/suzanne_blender --downsample 4

Pass the SAME --scene / --downsample / --width / --height you will train with.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trainer.data.colmap import load_colmap, load_colmap_image
from trainer.backend.densify.structure_analysis import (
    GAMMA,
    NUM_LEVELS,
    RHO_FACTOR,
    SIGMA_BASE,
    cache_path,
    compute_structure_tensor_field,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=str, required=True,
                    help="scene dir (same value you pass to v2_train.py)")
    ap.add_argument("--downsample", type=int, default=4,
                    help="downsample factor (must match the training run)")
    ap.add_argument("--width", type=int, default=None, help="force width (overrides --downsample)")
    ap.add_argument("--height", type=int, default=None, help="force height (overrides --downsample)")
    # Structure-analysis params — defaults match structure_analysis.py; only
    # override if you also override them at train time (there are no train
    # flags for these yet, so the defaults are the norm).
    ap.add_argument("--num-levels", type=int, default=NUM_LEVELS)
    ap.add_argument("--sigma-base", type=float, default=SIGMA_BASE)
    ap.add_argument("--rho-factor", type=float, default=RHO_FACTOR)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--force", action="store_true",
                    help="recompute and overwrite even if a matching cache exists")
    args = ap.parse_args()

    scene = load_colmap(args.scene, width=args.width, height=args.height,
                        downsample=args.downsample)
    n_cams = len(scene.images)
    W, H = scene.images[0].camera.image_size
    print(f"[preprocess] scene={args.scene}  {n_cams} views  {W}x{H}")

    out_path = cache_path(args.scene, W, H, num_levels=args.num_levels,
                          sigma_base=args.sigma_base, rho_factor=args.rho_factor,
                          gamma=args.gamma)
    if out_path.exists() and not args.force:
        print(f"[preprocess] cache already exists: {out_path}\n"
              f"             pass --force to recompute.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fields = np.empty((n_cams, H, W, 3), dtype=np.float32)
    names = []
    t0 = time.time()
    for i, entry in enumerate(scene.images):
        srgb = load_colmap_image(entry, scene.image_dir, W, H)
        # SADGS computes the structure tensor on the RAW sRGB image
        # (train.py:184 feeds cam.original_image straight in), NOT on linearized
        # RGB — match that so the cached field agrees with the reference.
        fields[i] = compute_structure_tensor_field(
            srgb[..., :3], num_levels=args.num_levels, sigma_base=args.sigma_base,
            rho_factor=args.rho_factor, gamma=args.gamma,
        )
        names.append(getattr(entry, "name", f"cam_{i:04d}"))
        print(f"  [{i + 1}/{n_cams}] {names[-1]}  ({time.time() - t0:.1f}s elapsed)",
              end="\r", flush=True)
    dt = time.time() - t0

    # Store the fields plus the per-view manifest and params, so a loader can
    # verify the cache matches its scene (count + order) before trusting it.
    np.savez(
        out_path,
        fields=fields,
        names=np.array(names),
        width=W, height=H, n_cams=n_cams,
        num_levels=args.num_levels, sigma_base=args.sigma_base,
        rho_factor=args.rho_factor, gamma=args.gamma,
    )
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n[preprocess] wrote {out_path}  ({size_mb:.1f} MB, {dt:.1f}s "
          f"for {n_cams} views)")


if __name__ == "__main__":
    main()
