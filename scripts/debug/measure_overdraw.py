"""Measure per-tile binning pressure and per-pixel overlap on a DENSIFIED
geometry, to see the alpha-skip's effect where it matters (late training),
not at init.

Loads a trained .ply, uploads it, renders a few held-out cameras, and reports:
  - tile_count  = tile_end - tile_start  (splats BINNED per tile; skip-independent)
  - overlap     = gaussian_count_per_pixel (splats that CONTRIBUTED >= 1/255,
                  post alpha-skip + early-break — the honest per-pixel overlap)
The ratio tile_count/overlap shows how much the skip + T-early-break cut the
per-pixel walk vs the raw binned count.

Usage:
  python scripts/measure_overdraw.py --ply outputs/train/<run>/output.ply \
      --scene datasets/360_v2/bicycle --downsample 4 [--n-cams 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trainer.data.colmap import load_colmap
from trainer.data.dataset import load_ply
from trainer.backend.compute.device import create_device
from trainer.config.config import RasterConfig
from trainer.backend.render.pipeline import Pipeline


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--scene", default="datasets/360_v2/bicycle")
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--n-cams", type=int, default=5, help="how many test cams to sample")
    ap.add_argument("--stride", type=int, default=8, help="test-split stride (eval holdout)")
    ap.add_argument("--backend", default="vulkan")
    args = ap.parse_args()

    scene = load_colmap(args.scene, downsample=args.downsample)
    W, H = scene.images[0].camera.image_size
    print(f"scene: {len(scene.images)} cams @ {W}x{H}")

    gauss = load_ply(args.ply)
    n = gauss["means"].shape[0]
    print(f"ply: {n} splats  (opacity: mean={gauss['opacities'].mean():.3f} "
          f"p10={_pct(gauss['opacities'],10):.4f} p50={_pct(gauss['opacities'],50):.3f})")

    device = create_device(backend=args.backend)
    pipeline = Pipeline(
        device=device, width=W, height=H,
        max_splats=max(n + 1, 8_000_000),
        raster_config=RasterConfig(cluster_cull=False),  # raw binning, no cull
    )
    pipeline.set_gaussians_from_numpy(**gauss)

    # Test split = every stride-th camera (same as eval holdout).
    test_idx = [i for i in range(len(scene.images)) if i % args.stride == 0]
    test_idx = test_idx[: args.n_cams]

    tile_all, ov_all = [], []
    ratio_rows = []
    for cam_i in test_idx:
        pipeline.render(scene.images[cam_i].camera)
        tile_counts, overlap, ntx, nty = pipeline.read_bin_debug()
        tile_all.append(tile_counts.astype(np.int64))
        ov_all.append(overlap.astype(np.int64).ravel())
        # Per-view: how much did skip+break cut the per-pixel walk vs raw tile
        # occupancy? Compare the per-tile mean binned to the per-pixel mean
        # contributing.
        tm = float(tile_counts.mean())
        om = float(overlap.mean())
        ratio_rows.append((cam_i, tm, om, tm / max(om, 1e-9)))

    tile = np.concatenate(tile_all)
    ov = np.concatenate(ov_all)

    print("\n=== PER-TILE binned splats (tile_end - tile_start) — binning pressure ===")
    print(f"  mean={tile.mean():.1f}  p50={_pct(tile,50):.0f}  p99={_pct(tile,99):.0f}  max={tile.max()}")
    print("\n=== PER-PIXEL overlap (contributed >= 1/255, post alpha-skip + break) ===")
    print(f"  mean={ov.mean():.1f}  p50={_pct(ov,50):.0f}  p99={_pct(ov,99):.0f}  max={ov.max()}")
    print(f"\n=== tile_binned / pixel_overlap ratio (higher = skip+break cut more) ===")
    for cam_i, tm, om, r in ratio_rows:
        print(f"  cam {cam_i:3d}:  tile_mean={tm:7.1f}  overlap_mean={om:7.1f}  ratio={r:5.2f}x")
    print(f"\nINTERPRETATION:")
    print(f"  - overlap_max is the honest 'most splats actually hitting one pixel'.")
    print(f"  - if overlap << tile_count, the per-pixel walk is dominated by")
    print(f"    binned-but-non-contributing splats (skip/early-break working, but")
    print(f"    binning still pays to SORT them — footprint cap / fewer splats needed).")


if __name__ == "__main__":
    main()
