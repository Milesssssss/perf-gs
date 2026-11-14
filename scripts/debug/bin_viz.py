# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Debug visualization for binning / per-pixel splat overlap (``--debug-bin-viz``).

Overlap is the blended count before early-break: by design it undercounts geometric overlap.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _turbo_like(x: np.ndarray) -> np.ndarray:
    """Map normalized [0,1] scalars to a blue->red heat colormap (uint8 HxWx3) without matplotlib."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _legend_strip(width: int, vmax: float, title: str) -> Image.Image:
    """A ``width``-px colorbar with a title and 0 / mid / max ticks, using the heatmap's ramp."""
    from PIL import ImageDraw

    bar_h = 14
    pad = 4
    label_h = 12
    title_h = 12
    strip_h = title_h + bar_h + label_h + 2 * pad

    strip = Image.new("RGB", (width, strip_h), (0, 0, 0))
    draw = ImageDraw.Draw(strip)

    draw.text((pad, pad - 1), title, fill=(230, 230, 230))

    ramp = _turbo_like(np.linspace(0.0, 1.0, width, dtype=np.float32))  # (width, 3)
    bar = np.repeat(ramp[None, :, :], bar_h, axis=0)                    # (bar_h, width, 3)
    bar_y0 = title_h + pad
    strip.paste(Image.fromarray(bar, mode="RGB"), (0, bar_y0))

    mid = vmax / 2.0
    fmt = (lambda z: f"{int(round(z))}") if vmax >= 8 else (lambda z: f"{z:.2g}")
    ly = bar_y0 + bar_h + 1
    draw.text((pad, ly), "0", fill=(230, 230, 230))
    mid_s = fmt(mid)
    draw.text((width // 2 - 4 * len(mid_s), ly), mid_s, fill=(230, 230, 230))
    max_s = fmt(vmax)
    draw.text((width - 6 * len(max_s) - pad, ly), max_s, fill=(230, 230, 230))
    return strip


def _save_heatmap(values: np.ndarray, path: Path, legend_title: str,
                  vmax: float | None = None,
                  upscale_to: tuple[int, int] | None = None) -> float:
    """Colormap a 2D array to a PNG with a legend strip below; returns the vmax used.

    A given ``vmax`` pins a shared scale across images; otherwise each auto-scales to its own max.
    """
    v = values.astype(np.float32)
    used_vmax = float(vmax) if vmax is not None else float(v.max())
    norm = v / used_vmax if used_vmax > 0 else np.zeros_like(v)
    rgb = _turbo_like(norm)
    img = Image.fromarray(rgb, mode="RGB")
    if upscale_to is not None:
        # Nearest-neighbour so per-tile blocks stay crisp/legible.
        img = img.resize((upscale_to[1], upscale_to[0]), Image.NEAREST)

    legend = _legend_strip(img.width, used_vmax, legend_title)
    canvas = Image.new("RGB", (img.width, img.height + legend.height), (0, 0, 0))
    canvas.paste(img, (0, 0))
    canvas.paste(legend, (0, img.height))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return used_vmax


def write_bin_viz(
    *,
    cam_dir: Path,
    iter_idx: int,
    cam_idx: int,
    image_name: str,
    tile_counts: np.ndarray,          # uint[num_tiles] = end - start
    num_tiles_x: int,
    num_tiles_y: int,
    overlap: np.ndarray,              # uint[H, W] = gaussian_count_per_pixel
    jsonl_handle,
    tile_scale: float | None = None,     # fixed color-scale max for the tile map
    overlap_scale: float | None = None,  # fixed color-scale max for the overlap map
) -> dict:
    """Write the tile-count and overlap heatmaps plus one JSONL stats line for a view; returns the stats.

    Each map auto-scales to its own max (not comparable across views) unless tile_scale/overlap_scale pin it.
    """
    H, W = overlap.shape

    tile_grid = tile_counts[: num_tiles_x * num_tiles_y].reshape(num_tiles_y, num_tiles_x)
    tile_vmax = _save_heatmap(
        tile_grid,
        cam_dir / f"binviz_iter_{iter_idx:06d}_tilecount.png",
        legend_title="splats binned / tile",
        vmax=tile_scale,
        upscale_to=(num_tiles_y * 8, num_tiles_x * 8),  # 8px/tile block, legible
    )

    px_vmax = _save_heatmap(
        overlap,
        cam_dir / f"binviz_iter_{iter_idx:06d}_overlap.png",
        legend_title="splats blended / pixel",
        vmax=overlap_scale,
    )

    ov = overlap.astype(np.int64).ravel()
    max_ov = int(ov.max()) if ov.size else 0
    hist = np.bincount(ov, minlength=max_ov + 1)
    stats = {
        "iter": iter_idx,
        "cam_idx": cam_idx,
        "image_name": image_name,
        "num_tiles_x": num_tiles_x,
        "num_tiles_y": num_tiles_y,
        "tile_count_max": int(tile_grid.max()),
        "tile_count_mean": float(tile_grid.mean()),
        "tile_count_p50": int(np.percentile(tile_grid, 50)),
        "tile_count_p99": int(np.percentile(tile_grid, 99)),
        "tile_count_total_pairs": int(tile_grid.sum()),
        "overlap_max": max_ov,
        "overlap_mean": float(ov.mean()) if ov.size else 0.0,
        "overlap_p50": int(np.percentile(ov, 50)) if ov.size else 0,
        "overlap_p99": int(np.percentile(ov, 99)) if ov.size else 0,
        # Full histogram: hist[N] = number of pixels with exactly N blended splats.
        "overlap_histogram": hist.tolist(),
        "png_tilecount_vmax": tile_vmax,
        "png_overlap_vmax": px_vmax,
    }
    if jsonl_handle is not None:
        jsonl_handle.write(json.dumps(stats) + "\n")
    return stats
