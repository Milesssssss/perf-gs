# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Debug visualization for binning / per-pixel splat overlap.

Gated behind `--debug-bin-viz` in v2_train; produces, per rendered eval view:
  - <cam>/binviz_iter_NNNNNN_tilecount.png : per-tile heatmap of how many
    (tile, gaussian) pairs were binned into each tile (= tile_end - tile_start).
  - <cam>/binviz_iter_NNNNNN_overlap.png   : per-pixel heatmap of how many
    splats were BLENDED at each pixel before early-break
    (= gaussian_count_per_pixel). NOTE: this is the blended count, i.e. what the
    backward iterates; pixels behind saturated opacity stop early so it can
    undercount true geometric overlap. That is intentional (it's the cost that
    matters for binning/backward work).
  - binviz.jsonl (one line per view) : per-tile count stats + a global histogram
    of per-pixel overlap ("how many pixels had N overlapping splats").

All readback is of buffers the forward already fills every frame — no extra
kernels. Pure numpy + PIL (no torch), so it runs in the trainer's .venv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def _turbo_like(x: np.ndarray) -> np.ndarray:
    """Map normalized [0,1] scalars to an RGB heat colormap (uint8 HxWx3).

    A compact polynomial approximation of a perceptually-ordered map
    (blue->cyan->green->yellow->red). Good enough for debug heatmaps without a
    matplotlib dependency.
    """
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _legend_strip(width: int, vmax: float, title: str) -> Image.Image:
    """A horizontal colorbar with a title and 0 / mid / max tick labels, using
    the SAME blue->red ramp as the heatmap. Rendered at `width` px wide so it
    sits flush under the heatmap. Uses PIL's built-in bitmap font (no font-file
    dependency)."""
    from PIL import ImageDraw

    bar_h = 14          # height of the gradient bar
    pad = 4
    label_h = 12        # room for tick labels below the bar
    title_h = 12        # room for the title above the bar
    strip_h = title_h + bar_h + label_h + 2 * pad

    strip = Image.new("RGB", (width, strip_h), (0, 0, 0))
    draw = ImageDraw.Draw(strip)

    # Title (top).
    draw.text((pad, pad - 1), title, fill=(230, 230, 230))

    # Gradient bar: left = 0 (blue), right = vmax (red). Full-width ramp.
    ramp = _turbo_like(np.linspace(0.0, 1.0, width, dtype=np.float32))  # (width, 3)
    bar = np.repeat(ramp[None, :, :], bar_h, axis=0)                    # (bar_h, width, 3)
    bar_y0 = title_h + pad
    strip.paste(Image.fromarray(bar, mode="RGB"), (0, bar_y0))

    # Tick labels under the bar: 0, mid, max (max is the auto/fixed scale used).
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
    """Colormap a 2D uint/float array to a PNG WITH a legend colorbar strip
    composited below it. Returns the vmax used (so the caller can log the
    scale). Zero -> blue, vmax -> red. If `vmax` is given it is used as the
    color scale (so multiple images share one scale and are comparable);
    otherwise each image auto-scales to its own max."""
    v = values.astype(np.float32)
    used_vmax = float(vmax) if vmax is not None else float(v.max())
    norm = v / used_vmax if used_vmax > 0 else np.zeros_like(v)
    rgb = _turbo_like(norm)
    img = Image.fromarray(rgb, mode="RGB")
    if upscale_to is not None:
        # Nearest-neighbour so per-tile blocks stay crisp/legible.
        img = img.resize((upscale_to[1], upscale_to[0]), Image.NEAREST)

    # Composite the heatmap above a legend strip of the same width.
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
    """Write the two heatmap PNGs (each with a legend colorbar) + one JSONL
    stats line for a single view. Returns the stats dict (also written to
    jsonl_handle).

    Color scale: blue=0, red=max. By default each map auto-scales to ITS OWN
    max, so colors are NOT comparable across views (the legend shows each map's
    max). Pass tile_scale/overlap_scale to pin a shared scale so colors ARE
    comparable across views/iters."""
    H, W = overlap.shape

    # --- Per-tile binned-count heatmap (upscaled to image size for overlay-ability) ---
    tile_grid = tile_counts[: num_tiles_x * num_tiles_y].reshape(num_tiles_y, num_tiles_x)
    tile_vmax = _save_heatmap(
        tile_grid,
        cam_dir / f"binviz_iter_{iter_idx:06d}_tilecount.png",
        legend_title="splats binned / tile",
        vmax=tile_scale,
        upscale_to=(num_tiles_y * 8, num_tiles_x * 8),  # 8px/tile block, legible
    )

    # --- Per-pixel overlap heatmap ---
    px_vmax = _save_heatmap(
        overlap,
        cam_dir / f"binviz_iter_{iter_idx:06d}_overlap.png",
        legend_title="splats blended / pixel",
        vmax=overlap_scale,
    )

    # --- Per-view global histogram of per-pixel overlap ---
    ov = overlap.astype(np.int64).ravel()
    max_ov = int(ov.max()) if ov.size else 0
    # bincount over [0, max_ov]; keep it bounded for the JSON.
    hist = np.bincount(ov, minlength=max_ov + 1)
    stats = {
        "iter": iter_idx,
        "cam_idx": cam_idx,
        "image_name": image_name,
        "num_tiles_x": num_tiles_x,
        "num_tiles_y": num_tiles_y,
        # Per-tile binned (tile,gaussian) pair counts.
        "tile_count_max": int(tile_grid.max()),
        "tile_count_mean": float(tile_grid.mean()),
        "tile_count_p50": int(np.percentile(tile_grid, 50)),
        "tile_count_p99": int(np.percentile(tile_grid, 99)),
        "tile_count_total_pairs": int(tile_grid.sum()),
        # Per-pixel blended-overlap distribution.
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
