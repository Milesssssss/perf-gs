# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Benchmark: PIL (old) vs OpenCV (new) target-image loading.

Compares the previous ``load_colmap_image`` implementation against the current
one on a real batch of dataset images, both in the no-resize case (a
pre-downsampled ``images_N`` directory, which is what training actually uses) and
in the resize case (full-res source -> render target). Also reports how far apart
INTER_AREA and LANCZOS land, since that is the one visible behavior change.

    .venv/Scripts/python.exe scripts/bench_image_load.py [scene] [--count N]
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trainer.data.colmap import load_colmap_image          # noqa: E402
from trainer.data.parallel import default_workers_numbers, map_ordered  # noqa: E402


class _Entry:
    """Minimal stand-in for ColmapImageEntry (only image_name is read)."""

    def __init__(self, name: str) -> None:
        self.image_name = name


def load_pil(entry, image_dir: Path, width: int, height: int,
             load_alpha: bool = False) -> np.ndarray:
    """The PREVIOUS implementation, verbatim (git c36e699)."""
    from PIL import Image
    path = image_dir / entry.image_name
    img = Image.open(path)
    if load_alpha and img.mode == "RGBA":
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


KEEP = 4          # images retained for the correctness/quality comparison


def _time(fn, entries, image_dir, width, height, *, alpha=False, workers=1):
    """Time a full sweep, then FREE it: a decoded 360_v2 set is ~2.3 GB, and
    holding two or three of them at once turns every later phase into a
    page-fault benchmark instead of a decode benchmark. Only ``KEEP`` images
    survive, for the comparison."""
    if workers > 1:
        run = lambda: list(map_ordered(                        # noqa: E731
            lambda e: fn(e, image_dir, width, height, load_alpha=alpha),
            entries, workers=workers))
    else:
        run = lambda: [fn(e, image_dir, width, height, load_alpha=alpha)  # noqa: E731
                       for e in entries]
    gc.collect()
    start = time.perf_counter()
    out = run()
    seconds = time.perf_counter() - start
    kept = out[:KEEP]
    del out
    gc.collect()
    return seconds, kept


def _report(title, seconds, count):
    print(f"  {title:<34s} {seconds * 1e3:8.1f} ms total   "
          f"{seconds / count * 1e3:7.2f} ms/image")


def _compare(a: list[np.ndarray], b: list[np.ndarray]) -> None:
    shapes_a = {x.shape for x in a}
    shapes_b = {x.shape for x in b}
    print(f"  shapes: PIL {sorted(shapes_a)}  OpenCV {sorted(shapes_b)}   "
          f"identical: {shapes_a == shapes_b}")
    print(f"  dtypes: {a[0].dtype} / {b[0].dtype}   "
          f"contiguous: {a[0].flags['C_CONTIGUOUS']} / {b[0].flags['C_CONTIGUOUS']}   "
          f"range: [{min(x.min() for x in b):.3f}, {max(x.max() for x in b):.3f}]")
    diffs = [np.abs(x - y) for x, y in zip(a, b, strict=True)]
    mae = float(np.mean([d.mean() for d in diffs]))
    mx = float(max(d.max() for d in diffs))
    rmse = float(np.sqrt(np.mean([np.mean(d ** 2) for d in diffs])))
    psnr = float("inf") if rmse == 0.0 else 20.0 * np.log10(1.0 / rmse)
    print(f"  vs PIL: MAE {mae:.6f} ({mae * 255:.3f}/255)   "
          f"max {mx:.6f} ({mx * 255:.2f}/255)   RMSE {rmse:.6f}   PSNR {psnr:.2f} dB")
    if rmse == 0.0:
        return
    # Is the difference a BIAS (wrong exposure/gamma -> would shift the whole
    # optimization) or just filter character (sharpness)? Only the former matters.
    bias = float(np.mean([(y - x).mean() for x, y in zip(a, b, strict=True)]))
    over = float(np.mean([np.mean(np.clip(y - x, 0, None) > 4 / 255)
                          for x, y in zip(a, b, strict=True)]))
    sharp = [float(np.var(np.diff(x[..., 0], axis=1))) for x in (a[0], b[0])]
    print(f"  signed bias {bias:+.6f} ({bias * 255:+.3f}/255)   "
          f"pixels >4/255 brighter: {over * 100:.2f}%   "
          f"gradient energy PIL {sharp[0]:.5f} vs OpenCV {sharp[1]:.5f} "
          f"({sharp[1] / sharp[0]:.3f}x)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="datasets/360_v2/bicycle")
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--downsample", type=int, default=4)
    args = ap.parse_args()

    scene = Path(args.scene)
    small_dir = scene / f"images_{args.downsample}"
    full_dir = scene / "images"
    names = sorted(p.name for p in small_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:args.count]
    entries = [_Entry(n) for n in names]
    from PIL import Image
    width, height = Image.open(small_dir / names[0]).size
    full_w, full_h = Image.open(full_dir / names[0]).size
    workers = default_workers_numbers(len(entries))

    print(f"\n{len(entries)} images from {small_dir}  ({width}x{height})")
    print(f"CASE A — no resize (what training does with images_{args.downsample}/)")
    t_pil, pil = _time(load_pil, entries, small_dir, width, height)
    t_cv, cv = _time(load_colmap_image, entries, small_dir, width, height)
    t_mt, mt = _time(load_colmap_image, entries, small_dir, width, height,
                     workers=workers)
    _report("PIL, serial", t_pil, len(entries))
    _report("OpenCV, serial", t_cv, len(entries))
    _report(f"OpenCV, {workers} threads", t_mt, len(entries))
    print(f"  decode speedup: {t_pil / t_cv:.2f}x serial, "
          f"{t_pil / t_mt:.2f}x threaded ({t_cv / t_mt:.2f}x from threads)")
    _compare(pil, cv)
    assert all(np.array_equal(x, y) for x, y in zip(cv, mt, strict=True)), \
        "threaded result differs from serial"
    print("  threaded == serial, bit for bit")

    print(f"\nCASE B — decode {full_w}x{full_h} + resize to {width}x{height} "
          f"(LANCZOS vs INTER_AREA)")
    n = min(len(entries), 8)
    t_pil, pil = _time(load_pil, entries[:n], full_dir, width, height)
    t_cv, cv = _time(load_colmap_image, entries[:n], full_dir, width, height)
    t_mt, _ = _time(load_colmap_image, entries[:n], full_dir, width, height,
                    workers=workers)
    _report("PIL + LANCZOS, serial", t_pil, n)
    _report("OpenCV + INTER_AREA, serial", t_cv, n)
    _report(f"OpenCV + INTER_AREA, {workers} threads", t_mt, n)
    print(f"  decode+resize speedup: {t_pil / t_cv:.2f}x serial, "
          f"{t_pil / t_mt:.2f}x threaded")
    _compare(pil, cv)


if __name__ == "__main__":
    main()
