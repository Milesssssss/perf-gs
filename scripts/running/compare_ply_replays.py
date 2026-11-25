# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Render a native PLY at its held-out cameras and optionally compare with a saved replay.

On the first GPU, run: python scripts/running/compare_ply_replays.py RUN REFERENCE
On the second, copy RUN and REFERENCE and run the same command with RUN OUTPUT --reference REFERENCE.
Each output must be fresh. Use --image-dir if the dataset moved. The reference may also be an
archived Graphdeco package containing direct_ply_replay.json from the original RTX 4080 sweep.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from prepare_graphdeco import prepare

from trainer.data.colmap import _find_sparse_dir


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rgb(path, view):
    with Image.open(path) as im:
        if im.format != "PNG" or im.mode != "RGB" or im.size != (view["width"], view["height"]):
            raise ValueError(f"Expected a matching native RGB8 PNG at {path}")
        return np.asarray(im)


def compare_images(reference, replay):
    error = np.abs(replay.astype(np.int16) - reference)
    histogram = np.bincount(error.ravel(), minlength=256)
    # The 32x32 grid starts at the top-left. Partial edge tiles use their actual area.
    h, w = error.shape[:2]
    ys, xs = np.arange(0, h, 32), np.arange(0, w, 32)
    sums = np.add.reduceat(np.add.reduceat(error.sum(axis=2), ys), xs, axis=1)
    areas = np.diff(ys, append=h)[:, None] * np.diff(xs, append=w)
    means = sums / (3 * areas)
    y, x = np.unravel_index(np.argmax(means), means.shape)
    return histogram, {"x": int(xs[x]), "y": int(ys[y]), "mae_rgb8": float(means[y, x])}


def replay(run, output, reference, image_dir):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Expected a fresh replay directory: {output}")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if image_dir is None:
        image_dir = Path(config["scene"]) / f"images_{config['resolved']['downsample']}"
    sparse, fmt = _find_sparse_dir(image_dir.parent)
    if sparse is None:
        raise ValueError(f"Missing camera registration for {image_dir}")
    suffix = "txt" if fmt == "text" else "bin"
    ply = run / "output.ply"
    sources = {f"{key}_sha256": file_sha256(path) for key, path in (
        ("ply", ply), ("config", run / "config.json"), ("cameras", sparse / f"cameras.{suffix}"),
        ("images", sparse / f"images.{suffix}"))}
    if reference is not None:
        prior = json.loads((reference / "direct_ply_replay.json").read_text(encoding="utf-8"))
        for key, digest in sources.items():
            # Older archives have PLY/config hashes and per-camera identity but no COLMAP file hashes.
            check = key in ("ply_sha256", "config_sha256") or key in prior["sources"]
            if check and digest != prior["sources"].get(key):
                raise ValueError(f"Reference differs in {key}")

    method = prepare(run, output, image_dir, ply)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if reference is not None and (manifest["iterations"] != prior["completed_steps"] or
                                  len(manifest["views"]) != len(prior["per_view"])):
        raise ValueError("Reference step count or held-out camera count differs")
    counts = np.zeros(256, dtype=np.int64)
    views = []
    for view in manifest["views"]:
        i = view["ordinal"]
        name = f"{i:05d}.png"
        image = read_rgb(method / "renders" / name, view)
        gt = read_rgb(method / "gt" / name, view)
        current = {
            "cam_idx": view["camera_index"], "image_name": view["image_name"], "file": name,
            "gt": {"sha256": hashlib.sha256(gt.tobytes()).hexdigest()},
            "renders": {"sha256": hashlib.sha256(image.tobytes()).hexdigest()},
        }
        if reference is not None:
            old = prior["per_view"][i]
            if (any(current[k] != old[k] for k in ("cam_idx", "image_name", "file")) or
                    current["gt"]["sha256"] != old["gt"]["sha256"]):
                raise ValueError(f"Reference camera or ground truth differs for {name}")
            archived = read_rgb(reference / "test" / method.name / "renders" / name, view)
            if hashlib.sha256(archived.tobytes()).hexdigest() != old["renders"]["sha256"]:
                raise ValueError(f"Reference pixels have changed for {name}")
            histogram, region = compare_images(archived, image)
            counts += histogram
            current["worst_region"] = region
        views.append(current)
    record = {"sources": sources, "completed_steps": manifest["iterations"], "per_view": views}
    if reference is not None:
        levels = np.arange(256)
        mse = counts.dot(levels ** 2) / counts.sum()
        worst = max(views, key=lambda v: v["worst_region"]["mae_rgb8"])
        record["comparison"] = {
            "views": len(views), "rgb8_channels": int(counts.sum()), "channels_identical": int(counts[0]),
            "channels_abs_diff_gt_1": int(counts[2:].sum()), "max_abs_diff_rgb8": int(np.flatnonzero(counts)[-1]),
            "mae_rgb8": float(counts.dot(levels) / counts.sum()),
            "pooled_psnr_db": float(10 * np.log10(255 ** 2 / mse)) if mse else None,
            "worst_region": {"image_name": worst["image_name"], **worst["worst_region"]},
        }
        print(json.dumps(record["comparison"], indent=2))
    (output / "direct_ply_replay.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="archived native training output with config.json and output.ply")
    parser.add_argument("output", type=Path, help="fresh replay package")
    parser.add_argument("--reference", type=Path, help="previous GPU's replay package")
    parser.add_argument("--image-dir", type=Path, help="native images_2 or images_4 directory")
    args = parser.parse_args()
    replay(args.run, args.output, args.reference, args.image_dir)
