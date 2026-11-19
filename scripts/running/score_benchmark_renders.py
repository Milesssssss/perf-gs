# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Score a trainer's native RGB8 PNGs against the matched PERF-GS benchmark photos."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


def read_rgb(path, size):
    with Image.open(path) as image:
        if image.format != "PNG" or image.size != size or image.mode not in {"RGB", "RGBA"}:
            raise ValueError(f"Expected a {size} RGB8 PNG at {path}")
        if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
            raise ValueError(f"Non-opaque render at {path}")
        return image.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphdeco", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="prepared Graphdeco model directory")
    parser.add_argument("--views", type=Path, required=True, help="native scene and image_name/render/optional gt entries")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.graphdeco.resolve()))
    from utils.image_utils import psnr

    native = json.loads(args.views.read_text(encoding="utf-8"))
    reference = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    images = {view["image_name"]: view for view in native["views"]}
    expected = reference["views"]
    if (not expected or len(images) != len(native["views"]) or len(images) != len(expected)
            or set(images) != {view["image_name"] for view in expected}):
        raise ValueError(f"Held-out camera names differ for {native['scene']}")
    if len({view["render"] for view in images.values()}) != len(images):
        raise ValueError("Each camera needs its own native render")

    torch.set_num_threads(2)
    results = []
    for ref in expected:
        view = images[ref["image_name"]]
        size = (ref["width"], ref["height"])
        target = read_rgb(args.reference / f"test/ours_{reference['iterations']}/gt/{ref['ordinal']:05d}.png", size)
        rendered = read_rgb(args.views.parent / view["render"], size)
        score = psnr(to_tensor(rendered).unsqueeze(0), to_tensor(target).unsqueeze(0)).item()
        if not math.isfinite(score):
            raise ValueError(f"Invalid PSNR for {ref['image_name']}")
        result = {key: ref[key] for key in ("image_name", "camera_index", "width", "height")}
        result.update(render=view["render"], gt_rgb_sha256=hashlib.sha256(target.tobytes()).hexdigest(), psnr_db=score)
        if view.get("gt"):
            gt = read_rgb(args.views.parent / view["gt"], size)
            result["native_gt_changed_channels"] = int(np.count_nonzero(np.asarray(gt) != np.asarray(target)))
        results.append(result)

    scene_psnr = torch.tensor([row["psnr_db"] for row in results]).mean().item()
    report = {"scene": native["scene"], "heldout_views": len(results), "psnr_db": scene_psnr,
              "metric_source": "Graphdeco RGB8 PSNR, CPU", "views": results}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{native['scene']}: {scene_psnr:.5f} dB over {len(results)} views")


if __name__ == "__main__":
    main()
