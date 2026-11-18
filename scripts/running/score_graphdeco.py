# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Run Graphdeco's metrics.py on exported PNG pairs, loading its VGG weights once."""

import argparse
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphdeco", required=True, type=Path,
                        help="checkout of graphdeco-inria/gaussian-splatting")
    parser.add_argument("models", type=Path, nargs="+",
                        help="directories containing test/ours_3000/{renders,gt}")
    args = parser.parse_args()
    if not (args.graphdeco / "metrics.py").is_file():
        parser.error("Graphdeco checkout must contain metrics.py")
    sys.path.insert(0, str(args.graphdeco.resolve()))
    import metrics
    import torch
    from lpipsPyTorch.modules.lpips import LPIPS

    vgg = LPIPS("vgg", "0.1").cuda().eval()

    def cached_lpips(render, target, net_type):
        if net_type != "vgg":
            raise ValueError(f"Unexpected LPIPS backbone {net_type!r}")
        return vgg(render, target)

    metrics.lpips = cached_lpips
    with torch.inference_mode():
        for model in args.models:
            method = "ours_3000"
            path = model / "test" / method
            names = {p.name for p in (path / "renders").iterdir()}
            if not names or names != {p.name for p in (path / "gt").iterdir()}:
                raise ValueError(f"Missing or mismatched test images in {path}")
            # Upstream catches exceptions, so require fresh and complete output for this invocation.
            per_view = model / "per_view.json"
            previous = per_view.stat().st_mtime_ns if per_view.exists() else None
            metrics.evaluate([str(model.resolve())])
            if not per_view.exists() or per_view.stat().st_mtime_ns == previous:
                raise RuntimeError(f"Graphdeco scoring failed for {model}")
            scores = json.loads(per_view.read_text())[method]
            for metric in ("PSNR", "SSIM", "LPIPS"):
                if scores[metric].keys() != names or not all(map(math.isfinite, scores[metric].values())):
                    raise ValueError(f"Incomplete {metric} scores for {model}")


if __name__ == "__main__":
    main()
