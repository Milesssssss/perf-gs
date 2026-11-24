# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Run Graphdeco's metrics.py on exported PNG pairs, loading its VGG weights once."""

import argparse
import csv
import filecmp
import json
import math
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from statistics import fmean

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"


def preset_rows(models, graphdeco, expected_gpu):
    """Match prepared renders to measured Vulkan runs under the published nine-scene protocol."""
    from trainer.frontend.settings import preset_values

    rev = subprocess.check_output(["git", "-C", graphdeco, "rev-parse", "HEAD"], text=True).strip()
    if rev != REFERENCE_COMMIT:
        raise ValueError(f"Preset benchmarks use Graphdeco {REFERENCE_COMMIT}, found {rev}")
    subprocess.run(["git", "-C", graphdeco, "diff", "--exit-code", "--quiet", "HEAD", "--",
                    "metrics.py", "utils/loss_utils.py", "utils/image_utils.py", "lpipsPyTorch"], check=True)
    with (ROOT / "docs/mipnerf360_3k_presets.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        reference = {(row["preset"], row["scene"]): row for row in reader}
        columns = reader.fieldnames
    if len(models) != len(reference):
        raise ValueError(f"Expected {len(reference)} preset/scene runs, received {len(models)}")
    rows = {}
    for model in models:
        run = ROOT / "outputs/train" / model.name
        report = json.loads((run / "run_report.json").read_text(encoding="utf-8"))
        timing = json.loads((run / "test_psnr.json").read_text(encoding="utf-8"))
        args, scene = report["cli_args"], report["scene"]
        key = (args["preset"], PureWindowsPath(report["scene_path"]).name)
        if key in rows or key not in reference:
            raise ValueError(f"Duplicate or unexpected preset/scene {key} in {model}")
        preset = preset_values(key[0])
        if any(args.get(field) != value for field, value in preset.items()):
            raise ValueError(f"{model} differs from the current {key[0]} preset")
        gpu = report["device"]["adapter_name"]
        if not gpu or (expected_gpu and gpu != expected_gpu):
            raise ValueError(f"{model} reports GPU {gpu!r}, expected {expected_gpu!r}")
        expected_gpu = gpu
        views = report["quality"]["per_view"]
        wall = report["timing"]["train_wall_s"]
        if (report["run_name"] != model.name or report["device"]["api_name"] != "Vulkan" or not args["no_viewer"]
                or any(args[k] for k in ("eval_every", "log_every", "preview_every", "eval_iters", "psnr_curve"))):
            raise ValueError(f"{model} must match its headless Vulkan training run with mid-loop observations disabled")
        if (not scene["eval_holdout"] or scene["holdout_stride"] != 8
                or scene["downsample"] != (2 if key[1] in ("bonsai", "counter", "kitchen", "room") else 4)
                or not views or not (len(views) == scene["n_test"] == timing["n_test_cams"])):
            raise ValueError(f"Incorrect held-out split or downsample for {model}")
        if (not (report["iters_requested"] == report["iters_completed"] == timing["iters"])
                or timing.get("sort_overflows") != 0 or timing.get("heldout_sort_overflows") != 0
                or not (math.isfinite(wall) and wall > 0 and wall == timing["train_wall_s"])):
            raise ValueError(f"Invalid training length, timing or sort capacity for {model}")
        # A retrained run can have the same name as a previously prepared set of PNGs.
        for index, view in enumerate(views):
            prepared = model / "test" / f"ours_{timing['iters']}" / "renders" / f"{index:05d}.png"
            if view["cam_idx"] != index * 8 or (view["width"], view["height"]) != (scene["width"], scene["height"]):
                raise ValueError(f"Unexpected evaluation camera or dimensions at {prepared}")
            if not filecmp.cmp(run / "test_renders" / view["render_png"], prepared, shallow=False):
                raise ValueError(f"{prepared} differs from its timed run. Prepare a fresh model directory.")
        rows[key] = {
            "seed": args["seed"], "preset": key[0], "scene": key[1], "steps": timing["iters"],
            "sh_degree": report["model"]["sh_degree"], "pixel_fraction": report["stbn"]["pixel_fraction"],
            "train_width": scene["width"], "train_height": scene["height"],
            "eval_width": views[0]["width"], "eval_height": views[0]["height"], "heldout_views": len(views),
            "train_loop_s": wall, "final_gaussians": report["model"]["n_gaussians"],
            "vram_usage_bytes": timing.get("vram_usage_bytes"), "gpu": gpu, "run_name": model.name,
            "metric_source": f"Graphdeco RGB8 PSNR/SSIM/VGG-LPIPS ({rev[:7]})",
        }
        for field in ("seed", "train_width", "train_height", "heldout_views"):
            if rows[key][field] != float(reference[key][field]):
                raise ValueError(f"{model} has {field}={rows[key][field]}, expected {reference[key][field]}")
    return columns, {rows[key]["run_name"]: rows[key] for key in reference}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphdeco", required=True, type=Path,
                        help="checkout of graphdeco-inria/gaussian-splatting")
    parser.add_argument("--preset-csv", type=Path, help="write a new per-scene CSV for a complete preset benchmark")
    parser.add_argument("--expect-gpu", help="require this exact trainer adapter name in the preset CSV")
    parser.add_argument("models", type=Path, nargs="+",
                        help="directories containing one test/ours_<steps>/{renders,gt}")
    args = parser.parse_args()
    if not (args.graphdeco / "metrics.py").is_file():
        parser.error("Graphdeco checkout must contain metrics.py")
    if args.expect_gpu and not args.preset_csv:
        parser.error("--expect-gpu requires --preset-csv")
    if args.preset_csv:
        if args.preset_csv.exists():
            parser.error(f"CSV already exists: {args.preset_csv}")
        columns, rows = preset_rows(args.models, args.graphdeco, args.expect_gpu)
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
            methods = list((model / "test").glob("ours_*"))
            if len(methods) != 1:
                raise ValueError(f"Expected one prepared method in {model}/test")
            path = methods[0]
            method = path.name
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
            if args.preset_csv:
                row = rows[model.name]
                summary = json.loads((model / "results.json").read_text())[method]
                if len(names) != row["heldout_views"]:
                    raise ValueError(f"Wrong number of scored views for {model}")
                for metric, column in (("PSNR", "psnr_db"), ("SSIM", "ssim"), ("LPIPS", "vgg_lpips")):
                    if not math.isclose(summary[metric], fmean(scores[metric].values()), rel_tol=1e-6, abs_tol=1e-8):
                        raise ValueError(f"Inconsistent {metric} mean for {model}")
                    row[column] = summary[metric]
    if args.preset_csv:
        with args.preset_csv.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows.values():
                label = "{preset}_{scene}_s{seed}".format(**row)
                writer.writerow({**row, "run_name": label})
        print(args.preset_csv)


if __name__ == "__main__":
    main()
