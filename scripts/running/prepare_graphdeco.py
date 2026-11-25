# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Prepare native held-out photos and renders from a saved PLY or run report for Graphdeco metrics.py."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from trainer.data.colmap import _find_sparse_dir, _parse_images, _parse_images_bin


def _model_renderer(model, args, image_dir, factor, capacity, expected_count=None):
    import numpy as np

    from trainer.backend.render.pipeline import TrainingPipeline
    from trainer.config.config import AppearanceConfig, RasterConfig
    from trainer.data.colmap import load_colmap
    from trainer.data.ply import load_ply
    from trainer.frontend.factory import create_training_device

    state = load_ply(model)
    if not args.mip_splatting or "filter_3d" not in state:
        raise ValueError("--ply requires a native Mip PLY with its stored 3D filter")
    if expected_count is not None and len(state["means"]) != expected_count:
        raise ValueError("PLY Gaussian count differs from the run report")
    if args.mip_kernel_size != state["mip_kernel_size"]:
        raise ValueError("PLY Mip kernel differs from the training configuration")
    strength = getattr(args, "lens_strength", None)
    if strength is None:
        strength = float(getattr(args, "natural_vignetting", False))
    if state["active_sh_degree"] > args.sh_degree or not np.isclose(strength, state.get("natural_vignetting", 0)):
        raise ValueError("PLY appearance differs from the training configuration")
    cameras = load_colmap(image_dir.parent, downsample=factor).images
    width, height = cameras[0].camera.image_size
    raster = RasterConfig(mip_enabled=True, kernel_size=state["mip_kernel_size"],
                          z_near_cull=args.z_near_cull, cluster_cull=args.cluster_cull)
    device = create_training_device(args)
    print(f"Replaying PLY on {device.info.adapter_name} ({device.info.api_name})")
    pipe = TrainingPipeline(device, width, height, raster_config=raster,
                            appearance_config=AppearanceConfig(sh_degree=args.sh_degree),
                            max_splats=len(state["means"]), max_sort_pairs=capacity)
    pipe.load_snapshot(**state, reset_optimizer=False)
    pipe.set_extinction_hardness(args.extinction_hardness)

    def render(idx, name, path):
        if cameras[idx].image_name != name:
            raise ValueError(f"PLY camera order differs at {name}")
        pipe.render(cameras[idx].camera)
        if int(pipe.buffers.sort.stats.to_numpy().view(np.uint32)[1]):
            raise ValueError(f"PLY render exceeded the sort capacity for {name}")
        rgb = np.clip(pipe.read_output_image()[..., :3], 0.0, 1.0)
        Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8)).save(path)
    return render


def _check_saved_sort(views, memory):
    if "heldout_sort_overflows" in memory:
        if memory["heldout_sort_overflows"] != 0:
            raise ValueError("Held-out rendering exceeded the sort capacity")
        return
    # Older reports count stored pairs after clamping. Equality cannot establish that all pairs fit.
    capacity = memory.get("max_sort_pairs", 0)
    if not all(0 <= view.get("tile_pairs", -1) < capacity for view in views):
        raise ValueError("Cannot verify the saved held-out sort counts. Rerender with --ply.")


def prepare(run: Path, output: Path, image_dir: Path | None = None, ply: Path | None = None) -> Path:
    report_path = run / "run_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    memory = {}
    expected_count = None
    plateau_step = None
    if report is None:
        if ply is None:
            raise ValueError("Supply --ply for training-only output, or train with --run-report --save-test-renders.")
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        args = argparse.Namespace(**config["cli_args"])
        scene, scene_path = config["resolved"], Path(config["scene"])
        requested, completed, holdout = args.iters, None, args.eval_holdout
        with (run / "log.jsonl").open(encoding="utf-8") as log:
            for line in log:
                event = json.loads(line)
                if event.get("event") == "train_complete":
                    completed = event["iters"]
                elif (event.get("event") == "training_stop" and event.get("stop_reason") == "plateau"
                      and getattr(args, "preset", None) == "quality+"):
                    plateau_step = event["iter"]
                elif event.get("event") == "training_memory":
                    memory = event
    else:
        args = argparse.Namespace(**report["cli_args"]) if ply is not None else None
        scene, scene_path = report["scene"], Path(report["scene_path"])
        requested, completed, holdout = report["iters_requested"], report["iters_completed"], scene["eval_holdout"]
        if report.get("stop_reason") == "plateau" and report.get("cli_args", {}).get("preset") == "quality+":
            plateau_step = completed
        if ply is not None:
            expected_count = report["model"]["n_gaussians"]
        memory = report["memory"]
    reached_plateau = (completed is not None and completed > 0 and plateau_step == completed
                       and (requested == 0 or completed <= requested))
    if not reached_plateau and (requested < 1 or completed != requested):
        raise ValueError(f"Expected {requested:,} completed training steps, found {completed}")
    if not holdout or scene["holdout_stride"] != 8:
        raise ValueError("Training must hold out every eighth registered camera")
    psnr_path = run / "test_psnr.json"
    if psnr_path.is_file():
        timing = json.loads(psnr_path.read_text(encoding="utf-8"))
        if timing["sort_overflows"] != 0:
            raise ValueError("Training exceeded the sort capacity")
        memory = {**timing, **memory}
    if report is None and "sort_overflows" not in memory:
        print("Legacy training log has no sort overflow count. Check the original training console for warnings.")
    elif memory.get("sort_overflows") != 0:
        raise ValueError("The benchmark requires zero recorded sort overflows")
    if scene_path.name in {"bonsai", "counter", "kitchen", "room"}:
        factor = 2
    elif scene_path.name in {"bicycle", "flowers", "garden", "stump", "treehill"}:
        factor = 4
    else:
        raise ValueError(f"Unrecognized Mip-NeRF 360 scene: {scene_path.name}")
    if image_dir is None:
        image_dir = scene_path / f"images_{factor}"
    if scene["downsample"] != factor or image_dir.name != f"images_{factor}":
        raise ValueError(f"Expected the stored images_{factor} directory for {scene_path.name}")
    sparse, fmt = _find_sparse_dir(image_dir.parent)
    if sparse is None:
        raise ValueError(f"Cannot find COLMAP image registration in {image_dir.parent}")
    entries = _parse_images(sparse / "images.txt") if fmt == "text" else _parse_images_bin(sparse / "images.bin")
    names = sorted(entry["name"] for entry in entries)
    indices = list(range(0, len(names), 8))
    if report is None:
        if (scene["n_cams"] != len(names) or scene["n_test"] != len(indices)
                or scene["n_train"] != len(names) - len(indices)):
            raise ValueError("COLMAP cameras disagree with the recorded train/test split")
        views = [{"image_name": names[i], "width": scene["width"], "height": scene["height"]} for i in indices]
    else:
        views = report["quality"]["per_view"]
        if (scene["test_indices"] != indices or [view["cam_idx"] for view in views] != indices
                or len(indices) != report["quality"]["n_test_views"]):
            raise ValueError("Expected the complete every-eighth-camera held-out split")
    method = output / "test" / f"ours_{completed}"
    if method.exists():
        raise FileExistsError(f"Output method already exists: {method}")
    renderer = None
    if ply is not None:
        from trainer.backend.render.pipeline import default_sort_pairs
        capacity = memory.get("max_sort_pairs", default_sort_pairs(args.max_splats))
        renderer = _model_renderer(ply, args, image_dir, factor, capacity, expected_count)
    else:
        _check_saved_sort(views, memory)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as tmp:
        stage = Path(tmp) / "method"
        for directory in ("renders", "gt"):
            (stage / directory).mkdir(parents=True)
        for i, (idx, view) in enumerate(zip(indices, views)):
            if names[idx] != view["image_name"] or (renderer is None and "render_png" not in view):
                raise ValueError(f"Missing or mismatched saved render for {names[idx]}. Use --run-report --save-test-renders.")
            filename = f"{i:05d}.png"
            if renderer is None:
                render_path = run / "test_renders" / view["render_png"]
            else:
                render_path = stage / "renders" / filename
                renderer(idx, names[idx], render_path)
            with Image.open(image_dir / names[idx]) as photo, Image.open(render_path) as rendered:
                if rendered.format != "PNG" or rendered.mode != "RGB":
                    raise ValueError(f"Expected an RGB8 PNG at {render_path}")
                if (photo.size != rendered.size or rendered.size != (view["width"], view["height"])
                        or photo.size != (scene["width"], scene["height"])):
                    raise ValueError(f"Training, saved render and native source dimensions must agree for {names[idx]}")
                photo.convert("RGB").save(stage / "gt" / filename)
            if renderer is None:
                shutil.copyfile(render_path, stage / "renders" / filename)
        # External-trainer comparisons use these names to pair native cameras to the reference RGB8s.
        manifest = {"iterations": completed, "views": [
            {"ordinal": i, "camera_index": idx, "image_name": v["image_name"], "width": v["width"], "height": v["height"]}
            for i, (idx, v) in enumerate(zip(indices, views))]}
        staged_manifest = Path(tmp) / "manifest.json"
        staged_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        method.parent.mkdir(exist_ok=True)
        staged_manifest.replace(output / "manifest.json")
        stage.rename(method)
    return method


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="trainer output with a saved native PLY or a report and test renders")
    parser.add_argument("output", type=Path, help="model directory for Graphdeco metrics.py")
    parser.add_argument("--image-dir", type=Path, help="original ground-truth image directory if the dataset moved")
    parser.add_argument("--ply", type=Path, help="render the stored native Mip model at the recorded held-out cameras")
    args = parser.parse_args()
    print(prepare(args.run, args.output, args.image_dir, args.ply))
