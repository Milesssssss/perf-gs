# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Run the pinned LiteGS aggressive recipe and score its saved PLY in a separate process."""

import argparse
import hashlib
import json
import math
import os
import random
import re
import runpy
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor


def train(args, images):
    args.model.mkdir(parents=True, exist_ok=True)
    final_ply = args.model / "point_cloud/finish/point_cloud.ply"
    if final_ply.exists():
        raise FileExistsError(f"Model already trained: {final_ply}")
    from litegs.io_manager.colmap import load_frames
    from litegs.training import trainer

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # LiteGS opens tqdm immediately before its epoch loop, then synchronizes and
    # reads elapsed wall time before writing the final PLY. Drain setup work first.
    original_progress = trainer.tqdm

    def synchronized_progress(*positional, **keywords):
        torch.cuda.synchronize()
        return original_progress(*positional, **keywords)

    trainer.tqdm = synchronized_progress
    sys.argv = ["example_train.py", "-s", str(args.scene), "-m", str(args.model),
                "--eval", "--sh_degree", "3", "--target_primitives", "1000000", "-i", images,
                "--resolution", "1", "--iterations", "10000", "--position_lr_max_steps", "10000",
                "--position_lr_final", "0.000016", "--densification_interval", "2"]
    os.chdir(args.litegs)
    log = args.model / "train.log"
    with log.open("x") as output, redirect_stdout(output):
        runpy.run_path("example_train.py", run_name="__main__")
    seconds, = re.findall(r"^.* takes: ([\d.]+)$", log.read_text(), flags=re.MULTILINE)
    if not final_ply.is_file():
        raise RuntimeError(f"Missing final PLY in {args.model}")
    _, frames = load_frames(str(args.scene), images)
    n_train = len(frames) - len(frames[::8])
    timing = {"scene": args.scene.name, "gpu": torch.cuda.get_device_name(), "seed": 42,
              "requested_steps": 10000, "effective_steps": (10000 // n_train) * n_train,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "train_loop_s": float(seconds), "argv": sys.argv}
    (args.model / "time.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(f"{args.scene.name}: {seconds} s training loop")


@torch.no_grad()
def score(args, images):
    import litegs.config
    from utils.image_utils import psnr

    lp, _, pipeline, _ = litegs.config.get_default_arg()
    lp.resolution = 1
    cameras, frames, _, _ = litegs.io_manager.load_colmap_result(str(args.scene), images)
    heldout = frames[::8]
    if args.reference:
        reference = json.loads((args.reference / "manifest.json").read_text())
        reference_views = reference["views"]
        if [v["image_name"] for v in reference_views] != [frame.name for frame in heldout]:
            raise ValueError("LiteGS and PERF-GS held out different cameras")
    for frame in heldout:
        frame.load_image(lp.resolution)
    dataset = litegs.data.CameraFrameDataset(cameras, heldout, lp.resolution, pipeline.device_preload)

    ply = args.model / "point_cloud/finish/point_cloud.ply"
    parameters = [torch.as_tensor(p, device="cuda", dtype=torch.float32)
                  for p in litegs.io_manager.load_ply(str(ply), lp.sh_degree)]
    centers, extents = None, None
    if pipeline.cluster_size:
        ordered = litegs.scene.point.spatial_refine(False, None, *parameters)
        parameters = litegs.scene.cluster.cluster_points(pipeline.cluster_size, *ordered)
        xyz, scales, rotations = parameters[:3]
        centers, extents = litegs.scene.cluster.get_cluster_AABB(
            xyz, scales.exp(), torch.nn.functional.normalize(rotations, dim=0))

    out = args.model / "test/litegs_aggressive"
    for group in ("renders", "gt"):
        (out / group).mkdir(parents=True, exist_ok=True)
    results = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for i, (view, proj, frustum, rgb, _) in enumerate(loader):
        filename = f"{i:05d}.png"
        frame = heldout[i]
        with Image.open(frame.img_source) as original:
            gt = original.convert("RGB")
        decode_differences = 0
        if args.reference:
            expected = reference_views[i]
            if (expected["camera_index"], expected["width"], expected["height"]) != (i * 8, *gt.size):
                raise ValueError(f"Different evaluation camera or size for {frame.name}")
            with Image.open(args.reference / f"test/ours_{reference['iterations']}/gt" / filename) as benchmark_gt:
                benchmark_rgb = benchmark_gt.convert("RGB")
                if benchmark_rgb.size != gt.size:
                    raise ValueError(f"Different reference size for {frame.name}")
                decode_differences = int(np.count_nonzero(np.asarray(gt) != np.asarray(benchmark_rgb)))
                gt = benchmark_rgb
        if rgb.shape[-2:] != (gt.height, gt.width):
            raise ValueError(f"LiteGS resized {frame.name} during loading")
        view = view.cuda()
        _, _, *visible = litegs.render.render_preprocess(centers, extents, frustum.cuda(), view, *parameters,
                                                         None, None, pipeline, lp.sh_degree)
        rendered, *_ = litegs.render.render(view, proj.cuda(), *visible, None, None, None,
                                            lp.sh_degree, (gt.height, gt.width), pipeline)
        if not torch.isfinite(rendered).all():
            raise ValueError(f"Non-finite render for {frame.name}")
        rgb8 = rendered[0].mul(255).add(0.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        Image.fromarray(rgb8.cpu().numpy()).save(out / "renders" / filename)
        gt.save(out / "gt" / filename)
        with Image.open(out / "renders" / filename) as prediction, Image.open(out / "gt" / filename) as target:
            view_psnr = psnr(to_tensor(prediction).unsqueeze(0).cuda(), to_tensor(target).unsqueeze(0).cuda()).item()
        if not math.isfinite(view_psnr):
            raise ValueError(f"Non-finite PSNR for {frame.name}")
        results.append({"image_name": frame.name, "camera_index": i * 8, "png": filename,
                        "width": gt.width, "height": gt.height, "gt_rgb_sha256": hashlib.sha256(gt.tobytes()).hexdigest(),
                        "source_decode_changed_channels": decode_differences, "psnr_db": view_psnr})
    scene_psnr = torch.tensor([v["psnr_db"] for v in results]).mean().item()
    report = {"scene": args.scene.name, "images": images, "heldout_views": len(results), "psnr_db": scene_psnr,
              "render_source": "LiteGS native renderer, final SH3 PLY",
              "reference_source": str(args.reference) if args.reference else "PIL RGB decode of stored image",
              "metric_source": "Graphdeco utils.image_utils.psnr on saved RGB8 PNG pairs", "views": results}
    (args.model / "heldout_rgb8.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{args.scene.name}: {scene_psnr:.4f} dB over {len(results)} held-out views", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--litegs", required=True, type=Path, help="pinned LiteGS source directory")
    common.add_argument("--scene", required=True, type=Path)
    common.add_argument("--model", required=True, type=Path, help="LiteGS output directory")
    sub.add_parser("train", parents=[common])
    evaluation = sub.add_parser("score", parents=[common])
    evaluation.add_argument("--graphdeco", required=True, type=Path, help="Graphdeco source checkout")
    evaluation.add_argument("--reference", type=Path, help="PERF-GS Graphdeco input with manifest.json")
    args = parser.parse_args()
    args.scene, args.model, args.litegs = (p.resolve() for p in (args.scene, args.model, args.litegs))
    sys.path.insert(0, str(args.litegs))
    if (args.scene / "train_test_split.json").exists():
        raise ValueError("This benchmark uses the standard every-eighth holdout")
    if args.scene.name in {"bonsai", "counter", "kitchen", "room"}:
        images = "images_2"
    elif args.scene.name in {"bicycle", "flowers", "garden", "stump", "treehill"}:
        images = "images_4"
    else:
        raise ValueError(f"Unknown Mip-NeRF 360 scene: {args.scene.name}")
    if args.mode == "train":
        train(args, images)
    else:
        sys.path.insert(0, str(args.graphdeco.resolve()))
        score(args, images)


if __name__ == "__main__":
    main()
