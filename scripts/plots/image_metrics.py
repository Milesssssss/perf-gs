# EA-authored evaluation and reporting wrappers only:
# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.
# Vendored metric functions: Copyright (C) 2023, Inria / GRAPHDECO.
# SSIM origin identified by INRIA: Copyright Evan Su, 2017 (MIT).
# Third-party portions retain their own terms; see NOTICE.md.

"""Image quality metrics (PSNR, SSIM) + a run evaluator, comparable to the SAD-GS / 3DGS paper tables.

psnr/ssim are vendored verbatim (do not modify) and aggregated like SADGS metrics.py; needs torch + torchvision, which the .venv lacks.
    python scripts/plots/image_metrics.py -m outputs/train/<run_dir> --stride 8
"""
import sys
from pathlib import Path
from argparse import ArgumentParser
from math import exp

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms.functional as tf
from PIL import Image


# --- VERBATIM from INRIA 3DGS utils/image_utils.py (via SADGS) --------------
def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# --- VERBATIM from INRIA 3DGS utils/loss_utils.py (via SADGS) ---------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
# --- end vendored code ------------------------------------------------------


def load_png(path: Path) -> torch.Tensor:
    """Match SADGS readImages: to_tensor(PIL), add batch dim, RGB, on CUDA."""
    img = Image.open(path)
    t = tf.to_tensor(img).unsqueeze(0)[:, :3, :, :]
    return t.cuda() if torch.cuda.is_available() else t


def evaluate_run(model_path: Path, stride: int = 8, score_all: bool = False):
    """Score a run's test-split cams (every ``stride``-th, or all with ``score_all``) with the vendored metrics.
    Returns (mean_psnr, mean_ssim, names, per_view_psnr, skipped)."""
    cams_dir = Path(model_path) / "cams"
    cam_dirs = sorted(cams_dir.glob("cam_*"),
                      key=lambda p: int(p.name.split("_")[1]))
    if not cam_dirs:
        raise SystemExit(f"no cam_* dirs under {cams_dir}")

    psnrs, ssims, names = [], [], []
    skipped = 0
    for cam_dir in cam_dirs:
        idx = int(cam_dir.name.split("_")[1])
        if not score_all and idx % stride != 0:
            continue
        gt_path = cam_dir / "gt.png"
        renders = sorted(cam_dir.glob("iter_*.png"))
        if not gt_path.exists() or not renders:
            skipped += 1
            continue
        r = load_png(renders[-1])          # highest iter
        g = load_png(gt_path)
        if r.shape != g.shape:
            skipped += 1
            print(f"  [skip] {cam_dir.name}: shape {tuple(r.shape)} vs gt {tuple(g.shape)}")
            continue
        psnrs.append(psnr(r, g))
        ssims.append(ssim(r, g))
        names.append(cam_dir.name)

    if not psnrs:
        raise SystemExit("no comparable render/gt pairs found")

    # Paper aggregation (metrics.py::evaluate): mean of per-image values, not dB of mean MSE.
    mean_psnr = torch.tensor(psnrs).mean().item()
    mean_ssim = torch.tensor(ssims).mean().item()
    return mean_psnr, mean_ssim, names, [p.item() for p in psnrs], skipped


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("-m", "--model_path", required=True,
                    help="v2_train run dir (contains cams/cam_XX/{gt.png,iter_*.png})")
    ap.add_argument("--stride", type=int, default=8,
                    help="test-split holdout stride (paper llffhold=8). Test = "
                         "cam indices where idx %% stride == 0.")
    ap.add_argument("--all", action="store_true",
                    help="score ALL cameras instead of just the test split.")
    args = ap.parse_args()

    mean_psnr, mean_ssim, names, per_view, skipped = evaluate_run(
        args.model_path, stride=args.stride, score_all=args.all)

    scope = "ALL cameras" if args.all else f"TEST split (every {args.stride}th)"
    print(f"\n3DGS/SAD-GS metric code (vendored, verbatim)  —  {scope}")
    print(f"  scored {len(names)} views" + (f" ({skipped} skipped)" if skipped else ""))
    print(f"  PSNR : {mean_psnr:>10.7f}   <- mean of per-image PSNR (paper table metric)")
    print(f"  SSIM : {mean_ssim:>10.7f}")
    print("\n  per-view PSNR:")
    for n, p in zip(names, per_view):
        print(f"    {n}: {p:6.2f}")


if __name__ == "__main__":
    main()
