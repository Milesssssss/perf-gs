# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Evaluation renders and metrics: full-scene camera dumps, held-out sRGB PSNR, eval sweeps."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
from PIL import Image

from trainer.common.color import srgb_to_linear
from trainer.common.metrics import psnr as psnr_image, ssim as ssim_image
from trainer.frontend.run_control import Checkpoint, uninterrupted

if TYPE_CHECKING:
    from trainer.backend.render.pipeline import TrainingPipeline


def render_all_cameras(pipeline: "TrainingPipeline", scene, iter_idx: int,
                       cams_dir: Path, *,
                       checkpoint: Checkpoint = uninterrupted) -> None:
    for cam_i, entry in enumerate(scene.images):
        checkpoint()
        pipeline.render(entry.camera)
        img = np.clip(pipeline.read_output_image()[..., :3], 0.0, 1.0)
        out_dir = cams_dir / f"cam_{cam_i:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"iter_{iter_idx:06d}.png"
        rgba = np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)
        png = (rgba * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(png, mode="RGBA").save(out_path)
    checkpoint()


def eval_test_psnr_srgb(pipeline: "TrainingPipeline", scene,
                        test_indices: Sequence[int],
                        targets_srgb: Sequence[np.ndarray], *,
                        checkpoint: Checkpoint = uninterrupted) -> tuple[float, int]:
    checkpoint()
    pipeline.runtime.wait_for_idle()
    n_g = pipeline.primitives.count()
    frames = []
    for cam_i in test_indices:
        checkpoint()
        pipeline.render(scene.images[cam_i].camera)
        r = np.clip(pipeline.read_output_image()[..., :3], 0.0, 1.0)
        frames.append(psnr_image(r, targets_srgb[cam_i][..., :3]))
    checkpoint()
    return (float(np.mean(frames)) if frames else 0.0), n_g


def run_eval_sweep(
    *,
    pipeline: "TrainingPipeline",
    scene,
    it: int,
    test_indices: Sequence[int],
    targets_srgb: Sequence[np.ndarray],
    targets_linear: Callable[[int], np.ndarray],
    cams_dir: Path,
    eval_curve_json,
    t_start: float,
    eval_time_total: float,
    eval_holdout: bool,
    with_lpips: bool = False,
    progress: Callable[[int, int], None] = lambda _done, _total: None,
    checkpoint: Checkpoint = uninterrupted,
) -> float:

    checkpoint()
    eval_t0 = time.time()

    pipeline.runtime.wait_for_idle()
    eval_n_g = pipeline.primitives.count()

    psnr_srgb_frames = []
    psnr_lin_frames = []
    psnr_srgb_mask_frames = []
    ssim_srgb_frames = []
    lpips_frames: list[float] = []

    lpips_ready = False
    if with_lpips:
        from trainer.common import perceptual
        lpips_ready = perceptual.load()

    for done, cam_i in enumerate(test_indices, start=1):
        progress(done, len(test_indices))
        checkpoint()
        entry = scene.images[cam_i]
        pipeline.render(entry.camera)

        r_srgb = np.clip(pipeline.read_output_image()[..., :3], 0.0, 1.0)
        t_srgb = targets_srgb[cam_i][..., :3]
        psnr_srgb_frames.append(psnr_image(r_srgb, t_srgb))
        ssim_srgb_frames.append(ssim_image(r_srgb, t_srgb))
        r_lin = srgb_to_linear(r_srgb)
        t_lin = targets_linear(cam_i)
        psnr_lin_frames.append(psnr_image(r_lin, t_lin))
        mask = (t_srgb.sum(-1) > 0.05) | (r_srgb.sum(-1) > 0.05)
        if mask.sum() > 0:
            psnr_srgb_mask_frames.append(psnr_image(r_srgb[mask], t_srgb[mask]))
        if lpips_ready:
            value = perceptual.lpips(r_srgb, t_srgb)
            if value is not None:
                lpips_frames.append(value)

    checkpoint()

    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    psnr_srgb = _mean(psnr_srgb_frames)
    ssim_srgb = _mean(ssim_srgb_frames)
    psnr_lin = _mean(psnr_lin_frames)
    psnr_srgb_mask = _mean(psnr_srgb_mask_frames)
    eval_dt = time.time() - eval_t0
    eval_time_total += eval_dt
    elapsed_train = time.time() - t_start - eval_time_total
    eval_curve_json.write(json.dumps({
        "iter": it,
        "elapsed_s": elapsed_train,
        "n_gaussians": eval_n_g,
        "n_eval_cams": len(test_indices),
        "eval_holdout": bool(eval_holdout),
        # Paper metric: mean of per-image PSNR/SSIM over the test split.
        "psnr_srgb": psnr_srgb,
        "ssim_srgb": ssim_srgb,
        "psnr_linear": psnr_lin,
        "psnr_srgb_masked": psnr_srgb_mask,
        # null when unavailable: 0.0 would read as a perfect LPIPS score.
        "lpips": (float(np.mean(lpips_frames)) if lpips_frames else None),
        "n_lpips_views": len(lpips_frames),
        "eval_dt_s": eval_dt,
    }) + "\n")
    lpips_text = (f"  lpips={float(np.mean(lpips_frames)):.4f}"
                  if lpips_frames else "")
    print(f"[eval] iter {it:6d}  "
          f"psnr_srgb={psnr_srgb:5.2f}  ssim={ssim_srgb:.4f} (test, mean-of-frames)"
          f"{lpips_text}  "
          f"n_g={eval_n_g:>7d}  train_elapsed={elapsed_train:.1f}s  "
          f"(eval took {eval_dt:.1f}s)")
    return eval_time_total
