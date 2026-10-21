# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Evaluation renders and metrics for the training frontend.

Groups the "measure the model" responsibility into one place: full-scene camera
dumps and the held-out TEST-split sRGB PSNR used by the ``--psnr-curve`` capture.

These were previously a module-level helper plus a closure inside ``app.main()``
that implicitly captured ``pipeline``/``scene``/``test_indices``/``targets_srgb``
from the enclosing scope; here they take those as explicit parameters so the
evaluation path is self-contained and testable in isolation.
"""
from __future__ import annotations

import numpy as np
import random

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


class CameraSampler:

    def __init__(self, mode: str, train_indices, test_indices, seed: int,
                 rng: np.random.Generator) -> None:
        self.mode = mode
        self._rng = rng
        self._train_arr = np.asarray(train_indices, dtype=np.int64)
        if mode == "sadgs":
            self._py = random.Random(seed)
            train = list(train_indices)   # name-sorted ascending, minus holdout
            test = list(test_indices)
            self._py.shuffle(train)       # Scene load-time shuffle (train)
            self._py.shuffle(test)        # Scene load-time shuffle (test) — consumes stream
            self._base = list(train)      # getTrainCameras() order
            self._stack = list(self._base)
            print(f"[sample-mode=sadgs] replicating SADGS random-shuffle sampling "
                  f"(seed={seed}, {len(self._base)} train cams, batch_size=1, "
                  f"first epoch in load-shuffled order, reshuffle on refill)")

    def next(self) -> int:
        """Return the train-camera index to optimize this iteration."""
        if self.mode == "sadgs":
            if not self._stack:
                self._stack.extend(self._base)
                self._py.shuffle(self._stack)
            return self._stack.pop(0)
        return int(self._train_arr[self._rng.integers(0, self._train_arr.shape[0])])


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
    n_g = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
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
    eval_n_g = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])

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
        # null rather than 0.0 when unavailable: 0.0 is a PERFECT LPIPS score and
        # would read as the best possible result instead of a missing one.
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
