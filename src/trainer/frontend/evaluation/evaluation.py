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
            # SADGS pops from the front; on an exhausted epoch it refills from
            # getTrainCameras().copy() and reshuffles before the next pop.
            if not self._stack:
                self._stack.extend(self._base)
                self._py.shuffle(self._stack)
            return self._stack.pop(0)
        return int(self._train_arr[self._rng.integers(0, self._train_arr.shape[0])])


def render_all_cameras(pipeline: "TrainingPipeline", scene, iter_idx: int,
                       cams_dir: Path) -> None:
    """Render every scene camera at ``iter_idx`` and write one PNG per camera.

    The renderer composites in sRGB, so the output image is written directly.
    Creates each ``cam_XX/`` directory itself: this used to rely on the
    ground-truth dump having already made them, which silently turned "GT writing
    is off" into a mid-training FileNotFoundError.
    """
    for cam_i, entry in enumerate(scene.images):
        pipeline.render(entry.camera)
        img = np.clip(pipeline.read_output_image()[..., :3], 0.0, 1.0)
        out_dir = cams_dir / f"cam_{cam_i:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"iter_{iter_idx:06d}.png"
        rgba = np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)
        png = (rgba * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(png, mode="RGBA").save(out_path)


def eval_test_psnr_srgb(pipeline: "TrainingPipeline", scene,
                        test_indices: Sequence[int],
                        targets_srgb: Sequence[np.ndarray]) -> tuple[float, int]:
    """Mean per-image sRGB PSNR over the held-out TEST split (paper metric) plus
    the live gaussian count.

    Drains the GPU queue first so the count and the per-cam renders are settled.
    Used by the ``--psnr-curve`` capture and the final test-PSNR report.
    """
    pipeline.runtime.wait_for_idle()
    n_g = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    frames = []
    for cam_i in test_indices:
        pipeline.render(scene.images[cam_i].camera)
        r = np.clip(pipeline.read_output_image()[..., :3], 0.0, 1.0)
        frames.append(psnr_image(r, targets_srgb[cam_i][..., :3]))
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
    progress: Callable[[int, int], None] = lambda _done, _total: None,
) -> float:
    """Run the held-out TEST-split eval sweep and append a row to
    ``eval_curve.jsonl``; return the updated ``eval_time_total``.

    ``progress(done, total)`` is called per camera, so a caller can say what the
    sweep is doing — at the last iteration it IS the wait the user sees.

    Reports the paper metric — the mean of per-image PSNR/SSIM (NOT dB-of-mean-
    MSE) — over the held-out test cameras (all cams when holdout is off). Sweep
    time is folded into ``eval_time_total`` and subtracted from the reported
    training-elapsed clock so the convergence-vs-time plot compares training
    speed fairly.
    """
    eval_t0 = time.time()
    # Drain GPU queue so num_gaussians_counter is accurate and no in-flight
    # work skews the per-cam renders.
    pipeline.runtime.wait_for_idle()
    eval_n_g = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
    # Accumulate PER-IMAGE PSNR (dB) and average the dB values — the paper's
    # mean-of-per-image metric. Scored on the held-out TEST split (all cams
    # when holdout is off, since test_indices == all).
    psnr_srgb_frames = []
    psnr_lin_frames = []
    psnr_srgb_mask_frames = []
    ssim_srgb_frames = []

    for done, cam_i in enumerate(test_indices, start=1):
        progress(done, len(test_indices))
        entry = scene.images[cam_i]
        pipeline.render(entry.camera)
        # Binning/overlap debug viz: read the buffers this forward just filled
        # (tile ranges + per-pixel blended count) and dump the per-view heatmaps
        # + histogram. render() already drained the queue, so buffers are ready.

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
        "eval_dt_s": eval_dt,
    }) + "\n")
    print(f"[eval] iter {it:6d}  "
          f"psnr_srgb={psnr_srgb:5.2f}  ssim={ssim_srgb:.4f} (test, mean-of-frames)  "
          f"n_g={eval_n_g:>7d}  train_elapsed={elapsed_train:.1f}s  "
          f"(eval took {eval_dt:.1f}s)")
    return eval_time_total
