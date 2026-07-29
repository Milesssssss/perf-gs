"""Post-training analysis: final per-camera metrics, summary + PLY export.

This is the ``#TODO: make stuff under here into a cleaner post-training analysis
routine`` block lifted verbatim out of ``app.main()``. It runs after the hot
loop (and after the ``--profile-run`` early return) to:

  * refresh the Mip-Splatting 3D filter so metrics reflect final positions,
  * render every camera and compute full-frame + foreground-masked PSNR/SSIM,
  * write ``final_psnr.json`` (paper metrics = mean of per-image dB) and
    ``output.ply``, and print the headline report.

Internal metric aliases (``_srgb_to_linear`` / ``_psnr_image`` / ``_ssim_image``)
match the names used in the original inline block so the body is a verbatim move.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from trainer.color import srgb_to_linear as _srgb_to_linear
from trainer.export import save_ply
from trainer.metrics import psnr as _psnr_image, ssim as _ssim_image
from trainer.training.evaluation import render_all_cameras

if TYPE_CHECKING:
    import argparse

    from trainer.pipeline import Pipeline


def run_post_training(
    *,
    pipeline: "Pipeline",
    scene,
    args: "argparse.Namespace",
    run_dir: Path,
    cams_dir: Path,
    targets_srgb: Sequence[np.ndarray],
    targets_linear: Callable[[int], np.ndarray],
    test_indices: Sequence[int],
    stride: int,
    n_cams: int,
    train_wall_s: float,
    completed_iters: int,
    t_start: float,
) -> None:
    """Compute + report the final metrics and export the trained PLY."""
    # Mip-Splatting: refresh the 3D filter one last time so the final metrics
    # render reflects the FINAL gaussian positions. The periodic recompute above
    # skips the last ~100 iters, so without this the reported PSNR would use a
    # filter ~100-200 iters stale.
    if args.mip_splatting:
        pipeline.compute_3d_filter()
        #_log_filter3d("final")

    # Post-training per-camera metrics. Include BOTH the full-frame PSNR
    # (linear and sRGB) and the foreground-masked sRGB PSNR — the latter
    # is the honest silhouette metric (avoids the black-background inflation
    # that misled us on the v1 10k run).
    print("computing final per-camera metrics...")
    per_cam: list[dict] = []
    total_mse_lin = 0.0
    total_mse_srgb = 0.0
    total_mse_srgb_mask = 0.0
    n_mask_used = 0

    #render all cameras in the scene and store them as PNGs in the run_dir/cams/cam_XX/ folder, and compute PSNR/MSE metrics
    if args.compute_final_metrics:
        render_all_cameras(pipeline, scene, args.iters, cams_dir)

        for cam_i, entry in enumerate(scene.images):
            pipeline.render(entry.camera)
            # Render is sRGB now (INRIA/gsplat/SADGS convention). The sRGB PSNR
            # (this is the paper-reported metric) is a direct compare vs t_srgb;
            # the linear PSNR is kept for continuity — both are linearized.
            r_srgb = np.clip(pipeline.read_output_image()[..., :3], 0.0, 1.0)
            t_srgb = targets_srgb[cam_i][..., :3]
            mse_srgb = float(np.mean((r_srgb - t_srgb) ** 2))

            r_lin = _srgb_to_linear(r_srgb)
            t_lin = targets_linear(cam_i)
            mse_lin = float(np.mean((r_lin - t_lin) ** 2))

            mask = (t_srgb.sum(-1) > 0.05) | (r_srgb.sum(-1) > 0.05)
            if mask.sum() > 0:
                mse_srgb_mask = float(np.mean((r_srgb[mask] - t_srgb[mask]) ** 2))
                total_mse_srgb_mask += mse_srgb_mask
                n_mask_used += 1
            else:
                mse_srgb_mask = 0.0

            total_mse_lin += mse_lin
            total_mse_srgb += mse_srgb
            per_cam.append({
                "cam_idx": cam_i, "image_name": entry.image_name,
                "in_test_split": cam_i in set(test_indices),
                "mse_linear": mse_lin,
                "psnr_linear": _psnr_image(r_lin, t_lin),
                "mse_srgb": mse_srgb,
                "psnr_srgb": _psnr_image(r_srgb, t_srgb),
                "ssim_srgb": _ssim_image(r_srgb, t_srgb),
                "mse_srgb_masked": mse_srgb_mask,
                "psnr_srgb_masked": (_psnr_image(r_srgb[mask], t_srgb[mask])
                                    if mask.sum() > 0 else 0.0),
            })

        # --- Reported metric = paper metric everywhere ---
        # PSNR is ALWAYS the arithmetic mean of per-image PSNR (3DGS/SAD-GS
        # convention; see _psnr_image + scripts/image_metrics.py). The headline
        # `psnr_srgb` is that mean over the held-out TEST split — THE number to
        # compare against the SAD-GS / 3DGS tables. We also break it out by
        # train/all and keep the old dB-of-mean-MSE values only as explicitly
        # suffixed `*_db_of_mean_mse` fields, so nothing silently reports the
        # pessimistic aggregation.
        test_set = set(test_indices)
        test_cams = [c for c in per_cam if c["cam_idx"] in test_set]
        train_cams = [c for c in per_cam if c["cam_idx"] not in test_set]

        def mean_metric(cams, key):
            return float(np.mean([c[key] for c in cams])) if cams else 0.0

        def db_of_mean_mse(total_mse, n): return -10.0 * math.log10(max(total_mse / max(n, 1), 1e-12))

        summary = {
            # === HEADLINE: paper metrics (mean of per-image PSNR/SSIM, test split) ===
            "psnr_srgb": mean_metric(test_cams, "psnr_srgb"),
            "ssim_srgb": mean_metric(test_cams, "ssim_srgb"),
            "psnr_linear": mean_metric(test_cams, "psnr_linear"),
            "psnr_srgb_masked": mean_metric(test_cams, "psnr_srgb_masked"),
            "eval_holdout": bool(args.eval_holdout),
            "eval_holdout_stride": stride,
            "n_test_cams": len(test_cams),
            "n_train_cams": len(train_cams),
            # Same paper metric, other splits (train / all cams).
            "train_psnr_srgb": mean_metric(train_cams, "psnr_srgb"),
            "train_ssim_srgb": mean_metric(train_cams, "ssim_srgb"),
            "all_psnr_srgb": mean_metric(per_cam, "psnr_srgb"),
            "all_ssim_srgb": mean_metric(per_cam, "ssim_srgb"),
            # Legacy dB-of-mean-MSE (pessimistic pooling)
            "test_psnr_srgb_db_of_mean_mse": db_of_mean_mse(
                sum(c["mse_srgb"] for c in test_cams), len(test_cams)),
            "all_psnr_srgb_db_of_mean_mse": db_of_mean_mse(total_mse_srgb, n_cams),
            "all_psnr_linear_db_of_mean_mse": db_of_mean_mse(total_mse_lin, n_cams),
            "per_cam": per_cam,
            "n_cams": n_cams,
            "iters": args.iters,
            # Pure training-loop wall-clock (excludes this final metrics pass),
            # captured right after the loop with the GPU drained.
            "train_wall_s": train_wall_s,
            "train_it_per_s": completed_iters / max(train_wall_s, 1e-9),
            "final_n_gaussians": int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0]),
        }
        (run_dir / "final_psnr.json").write_text(json.dumps(summary, indent=2))

        ply_path = run_dir / "output.ply"
        save_ply(pipeline, ply_path)

        print(f"done in {time.time() - t_start:.1f}s")
        if args.eval_holdout:
            print(f"TEST PSNR sRGB (paper):  {summary['psnr_srgb']:6.2f} dB  "
                f"<- mean of per-image dB over {summary['n_test_cams']} held-out views "
                f"(compare to SAD-GS/3DGS table)")
            print(f"TEST SSIM sRGB (paper):  {summary['ssim_srgb']:6.4f}     "
                f"<- mean of per-image SSIM, {summary['n_test_cams']} held-out views")
            print(f"train PSNR/SSIM sRGB:    {summary['train_psnr_srgb']:6.2f} dB / "
                f"{summary['train_ssim_srgb']:.4f}  (mean per-image, {summary['n_train_cams']} train views)")
        else:
            print(f"PSNR sRGB (paper):       {summary['psnr_srgb']:6.2f} dB  "
                f"<- mean of per-image dB over all {n_cams} views (no holdout)")
            print(f"SSIM sRGB (paper):       {summary['ssim_srgb']:6.4f}     "
                f"<- mean of per-image SSIM over all {n_cams} views")
        print(f"PSNR sRGB masked:        {summary['psnr_srgb_masked']:6.2f} dB  <- honest silhouette (test split)")
        print(f"PSNR linear:             {summary['psnr_linear']:6.2f} dB  (test split, mean per-image)")
        print(f"  [legacy dB-of-mean-MSE, all cams: {summary['all_psnr_srgb_db_of_mean_mse']:.2f} dB — NOT the paper metric]")
        print(f"final gaussian count:    {summary['final_n_gaussians']}")
        print(f"saved ply:               {ply_path}")
        print(f"run dir: {run_dir}")
