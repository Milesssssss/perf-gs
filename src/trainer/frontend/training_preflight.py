"""Setup-phase (pre-training) derivations for the training frontend.

Pure, GPU-free helpers lifted out of ``app.main()`` so the setup reads as a
sequence of named responsibilities instead of one long preamble:

  * :func:`resolve_holdout_split` — the 3DGS/SAD-GS held-out eval split,
  * :func:`compute_morton_bounds` — the Morton normalization AABB,
  * :func:`build_densify_configs` — the densify + structure config objects,
  * :func:`resolve_loss_and_lrs` — the SAD-GS-matched loss weights + base LRs.

None of these touch the GPU device/pipeline, so they stay trivially testable and
have no bearing on the delicate texture-upload / viewer-start ordering that
remains inline in ``main()``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from trainer.config.config import AdamLRs
from trainer.config.config_types import DensificationMode
from trainer.backend.densify.densify import StructureConfig
from trainer.backend.render.pipeline import DensifyConfig

if TYPE_CHECKING:
    import argparse


def resolve_holdout_split(
    args: "argparse.Namespace", n_cams: int
) -> tuple[int, list[int], list[int]]:
    """Compute the held-out eval split (paper protocol) and print it.

    3DGS/SAD-GS hold out every ``stride``-th camera (name-sorted; the loader
    already sorts by image name) and report PSNR on ONLY the held-out set. With
    ``--eval`` on, the loop samples exclusively from ``train_indices`` so test
    frames are never optimized against nor fed to the structure-densify eta
    accumulator. With ``--eval`` off, every camera is both trained and evaluated.

    Returns ``(stride, test_indices, train_indices)``.
    """
    stride = max(args.eval_holdout_stride, 1)
    if args.eval_holdout:
        test_indices = [i for i in range(n_cams) if i % stride == 0]
        train_indices = [i for i in range(n_cams) if i % stride != 0]
        print(f"[eval] holdout protocol ON (stride={stride}): "
              f"{len(train_indices)} train / {len(test_indices)} test cameras. "
              f"Training samples train views only; PSNR reported on test views "
              f"as mean-of-per-image dB (paper metric).")
    else:
        test_indices = list(range(n_cams))
        train_indices = list(range(n_cams))
        print(f"[eval] holdout protocol OFF: training on all {n_cams} cameras, "
              f"reporting over all cameras (internal metric).")
    return stride, test_indices, train_indices


def compute_morton_bounds(
    initial_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """AABB of the initial point cloud, padded 20% per axis.

    Used only when ``--morton-every > 0`` (Morton quantization needs a
    normalized coord frame). A stale bbox is fine — the shader saturates
    out-of-box means, so densification drift outside these bounds is a
    cache-locality miss, not a correctness bug.

    Returns ``(morton_world_min, morton_world_max)``.
    """
    means = initial_means.astype(np.float32).reshape(-1, 3)
    bbox_min = means.min(axis=0)
    bbox_max = means.max(axis=0)
    bbox_extent = np.maximum(bbox_max - bbox_min, 1e-6)
    return bbox_min - 0.2 * bbox_extent, bbox_max + 0.2 * bbox_extent


def build_densify_configs(
    args: "argparse.Namespace", extent: float
) -> tuple[DensifyConfig, StructureConfig]:
    """Construct the :class:`DensifyConfig` + :class:`StructureConfig` from CLI
    args and the scene extent (extent-scaled scale thresholds)."""
    cfg = DensifyConfig(
        t_opacity=args.t_opacity,
        t_big_scale=args.t_big_scale * extent,
        t_opacity_reset_value=args.opacity_reset_value,
        # SADGS RELATIVE opacity reset (opacity bundle ablation).
        reset_relative=args.sad_relative_reset,
        reset_decay=args.sad_reset_decay,
    )

    structure_cfg = StructureConfig(
        n_scoring_views=args.n_scoring_views,
        tau_split=args.sad_tau_split,
        tau_prune=args.sad_tau_prune,
        grad_threshold=args.sad_grad_threshold,
        split_power=args.sad_split_power,
        max_split_factor=args.sad_max_split_factor,
        ks_scale_power=args.sad_ks_scale_power,
        # Structure-pass opacity floor (0.005 baseline / 0.1 = SADGS reference).
        t_opacity=args.sad_structure_min_opacity,
        # Post-densify opacity cap (0 off / 0.8 = SADGS reference).
        opacity_cap=args.sad_opacity_cap,
        # clone (small) vs split (large) threshold = SADGS args.dense * extent.
        dense_scale=args.grow_scale3d * extent,
    )
    return cfg, structure_cfg


def resolve_loss_and_lrs(
    args: "argparse.Namespace",
) -> tuple[float, float, AdamLRs]:
    """Resolve the effective loss weights + base learning rates.

    When running structure-aware densification, optionally train with the EXACT
    loss + LRs SAD-GS uses (``--sad-match-loss-lrs``, on by default) so the only
    variable vs. their setup is our renderer. Otherwise use our own defaults.

    Returns ``(eff_lambda_l2, eff_lambda_ssim, base_lrs)``.
    """
    sad_match = bool(
        DensificationMode.from_args(args) == DensificationMode.STRUCTURE_AWARE
        and args.sad_match_loss_lrs)
    if sad_match:
        eff_lambda_l2 = args.sad_lambda_l2      # SADGS opt.lambda_l2 = 2.0
        # SSIM lambda: default 0.2 (SADGS opt.lambda_dssim). --lambda-ssim
        # overrides — any explicitly-passed value wins over the sad_match
        # default. This lets A/B experiments like "does turning SSIM off
        # change our convergence?" work without also flipping the LR schedule
        # and L2 term (which --sad-match-loss-lrs=0 would do).
        eff_lambda_ssim = args.lambda_ssim      # default 0.2 matches SADGS
        # SADGS per-parameter LRs (their opt.*_lr). mean/position is scheduled
        # separately below (position_lr_init already matches at 1.6e-4*extent).
        base_lrs = AdamLRs(scale=0.01, rot=0.002, opacity=0.05, color=0.0025)
        print(f"[SAD] matching SADGS loss + LRs: lambda_l2={eff_lambda_l2}, "
              f"lambda_ssim={eff_lambda_ssim}, scale_lr=0.01, rot_lr=0.002, "
              f"opacity_lr=0.05, sh0_lr=0.0025")
    else:
        eff_lambda_l2 = args.lambda_l2
        eff_lambda_ssim = args.lambda_ssim
        base_lrs = AdamLRs()                    # our defaults
    return eff_lambda_l2, eff_lambda_ssim, base_lrs
