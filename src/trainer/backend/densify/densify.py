# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Densification thresholds shared by the simple and structure-aware densifiers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DensifyConfig:
    """Shared simple-densify and opacity-reset thresholds."""
    t_opacity: float = 0.005        # prune if sigmoid(opacity) < this. MUST stay BELOW
                                    # t_opacity_reset_value (0.01): reset clamps opacity down
                                    # to the reset value, so a floor ABOVE it prunes every
                                    # splat the instant reset fires (total wipeout). 3DGS uses
                                    # prune 0.005 < reset 0.01 so reset splats survive and only
                                    # genuinely-dead ones decaying below 0.005 get culled.
    t_big_scale: float = 0.1        # prune if max_scale > this (world-space floater removal)
    t_opacity_reset_value: float = 0.01  # reset opacity to this value (CLAMP mode)
    # Opacity-reset mode. False (default): gsplat CLAMP — opacity <- min(opacity,
    # logit(t_opacity_reset_value)). True: SADGS RELATIVE decay — opacity <-
    # opacity * reset_decay (preserves strong/weak ordering; reset_decay is the
    # multiplier, e.g. 0.1). SADGS reset_opacity(opacity_reset_decay=0.1).
    reset_relative: bool = False
    reset_decay: float = 0.1        # RELATIVE-mode multiplier (used iff reset_relative)


@dataclass
class StructureConfig:
    """Structure-aware densification config (SAD-GS, arXiv 2604.28016).

    Matches SADGS train.py gates:
      split_mask = (high_ratio > split_ratio_threshold) & is_grad_high
      prune_mask = (low_ratio  > prune_ratio_threshold) & valid
    where high_ratio/low_ratio are the SCALAR (max-over-axes eta) view
    fractions, and the split factor n_k = ceil(eta_max_k ** split_power)."""
    n_scoring_views: int = 10        # views sampled per OFFLINE scoring step (tests/fallback)
    tau_split: float = 0.8           # split_ratio_threshold: high_ratio must exceed this
    tau_prune: float = 0.8           # prune_ratio_threshold: low_ratio must exceed this
    grad_threshold: float = 1e-5     # is_grad_high: avg 2D-grad magnitude floor (SADGS 1e-5)
    split_power: float = 0.5         # n_k = ceil(eta_max_k ** split_power); 0.5 = sqrt (paper)
    max_split_factor: int = 8        # per-axis cap on n_k (SADGS max_clones_per_axis=8)
    # Child scale along axis k = sigma_k / n_k**ks_scale_power (SADGS
    # scaling / ks**scale_power). 1.0 for Mip-NeRF 360 (the variance-preserving
    # choice paired with the sqrt(12) child spacing in split_anisotropic.slang);
    # 1.2 for Tanks & Temples / Deep Blending (SADGS ks_scale_power=1.2).
    ks_scale_power: float = 1.0
    t_opacity: float = 0.005
    # Post-densify opacity cap (SADGS gaussian_model.py:1046:
    # inverse_sigmoid(min(get_opacity, 0.8)) applied to ALL survivors after the
    # structure prune)
    opacity_cap: float = 0.8
    # Scale threshold separating clone (small) from split (large), = SADGS
    # args.dense * scene_extent. A triggered gaussian with max world-space scale
    # <= this is CLONED (exact duplicate); larger ones are anisotropically split.
    dense_scale: float = 0.0         # dense * extent, in world units; set by the caller
    detailed_diagnostics: bool = False  # full eta readback only when explicitly requested
