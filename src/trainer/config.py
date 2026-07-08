"""Pipeline configuration objects.

``RenderConfig`` holds the render/loss knobs the trainer decides and the
pipeline reads deep inside its dispatches. Previously these were poked onto the
Pipeline instance after construction (``pipeline.z_near_cull = ...``); making
them a frozen config passed at construction puts ownership at the trainer
decision level and keeps the pipeline's per-dispatch reads immutable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderConfig:
    """Render + loss knobs, owned by the trainer, read by the pipeline.

    Defaults match the previous post-construction Pipeline defaults.
    """
    # Near-plane cull distance used by the bin kernel.
    z_near_cull: float = 0.01
    # Regularization weights (gsplat defaults: both 0; MCMC uses 0.001).
    opacity_reg: float = 0.0
    scale_reg: float = 0.0
    # Alpha loss weight. Set > 0 when target images have meaningful alpha.
    w_alpha: float = 0.0
    # SSIM loss mixing. 0 = pure L2, >0 = L1+SSIM blend (3DGS default 0.2).
    lambda_ssim: float = 0.2


@dataclass
class AdamLRs:
    """Adam learning rates, matching gsplat defaults:
        means_lr     = 1.6e-4
        scales_lr    = 5e-3
        quats_lr     = 1e-3
        opacities_lr = 2.5e-2
        sh0_lr       = 2.5e-3
    Only ``mean`` has LR decay applied (PLAN.md Q15).
    """
    mean: float = 1.6e-4
    scale: float = 5e-3
    rot: float = 1e-3
    opacity: float = 2.5e-2
    color: float = 2.5e-3  # in reality this is the SH0 LR
