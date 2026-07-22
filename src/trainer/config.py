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
    # SSIM loss mixing. 0 = pure L1, >0 = L1+SSIM blend (3DGS default 0.2).
    lambda_ssim: float = 0.2
    # Additive L2 term weight (SADGS lambda_l2). 0 = pure L1(+SSIM); SADGS
    # trains with 2.0. Applied on BOTH loss paths (l1_l2_loss + ssim_loss).
    lambda_l2: float = 0.0
    # Cluster-Cull-Compact (Laine 2011, adapted for 3DGS). When True, each
    # _run_forward rebuilds a per-cluster AABB (128 splats per cluster,
    # Morton-adjacent) and skips clusters whose AABB is fully outside the
    # view frustum. Mask-in-bin_count design: no compaction, no slot remap;
    # backward and Adam are entirely unaware. Requires --morton-every > 0
    # for the clusters to be spatially coherent; without Morton sort a
    # cluster's 128 splats are random across the scene, so the AABB
    # covers the whole scene and never culls anything.
    cluster_cull: bool = True
    # Mip-Splatting (Yu et al., CVPR 2024). When True, the forward folds the
    # per-gaussian 3D smoothing filter (buffers.filter_3D, via set_cameras +
    # compute_3d_filter) into the projected covariance and multiplies alpha by
    # the coef3d*coef2d compensation. kernel_size is the 2D low-pass dilation:
    # 0.3 (vanilla, no compensation) when off; 0.1 (mip canonical) when on.
    # OFF is bit-identical to the pre-mip renderer (filter=0, kernel_size=0.3,
    # compensation=1).
    mip_enabled: bool = False
    kernel_size: float = 0.3
    # Compositor extinction hardness k (shaders/core/compositor.slang). 1.0 =
    # standard alpha-over (bit-identical); k>1 sharpens the per-splat extinction
    # to (1-alpha)^k so transmittance decays faster => stronger occlusion culling
    # (earlier early-out in forward AND backward) at a quality tradeoff. Passed as
    # a shader uniform, so it is tunable per run with no recompile.
    extinction_hardness: float = 1.0


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
