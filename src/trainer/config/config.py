"""Responsibility-specific immutable configuration objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RasterConfig:
    """Projection, binning, culling, and Mip-Splatting policy."""

    z_near_cull: float = 0.01
    cluster_cull: bool = True
    mip_enabled: bool = False
    kernel_size: float = 0.3


@dataclass(frozen=True)
class AppearanceConfig:
    """View-dependent color (spherical-harmonics) policy.

    ``sh_degree == 0`` (default) is the original view-independent SH band-0 (DC)
    color: no rest coefficients are allocated and the render path is unchanged.
    ``1..3`` enables view-dependent bands (classic 3DGS), activated progressively
    one band every ``band_interval`` iterations (``oneupSHdegree``).
    """

    sh_degree: int = 0
    band_interval: int = 1000

    @property
    def enabled(self) -> bool:
        return self.sh_degree > 0


@dataclass(frozen=True)
class CompositorConfig:
    """Front-to-back extinction/compositing policy."""

    extinction_hardness: float = 1.0


@dataclass(frozen=True)
class LossConfig:
    """Image-space objective weights owned by :class:`ImageLoss`."""

    alpha_weight: float = 0.0
    ssim_weight: float = 0.2
    l2_weight: float = 0.0


@dataclass(frozen=True)
class RegularizationConfig:
    """Per-primitive opacity/scale regularization weights."""

    opacity_weight: float = 0.0
    scale_weight: float = 0.0


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
    # Higher-band SH LR. None => color / 20 (INRIA features_rest = features_dc/20).
    sh_rest: float | None = None

    @property
    def sh_rest_lr(self) -> float:
        return self.sh_rest if self.sh_rest is not None else self.color / 20.0
