"""Responsibility-specific immutable configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trainer.data.colmap import ColmapImageEntry
from trainer.frontend.schedule import IterationEvents


@dataclass(frozen=True)
class RasterConfig:
    """Projection, binning, culling, and Mip-Splatting policy."""

    z_near_cull: float = 0.01
    cluster_cull: bool = True
    mip_enabled: bool = False
    kernel_size: float = 0.3


@dataclass(frozen=True)
class AppearanceConfig:

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
class DensificationProperties:
    structure_tensor_texture: Any = None
    structure_kwargs: dict[str, Any] | None = None
            

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


@dataclass(frozen=True)
class IterationContextualVariables:
    hardness_value: float
    sh_degree: int
    densification_variables: DensificationProperties | None = None

@dataclass(frozen=True)
class IterationContext:
    step_number: int
    camera_index: int
    camera_entry: ColmapImageEntry
    learning_rates: AdamLRs
    events: IterationEvents
    iteration_contextual_variables: IterationContextualVariables

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
    sh_rest: float = 0.0025 / 20.0 #TODO lr is too low for the 3k iters, fix this