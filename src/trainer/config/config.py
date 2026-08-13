"""Responsibility-specific immutable configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trainer.data.colmap import ColmapImageEntry


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
class IterationEvents:
    iteration: int
    densify: bool
    opacity_reset: bool
    structure: bool
    preview: bool
    render_all: bool
    log: bool
    eval: bool
    psnr_curve: bool
    morton: bool
    mip_after_densify: bool   # recompute the 3D filter INSIDE the densify block
    mip_periodic: bool         # standalone periodic recompute past densify_to

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
    mean: float = 1.6e-4
    scale: float = 5e-3
    rot: float = 1e-3
    opacity: float = 2.5e-2
    color: float = 2.5e-3  # in reality this is the SH0 LR
    sh_rest: float = 0.0025 / 20.0 #TODO play around more with a more optimal LR value for the sh rests