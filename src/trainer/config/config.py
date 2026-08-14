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
class StbnConfig:
    """Spatiotemporal-blue-noise pixel subsampling policy (OFF by default).

    ``pixel_fraction`` is the steady-state fraction rho of pixels each backward
    supervises; ``1.0`` means "every pixel", i.e. the feature disabled and the
    pipeline bit-identical to a build without it. See
    :mod:`trainer.backend.render.stbn` for the mask and
    docs/stbn_pixel_sampling.md for the whole design.

    Everything here is a runtime uniform except the two compile-time facts, and
    those are recorded rather than derived because the SHADER DEFINES are fixed
    when the device is created — which in hub mode is before the panel that sets
    the rest exists. A config whose ``backward_compacted`` disagreed with the
    injected define would bind parameters the shader never declared.
    """

    # Steady-state fraction rho of pixels each backward supervises. 1.0 = the
    # feature is off and the pipeline is bit-identical to a build without it.
    pixel_fraction: float = 1.0
    # Iterations spent ramping the density from 1.0 down to pixel_fraction.
    # Sparse supervision is hardest on exactly the thin structures early
    # densification is trying to create, so starting dense is the cheap hedge.
    warmup_iters: int = 0
    # True when STBN_ENABLED was injected, i.e. the backward walks the compacted
    # visit list. Deliberately separable from :attr:`enabled`: with the mask on
    # and this off the loss still zeroes unsampled pixels, so the gradient is the
    # sparse one while the backward keeps its exact full-pixel loop. That is a
    # correct-but-slow REFERENCE configuration, and what the equivalence test
    # compares the fast path against.
    backward_compacted: bool = False
    source: str = "nvidia"
    file_path: str = ""
    seed: int = 0

    @property
    def enabled(self) -> bool:
        return self.pixel_fraction < 1.0

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
class StbnVariables:
    """The per-iteration half of STBN sampling: WHICH mask slice this step uses
    and HOW MANY pixels per tile it samples.

    ``slice_index`` is the drawn camera's VISIT count plus that camera's fixed
    phase — deliberately not the training iteration. The mask is built so that
    the values at one pixel are well distributed along its temporal axis, and
    the sequence that axis should track is "successive evaluations of the same
    image"; indexing it by iteration instead would hand a given camera an
    essentially arbitrary slice every time. The per-camera phase then keeps every
    camera from marching through the same slices in lockstep.

    ``pixels_per_tile`` == TILE_AREA means "sample everything", i.e. disabled.
    """
    slice_index: int = 0
    pixels_per_tile: int = 0    # 0 => resolved to TILE_AREA (disabled) downstream


@dataclass(frozen=True)
class IterationContextualVariables:
    hardness_value: float
    sh_degree: int
    densification_variables: DensificationProperties | None = None
    stbn: StbnVariables | None = None

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