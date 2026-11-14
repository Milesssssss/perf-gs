# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Responsibility-specific immutable configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    # Sum each splat's blend weight into the densify window (blend-weight pruning).
    track_blend_weight: bool = False
            

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
    """Spatiotemporal-blue-noise pixel subsampling policy (OFF at 1.0).

    ``backward_compacted`` records whether STBN_ENABLED was actually injected,
    rather than deriving it: the shader defines are fixed when the device is
    created, which in hub mode is before the panel that sets the rest exists, and
    a config disagreeing with the define would bind parameters the kernel never
    declared. With the mask on and this off the loss still zeroes unsampled
    pixels, so the gradient is the sparse one without the pixel-loop savings.
    """

    # Steady-state fraction rho of pixels each backward supervises.
    pixel_fraction: float = 1.0
    # Iterations spent ramping the density from 1.0 down to pixel_fraction.
    warmup_iters: int = 0
    backward_compacted: bool = False
    source: str = "nvidia"
    file_path: str = ""
    seed: int = 0
    # Draw a random slice each step (--stbn-independent) instead of the per-visit
    # temporal walk — the independent-per-step baseline for the temporal ablation.
    independent: bool = False
    # How a density picks pixels: "global" thresholds the whole slice (the
    # NVIDIA point set; per-tile counts vary), "tile_rank" keeps exactly k per
    # 8x8 tile (uniform work, but a different sampling process that measurably
    # degrades the blue mask AND flatters the white control). See
    # trainer.backend.render.stbn for the measured difference.
    select_mode: str = "global"

    @property
    def enabled(self) -> bool:
        return self.pixel_fraction < 1.0


@dataclass(frozen=True)
class AdaptiveStbnConfig:
    """Importance-guided sparse sampling policy (OFF by default).

    Strictly additive: with ``enabled=False`` nothing in this feature runs and
    every pre-existing regime (``full``, ``iid_white``, ``indep_blue``, ``stbn``)
    behaves exactly as before — the uniform STBN path remains the reference
    baseline. When it IS on, the run's density comes from a per-pixel probability
    map instead of a fixed rho, the loss runs DENSE and a separate pass applies
    the mask, and the backward reads a screen-sized table built each iteration.

    ``rho`` is not stored here: the budget is the sibling
    :class:`StbnConfig.pixel_fraction`, so an adaptive run and its uniform
    control are guaranteed to be density-matched by construction.

    See :mod:`trainer.backend.render.adaptive_stbn` for the math.
    """

    enabled: bool = False
    # Mixture weight between uniform exploration and gradient importance.
    # 0 = exactly uniform (and provably reduces to the existing mask), ->1 =
    # strongly importance-driven. Must stay < 1: alpha=1 admits p_i=0 pixels,
    # whose inverse-probability weight is undefined.
    alpha: float = 0.75
    # Smoothing scale of the importance field, in OUTPUT-IMAGE pixels. Not tied
    # to the 8x8 rasterizer tile or to the 128x128 STBN texture.
    sigma: float = 8.0
    # Resolution divisor the field is evaluated at (cost only; the Gaussian sigma
    # is compensated so the effective smoothing stays `sigma` full-res pixels).
    downsample: int = 4
    # Which uniform sequence the per-pixel threshold is compared against:
    # "iid" | "blue_independent" | "blue_temporal". See adaptive_stbn.UNIFORM_*.
    uniform: str = "blue_temporal"
    # False = the labelled DIAGNOSTIC control (adaptive_stbn_biased): sample from
    # p_i but omit the 1/p_i weight, i.e. deliberately optimize an error-weighted
    # objective. Never the proposed method.
    unbiased: bool = True


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
    blend_prune: bool = False  # this densify tick also prunes by blend weight

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
    # Per-step entropy for the adaptive path's ``iid`` uniform mode, which needs a
    # FRESH independent field each visit rather than a slice of a fixed volume.
    # Drawn from the same --stbn-seed stream as slice_index, so the mask stays
    # reproducible and is still decoupled from the camera draw order. Ignored by
    # every other mode, including all four pre-existing regimes.
    sample_seed: int = 0


@dataclass(frozen=True)
class IterationContextualVariables:
    hardness_value: float
    sh_degree: int
    densification_variables: DensificationProperties | None = None
    contextual_stbn: StbnVariables | None = None


@dataclass
class AdamLRs:
    mean: float = 1.6e-4
    scale: float = 5e-3
    rot: float = 1e-3
    opacity: float = 2.5e-2
    color: float = 2.5e-3  # in reality this is the SH0 LR
    sh_rest: float = 0.0025 / 20.0 #TODO play around more with a more optimal LR value for the sh rests
