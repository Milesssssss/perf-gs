# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Photometric loss: writes dL/dC for the backward pass."""

from __future__ import annotations

import slangpy as spy

from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import LossResources
from trainer.config.config import LossConfig


class ImageLoss:
    """SSIM + L1 (+ L2) when ``ssim_weight > 0``, otherwise L1 + L2.

    The STBN mask applies here: unsampled pixels get zero gradient, sampled ones are reweighted.
    """

    KERNELS = {
        "loss": ("losses/l1_l2_loss.slang", "computeLossGrad"),
        "ssim_maps": ("losses/ssim_loss.slang", "ssimComputeMaps"),
        "ssim_grad": ("losses/ssim_loss.slang", "ssimGradient"),
    }

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary, *,
                 resources: LossResources, width: int, height: int,
                 config: LossConfig) -> None:
        self._commands = commands
        self._k = kernels.load(self.KERNELS)
        self._r = resources
        self._width = width
        self._height = height
        self.config = config

    @property
    def image_gradient(self) -> spy.Texture:
        return self._r.image_gradient

    def record(self, *, rendered: spy.Texture, target: spy.Texture, stbn_kwargs: dict) -> None:
        commands, config, r = self._commands, self.config, self._r
        threads = [self._width, self._height, 1]
        dims = spy.uint2(self._width, self._height)
        if config.ssim_weight > 0:
            with commands.marker_group("ssim_maps"):
                commands.dispatch(self._k.ssim_maps, "ssim_maps",
                    thread_count=threads,
                    output_image=rendered,
                    target_image=target,
                    ssim_m1=r.ssim_m1,
                    ssim_m2=r.ssim_m2,
                    ssim_m3=r.ssim_m3,
                    frame_dim=dims)
            with commands.marker_group("ssim_grad"):
                commands.dispatch(self._k.ssim_grad, "ssim_grad",
                    thread_count=threads,
                    output_image=rendered,
                    target_image=target,
                    ssim_m1=r.ssim_m1,
                    ssim_m2=r.ssim_m2,
                    ssim_m3=r.ssim_m3,
                    dL_dC=r.image_gradient,
                    frame_dim=dims,
                    lambda_ssim=config.ssim_weight,
                    w_alpha=config.alpha_weight,
                    lambda_l2=config.l2_weight,
                    **stbn_kwargs)
        else:
            with commands.marker_group("l1_l2_loss"):
                commands.dispatch(self._k.loss, "loss",
                    thread_count=threads,
                    output_image=rendered,
                    target_image=target,
                    dL_dC=r.image_gradient,
                    frame_dim=dims,
                    w_alpha=config.alpha_weight,
                    lambda_l2=config.l2_weight,
                    **stbn_kwargs)
