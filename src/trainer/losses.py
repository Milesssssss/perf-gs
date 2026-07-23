"""Image-space loss recording, decoupled from the primitive rasterizer.

Extracts ``Pipeline._run_loss``'s SSIM / L1+L2 logic so the loss can be recorded
independently of HOW the pixels were produced (a future learned render head or a
different primitive reuses it unchanged). ``ImageLoss.record`` emits into a
caller batch or creates one runtime-owned transient batch for the complete
multi-pass loss. It returns the image-space gradient texture the backward uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import slangpy as spy

from trainer.config import LossConfig
from trainer.gpu import CommandBatch, GpuRuntime


@dataclass(frozen=True)
class LossResult:
    """Output of a loss pass: dL/d(rendered color), written per pixel into the
    image-gradient texture the backward rasterizer reads."""
    image_gradient: spy.Texture


class ImageLoss:
    """Records the photometric loss (SSIM+L1+L2 or plain L1+L2) that seeds the
    per-pixel color gradient. Owns the loss kernels + SSIM scratch textures; the
    rendered/target textures + config are supplied per call so a step can vary
    the target and loss weights freely."""

    def __init__(self, *, ssim_maps_kernel, ssim_grad_kernel, loss_kernel,
                 ssim_m1, ssim_m2, ssim_m3, image_gradient: spy.Texture,
                 width: int, height: int, runtime: GpuRuntime,
                 config: LossConfig) -> None:
        self._k_ssim_maps = ssim_maps_kernel
        self._k_ssim_grad = ssim_grad_kernel
        self._k_loss = loss_kernel
        self._ssim_m1 = ssim_m1
        self._ssim_m2 = ssim_m2
        self._ssim_m3 = ssim_m3
        self._image_gradient = image_gradient
        self._width = width
        self._height = height
        self._runtime = runtime
        self._config = config

    @staticmethod
    def _emit(batch: CommandBatch, kernel, marker: str, **kwargs) -> None:
        batch.dispatch(kernel, marker, **kwargs)

    @staticmethod
    def _group(batch: CommandBatch, label: str):
        return batch.marker_group(label)

    def record(self, batch: CommandBatch | None, *, rendered: spy.Texture,
               target: spy.Texture) -> LossResult:
        owned_batch = batch is None
        if batch is None:
            batch = self._runtime.command_batch("image_loss")
        config = self._config
        dims = spy.uint2(self._width, self._height)
        if config.ssim_weight > 0:
            # Two-pass SSIM: pass 1 builds the m1/m2/m3 back-projection maps,
            # pass 2 convolves them and blends with the L1 + alpha gradient.
            with self._group(batch, "ssim_maps"):
                self._emit(batch, self._k_ssim_maps, "ssim_maps",
                    thread_count=[self._width, self._height, 1],
                    output_image=rendered,
                    target_image=target,
                    ssim_m1=self._ssim_m1,
                    ssim_m2=self._ssim_m2,
                    ssim_m3=self._ssim_m3,
                    frame_dim=dims,
                )
            with self._group(batch, "ssim_grad"):
                self._emit(batch, self._k_ssim_grad, "ssim_grad",
                    thread_count=[self._width, self._height, 1],
                    output_image=rendered,
                    target_image=target,
                    ssim_m1=self._ssim_m1,
                    ssim_m2=self._ssim_m2,
                    ssim_m3=self._ssim_m3,
                    dL_dC=self._image_gradient,
                    frame_dim=dims,
                    lambda_ssim=config.ssim_weight,
                    w_alpha=config.alpha_weight,
                    lambda_l2=config.l2_weight,
                )
        else:
            with self._group(batch, "l1_l2_loss"):
                self._emit(batch, self._k_loss, "loss",
                    thread_count=[self._width, self._height, 1],
                    output_image=rendered,
                    target_image=target,
                    dL_dC=self._image_gradient,
                    frame_dim=dims,
                    w_alpha=config.alpha_weight,
                    lambda_l2=config.l2_weight,
                )
        if owned_batch:
            self._runtime.submit(batch)
        return LossResult(image_gradient=self._image_gradient)
