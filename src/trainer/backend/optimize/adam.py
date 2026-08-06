"""The Adam parameter-update step, extracted from the render pipeline.

``adamStep`` (shaders/optimizers/adam.slang) consumes the gradient buffers the
backward pass produced and updates the packed Gaussian parameters + m/v moments
in place. :class:`AdamOptimizer` owns that dispatch and the bias-correction step
counter, so the optimizer is a first-class object next to densification rather
than a method buried in ``Pipeline``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trainer.config.config import AdamLRs

# TODO: these should be matching the adam.slang code
_BETA1 = 0.9
_BETA2 = 0.999


class AdamOptimizer:
    """Owns the Adam step dispatch and its bias-correction step counter.

    The dispatch helpers and kernels are injected (bound from the pipeline that
    owns the command stream), so this class holds no device/pipeline internals
    beyond the buffers it reads and the two kernels it launches.
    """

    def __init__(self, *, dispatch, dispatch_indirect, prepare_args_kernel,
                 adam_kernel, clear_adam_state, buffers, max_splats: int, width: int, height: int,
                 adam_rest_kernel=None) -> None:
        self._dispatch = dispatch
        self._dispatch_indirect = dispatch_indirect
        self._k_prepare_args = prepare_args_kernel
        self._k_adam = adam_kernel
        self._k_clear_adam_state = clear_adam_state
        self._k_adam_rest = adam_rest_kernel
        self._buffers = buffers
        self._max_splats = max_splats
        self._width = width
        self._height = height
        self.step_count = 0

    def reset(self) -> None:
        self.step_count = 0

    def step(self, lrs: "AdamLRs") -> None:
        """Run one Adam parameter update over the active gaussians."""
        self.step_count += 1
        step = float(self.step_count)
        bc1 = 1.0 - _BETA1 ** step
        bc2 = 1.0 - _BETA2 ** step

        b = self._buffers

        self._dispatch(self._k_prepare_args, "adam_prepare_args",
            thread_count=[1, 1, 1],
            count_buf=b.num_gaussians_counter.storage,
            dispatch_args=b.adam_dispatch_args.storage,
            max_splats=self._max_splats,
        )

        self._dispatch_indirect(self._k_adam, "adam", b.adam_dispatch_args,
            gaussians=b.gaussians.storage,
            adam_state=b.adam_state.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            **b.grad_kwargs(),
            cluster_visible=b.cluster_visible.storage,
            cfg={
                "lr_mean": lrs.mean,
                "lr_scale": lrs.scale,
                "lr_rot": lrs.rot,
                "lr_opacity": lrs.opacity,
                "lr_color": lrs.color,
                "bias_correction1": bc1,
                "bias_correction2": bc2,
            },
            pixels_per_step=float(self._width * self._height),
        )

        if self._k_adam_rest is not None and b.sh_enabled:
            self._dispatch_indirect(self._k_adam_rest, "adam_rest", b.adam_dispatch_args,
                features_rest=b.features_rest.storage,
                features_rest_grad=b.features_rest_grad.storage,
                adam_rest=b.adam_rest_state.storage,
                num_gaussians_counter=b.num_gaussians_counter.storage,
                cluster_visible=b.cluster_visible.storage,
                lr_rest=lrs.sh_rest,
                bias_correction1=bc1,
                bias_correction2=bc2,
                pixels_per_step=float(self._width * self._height),
                rest_floats=b.sh.rest_floats,
            )

    def clear_state(self):
        self._dispatch(self._k_clear_adam_state, "clear_adam_state",
            thread_count=[self._max_splats, 1, 1],
            adam_state=self._buffers.adam_state.storage,
            max_splats=self._max_splats)
