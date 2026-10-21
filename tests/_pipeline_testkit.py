# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Test-only Pipeline subclass.

These helpers exist purely to drive tests/probes — they split the fused
``train_step`` so tests can read intermediates, and marshal GPU buffers back to
CPU for assertions. They have no production caller, so they live here instead
of polluting the shipped ``trainer.Pipeline``. Tests that need them construct
``TestPipeline`` instead of ``Pipeline``; it is otherwise a drop-in.
"""

from __future__ import annotations

import numpy as np

from trainer.backend.memory.buffers import GRAD_SCALAR_COMPONENTS
from trainer.config.config import (
    AdamLRs, DensificationProperties, IterationContextualVariables, StbnVariables,
)
from trainer.data.dataset import Camera
from trainer.config.config_types import DensificationMode
from trainer.backend.render.pipeline import TrainingPipeline


class TestPipeline(TrainingPipeline):
    """Pipeline plus test-only introspection / step-splitting helpers."""

    # Not a pytest test class despite the ``Test`` prefix — it's a Pipeline
    # subclass. Tells pytest's collector to skip it (avoids a collection warning).
    __test__ = False

    def __init__(self, *args, **kwargs):
        # Numerical tests exercise both densifiers on the same small pipeline.
        kwargs.setdefault("densify_mode", DensificationMode.STRUCTURE_AWARE)
        super().__init__(*args, **kwargs)

    def train_step(self, camera: Camera, lrs: AdamLRs | None = None,
                   active_sh_degree: int | None = None, hardness_value: float = 1.0,
                   structure_tensor=None, structure_kwargs: dict | None = None,
                   stbn: StbnVariables | None = None) -> None:
        degree = self._active_sh_degree if active_sh_degree is None else int(active_sh_degree)
        super().train_step(camera, lrs=lrs if lrs is not None else AdamLRs(),
            contextual=IterationContextualVariables(
                hardness_value=hardness_value, sh_degree=degree,
                densification_variables=DensificationProperties(structure_tensor, structure_kwargs),
                contextual_stbn=stbn))

    def set_primitive_from_numpy(self, **snapshot) -> None:
        """Re-load a primitive snapshot into an already-built pipeline (resets
        Adam + SH state). Production primes once at construction
        (``Pipeline(initial_state=...)``); this in-place re-prime is a TEST-ONLY
        affordance so one pipeline can be reused across many finite-difference
        perturbations without rebuilding the device + kernels each time."""
        self._load_snapshot(**snapshot)

    def forward_and_loss(self, camera: Camera,
                         stbn: "StbnVariables | None" = None) -> None:
        """Run forward + loss only. Used by tests that read ``dL_dC``
        before the backward pass runs.

        No ``wait_for_idle`` here — callers that read back GPU data
        should call ``device.wait_for_idle()`` themselves, or use a
        helper like :meth:`render` / :meth:`read_output_image` which
        handle the sync. Tests that explicitly sample intermediates
        insert the wait at their own callsite.

        ``stbn`` drives sparse pixel supervision for the STBN tests; ``None``
        (the default) samples every pixel, which is what every pre-existing
        caller expects.
        """
        with self.recording("forward_and_loss"):
            self.rasterize(camera)
            self.record_loss(IterationContextualVariables(
                hardness_value=self._extinction_hardness,
                sh_degree=self._active_sh_degree,
                contextual_stbn=stbn,
            ))

    def step_split_for_active_band_test(self, camera: Camera, lrs: AdamLRs,
                                        adam_degree: int) -> None:
        """Compare active-band Adam without changing the forward/backward degree."""
        contextual = IterationContextualVariables(
            hardness_value=self._extinction_hardness, sh_degree=self._active_sh_degree)
        with self.recording("train_step"):
            self._clear_2d_grad_seeds()
            self.record_forward(camera, contextual)
            self.record_loss(contextual)
            self.record_backward(contextual)
            self._run_regularization()
            self._adam.step(lrs, adam_degree)

    def backward(self, structure_tensor=None, structure_kwargs: dict | None = None,
                 stbn: StbnVariables | None = None) -> None:
        self.record_backward(IterationContextualVariables(
            hardness_value=self._extinction_hardness, sh_degree=self._active_sh_degree,
            densification_variables=DensificationProperties(structure_tensor, structure_kwargs),
            contextual_stbn=stbn))

    def regularize(self) -> None:
        """Run ONLY the opacity/scale regularization pass (adds its gradient
        onto the grad buffers). Used to isolate the reg gradient in tests."""
        self._run_regularization()

    def read_gradients(self) -> dict[str, np.ndarray]:
        """Read back the 14 SoA int64 fixed-point grad buffers."""
        n = int(self.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])
        inv = 1.0 / 65536.0
        out: dict[str, np.ndarray] = {}
        for name in GRAD_SCALAR_COMPONENTS:
            raw = self.buffers.grad_buffers[name].to_numpy()
            ints = np.frombuffer(raw.tobytes(), dtype=np.int64)[:n]
            out[name] = ints.astype(np.float64) * inv
        return out

    def read_loss(self) -> float:
        """Read the L1 loss as a scalar. Computed on CPU from the
        output_image and target_image — the shader doesn't accumulate
        the scalar itself (we only needed dL/dC for the backward).

        Matches the L1 photometric objective in l1_l2_loss.slang: render and
        target are both sRGB and compared DIRECTLY (no linearization), so the
        finite-difference gradient tests stay consistent with the shader."""
        C = self.buffers.output_image.to_numpy()[..., :3]
        T = self.buffers.target_image.to_numpy()[..., :3]   # sRGB, no linearization
        # target_image is rgba8_unorm: readback is uint8 [0,255]; the shader
        # sees it normalized to [0,1], so match that here for the L1 compare.
        if T.dtype == np.uint8:
            T = T.astype(np.float32) / 255.0
        diff = C - T
        return float(np.sum(np.abs(diff)))

    def cap_opacity(self, cap_value: float = 0.8) -> None:
        """Cap per-gaussian opacity at cap_value. Softer alternative to full
        opacity reset — lets gaussians keep earned opacity up to the cap."""
        b = self.buffers
        self._dispatch(self.k_opacity_reset, "opacity_reset",
            thread_count=[self.max_splats, 1, 1],
            gaussians=b.gaussians.storage,
            adam_state=b.adam_state.storage,
            num_gaussians_counter=b.num_gaussians_counter.storage,
            reset_value=cap_value,
            mode=0,   # CLAMP (downward clamp to logit(cap_value))
        )
