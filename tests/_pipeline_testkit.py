"""Test-only Pipeline subclass.

These helpers exist purely to drive tests/probes — they split the fused
``train_step`` so tests can read intermediates, and marshal GPU buffers back to
CPU for assertions. They have no production caller, so they live here instead
of polluting the shipped ``trainer.Pipeline``. Tests that need them construct
``TestPipeline`` instead of ``Pipeline``; it is otherwise a drop-in.
"""

from __future__ import annotations

import numpy as np

from trainer.buffers import GRAD_SCALAR_COMPONENTS
from trainer.dataset import Camera
from trainer.pipeline import Pipeline


class TestPipeline(Pipeline):
    """Pipeline plus test-only introspection / step-splitting helpers."""

    # Not a pytest test class despite the ``Test`` prefix — it's a Pipeline
    # subclass. Tells pytest's collector to skip it (avoids a collection warning).
    __test__ = False

    def forward_and_loss(self, camera: Camera) -> None:
        """Run forward + loss only. Used by tests that read ``dL_dC``
        before the backward pass runs.

        No ``wait_for_idle`` here — callers that read back GPU data
        should call ``device.wait_for_idle()`` themselves, or use a
        helper like :meth:`render` / :meth:`read_output_image` which
        handle the sync. Tests that explicitly sample intermediates
        insert the wait at their own callsite.
        """
        with self.recording("forward_and_loss"):
            self.rasterize(camera)
            self._run_loss()

    def backward(self) -> None:
        """Run the backward rasterizer. Assumes forward + loss have run."""
        self._run_backward()

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
