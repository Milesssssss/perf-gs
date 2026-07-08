"""CPU-side data marshalling into the trainer's GPU buffers.

Separated from :mod:`trainer.buffers` (which owns GPU resource allocation and
the :class:`~trainer.buffers.GaussianBuffers` container): this module is the
host-side plumbing that packs numpy arrays into the packed Gaussian3D layout,
converts opacity probabilities to logits, and uploads. The dependency is
one-way — this module depends on ``GaussianBuffers``, not the reverse.

``buffers.py`` re-exports these functions for back-compat, so existing callers
can keep using ``buffers.write_gaussians(...)``.
"""

from __future__ import annotations

import numpy as np

from trainer.buffers import GAUSSIAN_FLOAT_COUNT, GaussianBuffers


def set_num_gaussians(buffers: GaussianBuffers, n: int) -> None:
    assert 0 <= n <= buffers.max_splats, (
        f"num_gaussians {n} out of range [0, {buffers.max_splats}]"
    )
    buffers.num_gaussians_counter.copy_from_numpy(np.array([n], dtype=np.uint32))


def write_gaussians(
    buffers: GaussianBuffers,
    means: np.ndarray,
    log_scales: np.ndarray,
    rot_quats: np.ndarray,
    opacities: np.ndarray,
    color_logits: np.ndarray,
) -> None:
    """``opacities`` is a probability in (0, 1); we convert to the
    pre-sigmoid logit the shader expects."""
    n = means.shape[0]
    assert means.shape == (n, 3), f"means shape {means.shape}"
    assert log_scales.shape == (n, 3), f"log_scales shape {log_scales.shape}"
    assert rot_quats.shape == (n, 4), f"rot_quats shape {rot_quats.shape}"
    assert opacities.shape == (n,), f"opacities shape {opacities.shape}"
    assert color_logits.shape == (n, 3), f"color_logits shape {color_logits.shape}"
    assert n <= buffers.max_splats, f"{n} gaussians exceeds max_splats {buffers.max_splats}"

    # Convert probability → logit once on CPU. 1e-5 floor keeps log finite
    # when a caller passes exactly 0 or 1.
    p = np.clip(opacities, 1e-5, 1.0 - 1e-5).astype(np.float32)
    opacity_logits = np.log(p / (1.0 - p))

    # Gaussian3D is GAUSSIAN_FLOAT_COUNT contiguous float32s with no padding:
    #   mean(3), log_scale(3), rot_quat(4), opacity(1), color_logit(3).
    # Build the whole block with vectorized numpy and upload in one
    # copy_from_numpy — the per-element cursor loop below built ~5*n
    # temporary spy.float3/float4 objects (≈90ms per 5k gaussians, seconds
    # after densification grows the count into the hundreds of thousands).
    block = np.empty((n, GAUSSIAN_FLOAT_COUNT), dtype=np.float32)
    block[:, 0:3] = means
    block[:, 3:6] = log_scales
    block[:, 6:10] = rot_quats
    block[:, 10] = opacity_logits
    block[:, 11:14] = color_logits
    buffers.gaussians.copy_from_numpy(np.ascontiguousarray(block.reshape(-1)))

    set_num_gaussians(buffers, n)


def write_gaussians_raw(buffers: GaussianBuffers, floats: np.ndarray, n: int) -> None:
    """Upload a snapshot already in the packed Gaussian3D layout
    (mean3, log_scale3, rot_quat4, opacity1, sh0_3) straight into the GPU
    buffer. Used by the live viewer, which receives snapshots from the
    trainer via the shared-memory splat stream — no per-field unpacking.

    ``floats`` must hold at least ``n * GAUSSIAN_FLOAT_COUNT`` float32 values."""
    assert n <= buffers.max_splats, f"{n} gaussians exceeds max_splats {buffers.max_splats}"
    n_floats = n * GAUSSIAN_FLOAT_COUNT
    block = np.ascontiguousarray(floats[:n_floats].astype(np.float32, copy=False))
    if n > 0:
        buffers.gaussians.copy_from_numpy(block)
    set_num_gaussians(buffers, n)


def zero_adam_state(buffers: GaussianBuffers) -> None:
    """Reset all Adam first/second-moment state to zero.

    The whole AdamState buffer is contiguous POD, so a single zero-filled
    copy_from_numpy clears it regardless of the internal field layout. The
    old per-element cursor loop ran over all max_splats (~820ms per 50k,
    i.e. ~80s at a 5M cap) on every set_gaussians_from_numpy call.
    """
    n_floats = buffers.adam_state.to_numpy().view(np.float32).size
    buffers.adam_state.copy_from_numpy(np.zeros(n_floats, dtype=np.float32))
