"""CPU-side data marshalling into the trainer's GPU buffers.

Separated from :mod:`trainer.buffers` (which owns GPU resource allocation and
the :class:`~trainer.buffers.GaussianBuffers` container): this module is the
host-side plumbing that packs numpy arrays into the packed Gaussian3D layout,
converts opacity probabilities to logits, and uploads. The dependency is
one-way — this module depends on ``GaussianBuffers``, not the reverse.
"""

from __future__ import annotations

import numpy as np

from trainer.backend.memory.buffers import GAUSSIAN_FLOAT_COUNT, GaussianBuffers
from trainer.contracts.codegen import gaussian_numpy_dtype


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

    # Pack into the contract's structured Gaussian3D dtype (named fields, not
    # hand-coded scalar offsets), then upload as the flat float32 the shader
    # struct expects. One vectorized build + a single copy_from_numpy.
    block = np.zeros(n, dtype=gaussian_numpy_dtype())
    block["mean"] = means
    block["log_scale"] = log_scales
    block["rot_quat"] = rot_quats
    block["opacity"] = opacity_logits
    block["sh0"] = color_logits
    buffers.gaussians.copy_from_numpy(
        np.ascontiguousarray(block).view(np.float32).reshape(-1))

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


def sh_feature_uints(rest_floats: int) -> int:
    """uint32s per gaussian for the fp16-packed features_rest (2 coeffs/uint)."""
    return (rest_floats + 1) // 2


def pack_features_rest(feat: np.ndarray) -> np.ndarray:
    """``(N, rest_floats)`` float32 coefficients -> ``(N * feat_uints,)`` uint32,
    fp16-packed 2 scalars per uint (matches the shader's readHalfScalar)."""
    feat = np.ascontiguousarray(feat, dtype=np.float32)
    n, rf = feat.shape
    fu = sh_feature_uints(rf)
    padded = np.zeros((n, fu * 2), np.float32)
    padded[:, :rf] = feat
    h = padded.astype(np.float16).view(np.uint16).reshape(n, fu, 2).astype(np.uint32)
    return (h[..., 0] | (h[..., 1] << 16)).reshape(-1)


def unpack_features_rest(packed: np.ndarray, n: int, rest_floats: int) -> np.ndarray:
    """Inverse of :func:`pack_features_rest`: ``(N * feat_uints,)`` uint32 ->
    ``(N, rest_floats)`` float32."""
    fu = sh_feature_uints(rest_floats)
    p = np.ascontiguousarray(packed, dtype=np.uint32).reshape(n, fu)
    lo = (p & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (p >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
    out = np.empty((n, fu * 2), np.float32)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out[:, :rest_floats]
