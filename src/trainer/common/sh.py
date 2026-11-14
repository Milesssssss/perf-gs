# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Spherical-harmonics band math and the fp16 packing of the rest coefficients;
band limits mirror ``shaders/primitives/gaussian/sh_layout.slang``."""

from __future__ import annotations

import numpy as np

SH_MAX_DEGREE = 3


def sh_rest_coeffs(degree: int) -> int:
    """Rest coefficients per channel (excluding the DC term) at ``degree``."""
    if not 0 <= degree <= SH_MAX_DEGREE:
        raise ValueError(f"sh degree {degree} out of range [0, {SH_MAX_DEGREE}]")
    return (degree + 1) * (degree + 1) - 1


SH_REST_COEFFS_MAX = sh_rest_coeffs(SH_MAX_DEGREE)
SH_REST_FLOATS_MAX = SH_REST_COEFFS_MAX * 3


def sh_degree_from_rest_floats(rest_floats: int) -> int:
    """Inverse of ``3 * sh_rest_coeffs(degree)``."""
    if rest_floats <= 0:
        return 0
    degree = int(round((rest_floats // 3 + 1) ** 0.5)) - 1
    return max(0, min(degree, SH_MAX_DEGREE))


def sh_feature_uints(rest_floats: int) -> int:
    """uint32 words per splat for fp16-packed rest coefficients (two per word)."""
    return (rest_floats + 1) // 2


def pack_features_rest(features: np.ndarray) -> np.ndarray:
    """``(N, rest_floats)`` float32 -> flat uint32, two fp16 per word (shader readHalfScalar)."""
    features = np.ascontiguousarray(features, dtype=np.float32)
    n, rest_floats = features.shape
    words = sh_feature_uints(rest_floats)
    padded = np.zeros((n, words * 2), np.float32)
    padded[:, :rest_floats] = features
    halves = padded.astype(np.float16).view(np.uint16).reshape(n, words, 2).astype(np.uint32)
    return (halves[..., 0] | (halves[..., 1] << 16)).reshape(-1)


def unpack_features_rest(packed: np.ndarray, n: int, rest_floats: int) -> np.ndarray:
    """Inverse of :func:`pack_features_rest`."""
    words = np.ascontiguousarray(packed, dtype=np.uint32).reshape(n, sh_feature_uints(rest_floats))
    lo = (words & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float32)
    hi = (words >> 16).astype(np.uint16).view(np.float16).astype(np.float32)
    out = np.empty((n, words.shape[1] * 2), np.float32)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out[:, :rest_floats]
