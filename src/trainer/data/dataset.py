"""Camera model and view/projection matrix helpers.

``Camera`` dataclass + builders for the view and projection matrices, plus the
SH band-0 color constant. 3DGS ``.ply`` IO lives in :mod:`trainer.data.ply`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    view_matrix: np.ndarray   # (4, 4) float32, row-major, world -> view
    proj_matrix: np.ndarray   # (4, 4) float32, row-major, view -> clip
    image_size: tuple[int, int]  # (width, height) in pixels

    def __post_init__(self) -> None:
        assert self.view_matrix.shape == (4, 4), self.view_matrix.shape
        assert self.proj_matrix.shape == (4, 4), self.proj_matrix.shape
        assert self.view_matrix.dtype == np.float32
        assert self.proj_matrix.dtype == np.float32
        assert len(self.image_size) == 2 and all(isinstance(s, int) and s > 0 for s in self.image_size)


def build_view_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed look-at view matrix (world -> view).

    Convention: camera looks down +Z in view space (positive view-space depth
    is in front of the camera). This matches the shader's ``depth_vs > 0``
    visibility cull.
    """
    eye = np.asarray(eye, dtype=np.float32).reshape(3)
    target = np.asarray(target, dtype=np.float32).reshape(3)
    up = np.asarray(up, dtype=np.float32).reshape(3)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_ortho = np.cross(right, forward)

    # Row-major R with rows = basis vectors, translation = -R @ eye.
    # World point p -> view point v = R @ (p - eye).
    R = np.stack([right, up_ortho, forward], axis=0).astype(np.float32)
    t = -R @ eye

    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def build_projection_matrix(
    fovy_radians: float,
    aspect: float,
    z_near: float,
    z_far: float,
) -> np.ndarray:
    """Right-handed pinhole perspective, mapping view-space +Z into clip
    space such that ``mean_clip.w > 0`` for visible points and NDC xy lies in
    ``[-1, 1]``.

    The Jacobian in ``projectGaussian3DTo2D`` assumes ``P[0][0]`` and
    ``P[1][1]`` are the pixel-space focal lengths scaled by
    ``2 / frame_dim``; i.e. standard ``cot(fovy/2)`` / aspect form.
    """
    f = 1.0 / np.tan(0.5 * fovy_radians)

    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = z_far / (z_far - z_near)
    M[2, 3] = -z_far * z_near / (z_far - z_near)
    M[3, 2] = 1.0
    return M


def simple_camera(
    width: int,
    height: int,
    eye: tuple[float, float, float],
    target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: tuple[float, float, float] = (0.0, 1.0, 0.0),
    fovy_degrees: float = 60.0,
    z_near: float = 0.01,
    z_far: float = 100.0,
) -> Camera:
    """Convenience factory for hand-authored cameras."""
    view = build_view_matrix(np.array(eye), np.array(target), np.array(up))
    proj = build_projection_matrix(
        np.deg2rad(fovy_degrees), width / height, z_near, z_far
    )
    return Camera(view_matrix=view, proj_matrix=proj, image_size=(width, height))


# SH DC term -> linear RGB: the 3DGS format stores the zeroth SH coefficient
# (constant term) which is related to the linear RGB color by a factor
# derived from the spherical harmonic basis normalization:
#   SH_C0 = 0.5 * sqrt(1 / pi) ≈ 0.28209479
# The linear RGB from just the DC term is: rgb = 0.5 + SH_C0 * f_dc.
# The trainer's color param IS this band-0 coefficient (sh0), rendered with
# the same affine map (color = 0.5 + SH_C0 * sh0), so f_dc maps to sh0 as an
# identity — no sigmoid activation.
_SH_C0 = 0.28209479177387814
