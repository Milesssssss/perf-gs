# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

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
    eye = np.asarray(eye, dtype=np.float32).reshape(3)
    target = np.asarray(target, dtype=np.float32).reshape(3)
    up = np.asarray(up, dtype=np.float32).reshape(3)

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up_ortho = np.cross(right, forward)

    # Rows of R are the basis vectors; p -> R @ (p - eye), so t = -R @ eye.
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
