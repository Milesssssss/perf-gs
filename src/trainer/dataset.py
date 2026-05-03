"""Dataset and camera helpers.

v0 (M2 scope): ``Camera`` dataclass + helpers for building the view and
projection matrices; ``load_ply`` for 3DGS-format point clouds. Image
loading is a lazy stub per PLAN.md Q8.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData


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


class Dataset:
    """Lazy-per-call dataset stub (PLAN.md Q8).

    v0 does not train, so image loading is unimplemented. The shape of the
    API is frozen now so v1 can fill it in without restructuring callers.
    """

    def __init__(self, cameras: list[Camera]):
        self.cameras = cameras

    def load_image(self, idx: int) -> np.ndarray:
        raise NotImplementedError("image loading arrives with v1 training")


_PLY_REQUIRED_FIELDS = (
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "opacity",
    "f_dc_0", "f_dc_1", "f_dc_2",
)

# SH DC term -> linear RGB: the 3DGS format stores the zeroth SH coefficient
# (constant term) which is related to the linear RGB color by a factor
# derived from the spherical harmonic basis normalization:
#   SH_C0 = 0.5 * sqrt(1 / pi) ≈ 0.28209479
# The linear RGB from just the DC term is: rgb = 0.5 + SH_C0 * f_dc.
# For the v0 plain-RGB trainer (sigmoid activation, no SH), we convert to
# a color_logit that reproduces the same linear RGB via sigmoid.
_SH_C0 = 0.28209479177387814


def load_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Load a 3DGS-format ``.ply`` into numpy arrays matching
    :func:`trainer.buffers.write_gaussians` inputs.

    The 3DGS convention:
    - ``x, y, z`` — world position
    - ``scale_0..2`` — log-scale per axis
    - ``rot_0..3`` — quaternion (w, x, y, z)
    - ``opacity`` — pre-sigmoid (inverse sigmoid applied at render)
    - ``f_dc_0..2`` — SH zeroth-order coefficient (DC term)

    We do not handle higher-order SH bands in v0 (PLAN.md Q14). The DC
    term is converted to a plain-RGB color_logit so sigmoid(color_logit)
    approximates the same linear RGB as ``0.5 + SH_C0 * f_dc``.

    Opacity in the .ply is pre-sigmoid (i.e. ``sigmoid(opacity)`` gives
    the real [0,1] opacity). We apply the sigmoid here so the GPU struct
    holds ready-to-use opacity in [0,1] — matching M1's shader
    expectation. v1 can revisit this if training wants pre-activation
    storage for numerical reasons.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ply not found: {path}")

    ply = PlyData.read(str(path))
    if "vertex" not in [el.name for el in ply.elements]:
        raise ValueError(f"ply {path} has no 'vertex' element")
    verts = ply["vertex"].data

    available = set(verts.dtype.names)
    missing = [f for f in _PLY_REQUIRED_FIELDS if f not in available]
    if missing:
        raise ValueError(f"ply {path} missing required fields: {missing}")

    n = len(verts)
    means = np.stack([verts["x"], verts["y"], verts["z"]], axis=-1).astype(np.float32)
    log_scales = np.stack(
        [verts["scale_0"], verts["scale_1"], verts["scale_2"]], axis=-1
    ).astype(np.float32)
    rot_quats = np.stack(
        [verts["rot_0"], verts["rot_1"], verts["rot_2"], verts["rot_3"]], axis=-1
    ).astype(np.float32)

    if not np.all(np.isfinite(means)):
        raise ValueError(f"ply {path}: non-finite means")
    if not np.all(np.isfinite(log_scales)):
        raise ValueError(f"ply {path}: non-finite log_scales")
    if not np.all(np.isfinite(rot_quats)):
        raise ValueError(f"ply {path}: non-finite rot_quats")

    quat_norms = np.linalg.norm(rot_quats, axis=-1)
    if np.any(quat_norms < 1e-8):
        raise ValueError(f"ply {path}: {(quat_norms < 1e-8).sum()} zero quaternions")

    opacity_logit = np.asarray(verts["opacity"], dtype=np.float32)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))

    f_dc = np.stack(
        [verts["f_dc_0"], verts["f_dc_1"], verts["f_dc_2"]], axis=-1
    ).astype(np.float32)
    linear_rgb = 0.5 + _SH_C0 * f_dc
    # Clip strictly away from {0, 1} so log(rgb / (1 - rgb)) is finite.
    # 1e-5 sits at the float32 numerical floor, not a magnitude guess.
    linear_rgb = np.clip(linear_rgb, 1e-5, 1.0 - 1e-5)
    color_logit = np.log(linear_rgb / (1.0 - linear_rgb))

    return {
        "means": means,
        "log_scales": log_scales,
        "rot_quats": rot_quats,
        "opacities": opacity.astype(np.float32),
        "color_logits": color_logit.astype(np.float32),
    }
