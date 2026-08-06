"""3DGS ``.ply`` format IO — load an existing point cloud, write gaussians out.

Co-locates the read and write of the 3DGS PLY layout so the field convention
(per-axis log scales, ``wxyz`` quaternion, pre-sigmoid opacity, SH band-0 DC
color) lives in one place. This module is pure (numpy + plyfile) and imports no
pipeline/GPU code, so it stays in the shared ``data`` layer;
:func:`trainer.backend.export.save_ply` reads the GPU buffers and delegates the
serialization here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

_PLY_REQUIRED_FIELDS = (
    "x", "y", "z",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
    "opacity",
    "f_dc_0", "f_dc_1", "f_dc_2",
)


def load_ply(path: str | Path) -> dict[str, np.ndarray]:
    """Load a 3DGS-format ``.ply`` into numpy arrays matching
    :func:`trainer.backend.memory.buffers.write_gaussians` inputs.

    The 3DGS convention:
    - ``x, y, z`` — world position
    - ``scale_0..2`` — log-scale per axis
    - ``rot_0..3`` — quaternion (w, x, y, z)
    - ``opacity`` — pre-sigmoid (inverse sigmoid applied at render)
    - ``f_dc_0..2`` — SH zeroth-order coefficient (DC term)

    Opacity in the ``.ply`` is pre-sigmoid; this loader returns the [0,1]
    probability (``write_gaussians`` converts back to logit before upload). The
    DC term maps to ``sh0`` directly (color = 0.5 + C0*sh0), no round-trip.
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

    # Optional higher SH bands. INRIA writes f_rest channel-major (all coeffs of
    # R, then G, then B); convert back to our coeff-major [coeff*3 + channel].
    rest_names = sorted(
        (nm for nm in available if nm.startswith("f_rest_")),
        key=lambda s: int(s.rsplit("_", 1)[-1]))
    features_rest = None
    if rest_names:
        fr = np.stack([verts[nm] for nm in rest_names], axis=-1).astype(np.float32)
        n, rest_floats = fr.shape
        features_rest = (fr.reshape(n, 3, rest_floats // 3)
                         .transpose(0, 2, 1).reshape(n, rest_floats))

    return {
        "means": means,
        "log_scales": log_scales,
        "rot_quats": rot_quats,
        "opacities": opacity.astype(np.float32),
        "color_logits": f_dc.astype(np.float32),
        "features_rest": features_rest,
    }


def write_gaussian_ply(gaussians: np.ndarray, path: str | Path,
                       features_rest: np.ndarray | None = None) -> None:
    """Write a structured Gaussian array (``gaussian_numpy_dtype``) as a
    3DGS-format ``.ply``. Inverse of :func:`load_ply`.

    Opacity is written as the pre-sigmoid logit (as the GPU holds it), and
    ``sh0`` is written directly as ``f_dc`` (color = 0.5 + C0*sh0 IS sRGB,
    matching the 3DGS PLY convention), so no color round-trip.

    ``features_rest`` (optional, shape ``(N, rest_floats)`` in our coeff-major
    ``[coeff*3 + channel]`` layout) adds the higher SH bands as ``f_rest_*``,
    written in the INRIA channel-major order (all coeffs of R, then G, then B)
    so standard 3DGS viewers infer the degree from the count. Omitted for the
    SH0 model (14-field PLY, unchanged).
    """
    mean = gaussians["mean"]
    log_scale = gaussians["log_scale"]
    rot_quat = gaussians["rot_quat"]
    opacity_logit = gaussians["opacity"].astype(np.float32)
    f_dc = gaussians["sh0"].astype(np.float32)

    n = len(gaussians)
    f_rest = None
    if features_rest is not None and features_rest.size > 0:
        rest_floats = features_rest.shape[1]
        # coeff-major [coeff*3 + channel] -> channel-major [channel, coeff].
        f_rest = (features_rest.reshape(n, rest_floats // 3, 3)
                  .transpose(0, 2, 1).reshape(n, rest_floats).astype(np.float32))

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    if f_rest is not None:
        dtype += [(f"f_rest_{i}", "f4") for i in range(f_rest.shape[1])]
    dtype += [
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    records = np.empty(len(gaussians), dtype=dtype)
    records["x"] = mean[:, 0]
    records["y"] = mean[:, 1]
    records["z"] = mean[:, 2]
    records["nx"] = 0.0
    records["ny"] = 0.0
    records["nz"] = 0.0
    records["f_dc_0"] = f_dc[:, 0]
    records["f_dc_1"] = f_dc[:, 1]
    records["f_dc_2"] = f_dc[:, 2]
    if f_rest is not None:
        for i in range(f_rest.shape[1]):
            records[f"f_rest_{i}"] = f_rest[:, i]
    records["opacity"] = opacity_logit
    records["scale_0"] = log_scale[:, 0]
    records["scale_1"] = log_scale[:, 1]
    records["scale_2"] = log_scale[:, 2]
    records["rot_0"] = rot_quat[:, 0]
    records["rot_1"] = rot_quat[:, 1]
    records["rot_2"] = rot_quat[:, 2]
    records["rot_3"] = rot_quat[:, 3]

    elem = PlyElement.describe(records, "vertex")
    PlyData([elem], text=False).write(str(path))
