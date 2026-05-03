"""COLMAP text-format loader.

Parses a COLMAP sparse reconstruction in text form
(``cameras.txt``, ``images.txt``, ``points3D.txt``) into our trainer's
data structures. Binary ``.bin`` format is not supported — v1's data
generation tool emits text, and a binary parser is v2+ work.

COLMAP conventions:
- Image y-axis points DOWN (row 0 = top of image).
- Camera looks down +Z in camera frame.
- Quaternion is (w, x, y, z) representing a world -> camera rotation.
- Translation ``t`` is the world -> camera translation (post-rotation).
- Pinhole camera: ``SIMPLE_PINHOLE`` has one focal length ``f`` (fx = fy
  = f). ``PINHOLE`` has separate ``fx, fy``.

Our trainer conventions:
- View matrix row-major, maps world point to camera-space point.
- Projection maps camera point to clip space with ``P[0][0] = 2fx/W``,
  ``P[1][1] = 2fy/H`` (NDC y up — we flip sign below).
- Quaternion (w, x, y, z).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trainer.dataset import Camera


@dataclass
class ColmapImageEntry:
    image_id: int
    image_name: str
    camera: Camera


@dataclass
class ColmapDataset:
    images: list[ColmapImageEntry]
    image_dir: Path
    initial_points: dict[str, np.ndarray]  # matches write_gaussians signature


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert (w, x, y, z) quaternion to 3x3 rotation matrix.

    Same formula as our shader's ``matrixFromQuaternion`` for
    consistency; operates in numpy for CPU use.
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _parse_cameras(path: Path) -> dict[int, dict]:
    """Return dict of camera_id -> {model, width, height, fx, fy, cx, cy}."""
    cameras: dict[int, dict] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            if model == "SIMPLE_PINHOLE":
                f_, cx, cy = params
                fx = fy = f_
            elif model == "PINHOLE":
                fx, fy, cx, cy = params
            else:
                raise ValueError(f"unsupported COLMAP camera model {model}")
            cameras[cam_id] = {
                "model": model, "width": width, "height": height,
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            }
    return cameras


def _parse_images(path: Path) -> list[dict]:
    """Return list of dicts with keys
    {image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name}.
    Ignores the per-image feature-point lines (every second data line)."""
    entries: list[dict] = []
    with path.open("r") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    # Alternating: metadata, feature-points, metadata, feature-points, ...
    for meta in lines[::2]:
        parts = meta.split()
        entries.append({
            "image_id": int(parts[0]),
            "qw": float(parts[1]), "qx": float(parts[2]),
            "qy": float(parts[3]), "qz": float(parts[4]),
            "tx": float(parts[5]), "ty": float(parts[6]), "tz": float(parts[7]),
            "camera_id": int(parts[8]),
            "name": parts[9],
        })
    return entries


def _parse_points3d(path: Path) -> np.ndarray:
    """Return (N, 3) float32 array of world-space point positions."""
    pts: list[list[float]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(pts, dtype=np.float32)


def _view_matrix(qw: float, qx: float, qy: float, qz: float,
                 tx: float, ty: float, tz: float) -> np.ndarray:
    """Build a row-major 4x4 view matrix from COLMAP's world->camera
    quaternion + translation.

    COLMAP stores R and t such that ``x_cam = R @ x_world + t``.
    So the view matrix is:
        [ R | t ]
        [ 0 | 1 ]
    """
    R = _quat_wxyz_to_rotmat(np.array([qw, qx, qy, qz]))
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, tz]
    return M


def _projection_matrix(fx: float, fy: float, cx: float, cy: float,
                       width: int, height: int,
                       z_near: float = 0.01, z_far: float = 100.0) -> np.ndarray:
    """Build a row-major 4x4 projection matrix mapping view space to clip.

    Given pinhole intrinsics ``(fx, fy, cx, cy)`` for an image of size
    ``(width, height)``, produces the clip-space matrix such that:
      P[0][0] = 2 fx / W           # matches our shader's focal recovery
      P[1][1] = 2 fy / H
      P[0][2] = 2 cx / W - 1       # NDC x offset for off-center principal
      P[1][2] = 2 cy / H - 1       # NDC y offset
      P[2][2] = far / (far - near)
      P[2][3] = -far*near / (far - near)
      P[3][2] = 1

    Both COLMAP pixel-y and our shader's ``mean_2d.y = uv * H`` pixel
    space are row-0-at-top (y down). No sign flip on row 1 is needed
    because the shader's final ``uv * H`` step already inverts NDC y
    into pixel-row convention. Verified against the FOV matrix builder
    in dataset.py::build_projection_matrix which has the same sign.
    """
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 2.0 * fx / width
    M[1, 1] = 2.0 * fy / height
    M[0, 2] = 2.0 * cx / width - 1.0
    M[1, 2] = 2.0 * cy / height - 1.0
    M[2, 2] = z_far / (z_far - z_near)
    M[2, 3] = -z_far * z_near / (z_far - z_near)
    M[3, 2] = 1.0
    return M


def load_colmap(
    scene_dir: Path | str,
    width: int | None = None,
    height: int | None = None,
) -> ColmapDataset:
    """Load a COLMAP scene from ``<scene_dir>/sparse/0/text/``.

    ``width`` / ``height`` are the *target render resolution* — if they
    differ from the COLMAP camera's native width/height, the principal
    point and focal length scale linearly. All cameras in the scene are
    assumed to share the same native resolution (we assert this).
    """
    scene_dir = Path(scene_dir)
    text_dir = scene_dir / "sparse" / "0" / "text"
    img_dir = scene_dir / "images"

    cams = _parse_cameras(text_dir / "cameras.txt")
    imgs = _parse_images(text_dir / "images.txt")
    pts3d = _parse_points3d(text_dir / "points3D.txt")

    # Sanity: all cameras same native resolution.
    native_sizes = {(c["width"], c["height"]) for c in cams.values()}
    if len(native_sizes) != 1:
        raise ValueError(f"scene has mixed native camera sizes: {native_sizes}")
    native_w, native_h = next(iter(native_sizes))

    if width is None:
        width = native_w
    if height is None:
        height = native_h

    sx = width / native_w
    sy = height / native_h

    image_entries: list[ColmapImageEntry] = []
    for rec in imgs:
        cam = cams[rec["camera_id"]]
        view = _view_matrix(
            rec["qw"], rec["qx"], rec["qy"], rec["qz"],
            rec["tx"], rec["ty"], rec["tz"],
        )
        proj = _projection_matrix(
            fx=cam["fx"] * sx, fy=cam["fy"] * sy,
            cx=cam["cx"] * sx, cy=cam["cy"] * sy,
            width=width, height=height,
        )
        camera = Camera(
            view_matrix=view.astype(np.float32),
            proj_matrix=proj.astype(np.float32),
            image_size=(width, height),
        )
        image_entries.append(ColmapImageEntry(
            image_id=rec["image_id"], image_name=rec["name"], camera=camera,
        ))

    # Initial-points dict that matches write_gaussians signature.
    n = pts3d.shape[0]
    rng = np.random.default_rng(0)
    initial_points = {
        "means": pts3d,
        # Initial scale = 0.01 world units, isotropic. In v0 synthetic
        # experiments this gave visible but still-trainable sizes.
        "log_scales": np.full((n, 3), np.log(0.01), dtype=np.float32),
        # Identity quaternion.
        "rot_quats": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        # Initial opacity 0.1 (pre-sigmoid in training; we store the
        # activated value since our struct holds opacity in [0, 1]).
        "opacities": np.full(n, 0.1, dtype=np.float32),
        # Mid-gray color (sigmoid(0) = 0.5). The points.txt typically
        # carries real colors but we ignore those in v1 (PLAN.md): the
        # scene doesn't benefit materially and it's one fewer thing to
        # plumb through the loader.
        "color_logits": np.zeros((n, 3), dtype=np.float32),
    }

    return ColmapDataset(
        images=image_entries,
        image_dir=img_dir,
        initial_points=initial_points,
    )


def load_image(entry: ColmapImageEntry, image_dir: Path, width: int, height: int) -> np.ndarray:
    """Read a PNG/JPG from disk and downsample to (height, width, 3) float32.

    Returns sRGB values in [0, 1]. Alpha is dropped (PLAN.md v1 D table).
    """
    from PIL import Image
    path = image_dir / entry.image_name
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr
