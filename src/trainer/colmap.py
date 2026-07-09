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

from trainer.dataset import Camera, _SH_C0


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
    scene_extent: float = 1.0              # bounding sphere radius


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


def _parse_points3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions (N,3) float32, colors (N,3) float32 in [0,1])."""
    pts: list[list[float]] = []
    colors: list[list[float]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            colors.append([int(parts[4]) / 255.0, int(parts[5]) / 255.0, int(parts[6]) / 255.0])
    positions = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)
    rgb = np.array(colors, dtype=np.float32) if colors else np.zeros((0, 3), dtype=np.float32)
    return positions, rgb


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


def _srgb_colors_to_sh0(srgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] point colors to the SH band-0 (DC) coefficient the
    renderer expects: color = 0.5 + C0*sh0  =>  sh0 = (color - 0.5)/C0.
    COLORSPACE: the loss now runs in sRGB (INRIA/gsplat/SADGS convention), so
    we seed sh0 DIRECTLY from the sRGB color — no linearization. No clamp/
    logit — sh0 is unbounded (affine), which is the point of the SH0 switch."""
    return ((srgb - 0.5) / _SH_C0).astype(np.float32)


def _find_sparse_dir(scene_dir: Path) -> tuple[Path | None, str]:
    """Find the COLMAP sparse reconstruction, trying common layouts.

    Returns (directory, format) where format is 'text' or 'bin'.
    Tries in order:
      sparse/0/text/cameras.txt
      sparse/0/cameras.txt
      sparse/0/cameras.bin
    """
    # sparse/0/text/*.txt
    d = scene_dir / "sparse" / "0" / "text"
    if (d / "cameras.txt").exists():
        return d, "text"
    # sparse/0/*.txt
    d = scene_dir / "sparse" / "0"
    if (d / "cameras.txt").exists():
        return d, "text"
    # sparse/0/*.bin
    if (d / "cameras.bin").exists():
        return d, "bin"
    return None, ""


def _parse_cameras_bin(path: Path) -> dict[int, dict]:
    """Parse COLMAP binary cameras.bin."""
    import struct
    cameras: dict[int, dict] = {}
    with path.open("rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id, model_id, w, h = struct.unpack("<iIQQ", f.read(24))
            # Model IDs: 0=SIMPLE_PINHOLE(3), 1=PINHOLE(4), ...
            if model_id == 0:  # SIMPLE_PINHOLE
                params = struct.unpack("<3d", f.read(24))
                fx = fy = params[0]
                cx, cy = params[1], params[2]
                model = "SIMPLE_PINHOLE"
            elif model_id == 1:  # PINHOLE
                params = struct.unpack("<4d", f.read(32))
                fx, fy, cx, cy = params
                model = "PINHOLE"
            else:
                # Skip unknown models — read param count from model table
                num_params = {2: 4, 3: 5, 4: 5, 5: 12}.get(model_id, 4)
                f.read(num_params * 8)
                continue
            cameras[cam_id] = {
                "model": model, "width": int(w), "height": int(h),
                "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            }
    return cameras


def _parse_images_bin(path: Path) -> list[dict]:
    """Parse COLMAP binary images.bin."""
    import struct
    entries: list[dict] = []
    with path.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<i", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<i", f.read(4))[0]
            # Image name: null-terminated string.
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            # Skip 2D points (num_points2D * (x, y, point3D_id)).
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * 24)  # 2 doubles + 1 int64
            entries.append({
                "image_id": image_id,
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "tx": tx, "ty": ty, "tz": tz,
                "camera_id": camera_id,
                "name": name,
            })
    return entries


def _parse_points3d_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse COLMAP binary points3D.bin. Returns (positions, colors)."""
    import struct
    pts: list[list[float]] = []
    colors: list[list[float]] = []
    with path.open("rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point3d_id = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<3d", f.read(24))
            r, g, b = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(track_length * 8)
            pts.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
    positions = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 3), dtype=np.float32)
    rgb = np.array(colors, dtype=np.float32) if colors else np.zeros((0, 3), dtype=np.float32)
    return positions, rgb


def load_colmap(
    scene_dir: Path | str,
    width: int | None = None,
    height: int | None = None,
    downsample: int = 1,
    init_opacity: float = 0.1,
) -> ColmapDataset:
    """Load a COLMAP scene, auto-detecting text vs binary format.

    Resolution is determined by (in priority order):
      1. Explicit ``width``/``height`` if provided.
      2. ``downsample`` factor applied to native resolution.
         If ``images_N`` directory exists, images are loaded from there
         (already downsampled); otherwise full-res images are resized on load.
      3. Native resolution (downsample=1, no width/height).
    """
    scene_dir = Path(scene_dir)

    # Choose image directory: images_N if it exists, else images.
    if downsample > 1 and (scene_dir / f"images_{downsample}").exists():
        img_dir = scene_dir / f"images_{downsample}"
    else:
        img_dir = scene_dir / "images"

    sparse_dir, fmt = _find_sparse_dir(scene_dir)
    if sparse_dir is None:
        raise FileNotFoundError(
            f"No COLMAP sparse reconstruction found in {scene_dir}/sparse/0/ "
            f"(tried text/*.txt, *.txt, *.bin)")

    if fmt == "text":
        cams = _parse_cameras(sparse_dir / "cameras.txt")
        imgs = _parse_images(sparse_dir / "images.txt")
        pts3d, pts_colors = _parse_points3d(sparse_dir / "points3D.txt")
    else:
        cams = _parse_cameras_bin(sparse_dir / "cameras.bin")
        imgs = _parse_images_bin(sparse_dir / "images.bin")
        pts3d, pts_colors = _parse_points3d_bin(sparse_dir / "points3D.bin")

    # Sanity: all cameras same native resolution.
    native_sizes = {(c["width"], c["height"]) for c in cams.values()}
    if len(native_sizes) != 1:
        raise ValueError(f"scene has mixed native camera sizes: {native_sizes}")
    native_w, native_h = next(iter(native_sizes))

    if width is None or height is None:
        # When a pre-downsampled images_N directory is used, adopt its ACTUAL
        # on-disk pixel size as the render target. The COLMAP downsampler
        # rounds (e.g. 4946//4 = 1236 but the tool wrote 1237), so computing
        # native // downsample here disagrees with the files by ~1px, which
        # made load_image LANCZOS-resize every image on load (~110ms each,
        # ~21s over 194 cams) for nothing. Reading the real size skips that
        # resize entirely and keeps the projection intrinsics exact.
        disk_w = disk_h = None
        if img_dir != scene_dir / "images":
            try:
                from PIL import Image
                sample = next(p for p in sorted(img_dir.iterdir())
                              if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
                disk_w, disk_h = Image.open(sample).size
            except (StopIteration, OSError):
                disk_w = disk_h = None
        if width is None:
            width = disk_w if disk_w is not None else native_w // downsample
        if height is None:
            height = disk_h if disk_h is not None else native_h // downsample

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

    # gsplat-style k-NN scale init: each point's initial scale is the mean
    # distance to its k nearest neighbours. Lone points in sparse regions
    # get larger gaussians; points in dense clusters get smaller ones.
    # This is critical for SfM-initialized scenes where point density
    # varies dramatically — a uniform init wastes capacity on either
    # over-covered dense regions or leaves gaps in sparse regions.
    #
    # For k=3: query 4 neighbours (the closest is the point itself at
    # distance 0), average the next 3.
    from scipy.spatial import cKDTree
    k = 3
    init_scale = 1.0
    if n >= k + 1:
        tree = cKDTree(pts3d)
        dists, _ = tree.query(pts3d, k=k + 1)   # (n, k+1), col 0 is self
        dist_sq = (dists[:, 1:k + 1] ** 2).mean(axis=-1).astype(np.float32)
        dist_sq = np.clip(dist_sq, 1e-7, None)
        knn_dist = np.sqrt(dist_sq)
    else:
        knn_dist = np.full(n, 0.01, dtype=np.float32)
    log_scale_per_point = np.log(knn_dist * init_scale)

    initial_points = {
        "means": pts3d,
        # Isotropic, per-point kNN-derived scale.
        "log_scales": np.repeat(log_scale_per_point[:, None], 3, axis=1).astype(np.float32),
        # Identity quaternion.
        "rot_quats": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1)),
        # Initial opacity 0.1 (probability; converted to logit by
        # write_gaussians before upload).
        "opacities": np.full(n, init_opacity, dtype=np.float32),
        # SH band-0 (DC) color coeff from SfM point colors (sRGB -> sh0 directly).
        # Dict key stays "color_logits" as the buffers.write_gaussians plumbing name.
        "color_logits": _srgb_colors_to_sh0(pts_colors),
    }

    # Scene extent: radius of the camera bounding sphere * 1.1 (gsplat/3DGS
    # convention — getNerfppNorm). Used by densification scale thresholds:
    #   split/clone boundary = 0.01 * scene_extent
    #   big-scale prune      = 0.1  * scene_extent
    cam_positions = []
    for rec in imgs:
        R = _quat_wxyz_to_rotmat(np.array([rec["qw"], rec["qx"], rec["qy"], rec["qz"]]))
        t = np.array([rec["tx"], rec["ty"], rec["tz"]])
        cam_positions.append(-R.T @ t)
    cam_positions = np.array(cam_positions, dtype=np.float64)
    cam_centroid = cam_positions.mean(axis=0)
    scene_extent = float(np.linalg.norm(cam_positions - cam_centroid, axis=1).max()) * 1.1

    return ColmapDataset(
        images=image_entries,
        image_dir=img_dir,
        initial_points=initial_points,
        scene_extent=scene_extent,
    )


def load_image(entry: ColmapImageEntry, image_dir: Path, width: int, height: int,
               load_alpha: bool = False) -> np.ndarray:
    """Read a PNG/JPG from disk and downsample to float32.

    Returns sRGB values in [0, 1]. If ``load_alpha`` is True and the image
    has an alpha channel, returns (H, W, 4); otherwise (H, W, 3).
    """
    from PIL import Image
    path = image_dir / entry.image_name
    img = Image.open(path)
    if load_alpha and img.mode == "RGBA":
        img = img.convert("RGBA")
    else:
        img = img.convert("RGB")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr
