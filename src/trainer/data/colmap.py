# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trainer.data.dataset import Camera

_SH_C0 = 0.28209479177387814

@dataclass
class ColmapImageEntry:
    # image id starts at 1
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
    R = _quat_wxyz_to_rotmat(np.array([qw, qx, qy, qz]))
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = [tx, ty, tz]
    return M


def _projection_matrix(fx: float, fy: float, cx: float, cy: float,
                       width: int, height: int,
                       z_near: float = 0.01, z_far: float = 100.0) -> np.ndarray:
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
    return ((srgb - 0.5) / _SH_C0).astype(np.float32)


def _find_sparse_dir(scene_dir: Path) -> tuple[Path | None, str]:
    """
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

    imgs = sorted(imgs, key=lambda r: r["name"])

    size_counts: dict[tuple[int, int], int] = {}
    for rec in imgs:
        cam = cams[rec["camera_id"]]
        key = (cam["width"], cam["height"])
        size_counts[key] = size_counts.get(key, 0) + 1
    # Ties break on pixel count, so the choice never depends on dict order.
    native_w, native_h = max(size_counts,
                             key=lambda s: (size_counts[s], s[0] * s[1]))
    uniform = len(size_counts) == 1
    if not uniform:
        spread = ", ".join(f"{w}x{h}x{n}" for (w, h), n in
                           sorted(size_counts.items(), key=lambda kv: -kv[1]))
        print(f"[colmap] {len(size_counts)} native camera sizes in this model "
              f"({spread}) — typical of an undistorted reconstruction. Rendering "
              f"at the most common size ({native_w}x{native_h}); the other "
              f"{len(imgs) - size_counts[(native_w, native_h)]} views are "
              f"resampled on load and keep their own intrinsics.")

    if width is None or height is None:
        # When a pre-downsampled images_N directory is used, adopt its ACTUAL
        # on-disk pixel size as the render target. The COLMAP downsampler
        # rounds (e.g. 4946//4 = 1236 but the tool wrote 1237), so computing
        # native // downsample here disagrees with the files by ~1px, which
        # made load_image LANCZOS-resize every image on load (~110ms each,
        # ~21s over 194 cams) for nothing. Reading the real size skips that
        # resize entirely and keeps the projection intrinsics exact.
        # Only when the model agrees on one size: with mixed sizes the first file
        # on disk is an arbitrary one of them, while the camera records are exactly
        # what each view was undistorted to.
        disk_w = disk_h = None
        if uniform and img_dir != scene_dir / "images":
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

    image_entries: list[ColmapImageEntry] = []
    for rec in imgs:
        cam = cams[rec["camera_id"]]
        # Per-camera scale, not one global factor: a view whose native size
        # differs from the target is stretched to it by load_colmap_image, so its
        # focal length and principal point must follow ITS OWN size or the
        # projection silently disagrees with the pixels by a fraction of a pixel.
        sx = width / cam["width"]
        sy = height / cam["height"]
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


def load_colmap_image(entry: ColmapImageEntry, image_dir: Path, width: int, height: int,
               load_alpha: bool = False) -> np.ndarray:
    """Read a PNG/JPG from disk and downsample to float32.

    Returns sRGB values in [0, 1]. If ``load_alpha`` is True and the image
    has an alpha channel, returns (H, W, 4); otherwise (H, W, 3).

    OpenCV, not PIL: decoding a full dataset is startup's long pole (a 195-view
    360_v2 scene is ~200 JPEG decodes before the first iteration), and
    ``cv2.imread`` is several times faster than ``Image.open`` + ``convert`` for
    the same pixels. Two deliberate differences from the PIL version:

      * ``INTER_AREA`` replaces ``LANCZOS``. This path only ever DOWNsamples
        (a render target is never bigger than the source), where area averaging
        is the correct box filter and far cheaper; on the pre-downsampled
        ``images_N`` directories the resize is skipped entirely anyway.
      * EXIF orientation is explicitly IGNORED (``IMREAD_IGNORE_ORIENTATION``).
        PIL's ``Image.open`` never auto-rotates, and rotating here would silently
        disagree with the COLMAP intrinsics for the same view.
    """
    import cv2
    path = image_dir / entry.image_name
    # IMREAD_COLOR already collapses grayscale/alpha sources to 3-channel 8-bit
    # BGR, which is exactly what the PIL ``convert("RGB")`` branch did.
    flags = cv2.IMREAD_IGNORE_ORIENTATION | (
        cv2.IMREAD_UNCHANGED if load_alpha else cv2.IMREAD_COLOR)
    array = cv2.imread(str(path), flags)
    if array is None and path.exists():
        # cv2.imread() opens through the NARROW (ANSI) file API on Windows, so a
        # path outside the active code page fails even though the file is right
        # there. Decoding the bytes ourselves is the documented way around it.
        array = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if array is None:
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        raise OSError(f"failed to decode image: {path}")

    channels = 1 if array.ndim == 2 else array.shape[2]
    if load_alpha and channels == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
    elif channels == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
    elif channels == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    elif channels == 1:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    else:
        raise OSError(f"unsupported {channels}-channel image: {path}")

    if (array.shape[1], array.shape[0]) != (width, height):
        # INTER_AREA degenerates to nearest-neighbour when asked to ENLARGE, and
        # an undistorted model's off-size views (a pixel or two under the common
        # target) do enlarge. Shrinking still takes the area path.
        shrinking = width <= array.shape[1] and height <= array.shape[0]
        array = cv2.resize(array, (width, height),
                           interpolation=cv2.INTER_AREA if shrinking
                           else cv2.INTER_LINEAR)
    # Resize in the source's own integer precision (as the PIL path did), then
    # normalize by the dtype's full scale — 8-bit sources divide by 255, and a
    # 16-bit PNG read through IMREAD_UNCHANGED keeps its extra bits instead of
    # being silently truncated to 8.
    #
    # cv2.multiply, not ``astype(np.float32) / max``: the results are bit-for-bit
    # identical (checked across 1/3/4 channels and uint8/uint16) but numpy's cast
    # is one single-threaded pass that HOLDS THE GIL, which is what otherwise caps
    # the threaded dataset load in TargetDataset. The 4-element scale sidesteps
    # OpenCV's scalar-broadcast rule (a bare number can mean "channel 0 only").
    if array.dtype.kind in "ui":
        scale = 1.0 / float(np.iinfo(array.dtype).max)
        array = cv2.multiply(array, (scale,) * 4, dtype=cv2.CV_32F)
    else:
        array = array.astype(np.float32)
    return np.ascontiguousarray(array)
