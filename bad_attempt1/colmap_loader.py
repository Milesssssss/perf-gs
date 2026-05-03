"""
COLMAP Data Loader for 3DGS Training

Loads camera intrinsics, extrinsics, and images from COLMAP output.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import struct
from PIL import Image


def read_cameras_binary(path: Path) -> Dict[int, dict]:
    """
    Read cameras.bin file from COLMAP
    
    Returns dict mapping camera_id to camera parameters
    """
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            
            # Number of parameters depends on camera model
            # SIMPLE_PINHOLE: 3, PINHOLE: 4, SIMPLE_RADIAL: 4, RADIAL: 5, etc.
            model_params = {
                0: 3,  # SIMPLE_PINHOLE
                1: 4,  # PINHOLE
                2: 4,  # SIMPLE_RADIAL
                3: 5,  # RADIAL
                4: 8,  # OPENCV
                5: 12, # OPENCV_FISHEYE
                6: 5,  # FULL_OPENCV
            }
            
            num_params = model_params.get(model_id, 4)
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            
            cameras[camera_id] = {
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': params
            }
    
    return cameras


def read_images_binary(path: Path) -> List[dict]:
    """
    Read images.bin file from COLMAP
    
    Returns list of image data with camera poses
    """
    images = []
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        
        for idx in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            
            # Quaternion (w, x, y, z)
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            
            # Translation
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            
            camera_id = struct.unpack("<I", f.read(4))[0]
            
            # Image name (null-terminated string)
            name_chars = []
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name_chars.append(c.decode('utf-8'))
            name = ''.join(name_chars)
            
            # Skip 2D points (num_points2D * (x, y, point3D_id))
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)  # Each point is 24 bytes
            
            images.append({
                'image_id': image_id,
                'camera_id': camera_id,
                'name': name,
                'quat': np.array([qw, qx, qy, qz]),
                'translation': np.array([tx, ty, tz]),
                'index': idx
            })
    
    return images


def read_points3D_binary(path: Path) -> np.ndarray:
    """
    Read points3D.bin file from COLMAP
    
    Returns (N, 3) array of 3D point positions
    """
    points = []
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            
            # Skip track info
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(track_length * 8)  # Each track element is 8 bytes
            
            points.append(xyz)
    
    return np.array(points, dtype=np.float32)


def read_cameras_text(path: Path) -> Dict[int, dict]:
    """
    Read cameras.txt file from COLMAP
    
    Returns dict mapping camera_id to camera parameters
    """
    cameras = {}
    
    # Model name to ID mapping
    model_ids = {
        'SIMPLE_PINHOLE': 0,
        'PINHOLE': 1,
        'SIMPLE_RADIAL': 2,
        'RADIAL': 3,
        'OPENCV': 4,
        'OPENCV_FISHEYE': 5,
        'FULL_OPENCV': 6,
    }
    
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            camera_id = int(parts[0])
            model_name = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = tuple(float(p) for p in parts[4:])
            
            cameras[camera_id] = {
                'model_id': model_ids.get(model_name, 1),
                'width': width,
                'height': height,
                'params': params
            }
    
    return cameras


def read_images_text(path: Path) -> List[dict]:
    """
    Read images.txt file from COLMAP
    
    Returns list of image data with camera poses
    """
    images = []
    
    with open(path, "r") as f:
        lines = f.readlines()
    
    idx = 0
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        line_idx += 1
        
        # Skip comments and empty lines
        if not line or line.startswith("#"):
            continue
        
        # First line: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        parts = line.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]
        
        # Second line: 2D points (skip)
        if line_idx < len(lines):
            line_idx += 1
        
        images.append({
            'image_id': image_id,
            'camera_id': camera_id,
            'name': name,
            'quat': np.array([qw, qx, qy, qz]),
            'translation': np.array([tx, ty, tz]),
            'index': idx
        })
        idx += 1
    
    return images


def read_points3D_text(path: Path) -> np.ndarray:
    """
    Read points3D.txt file from COLMAP
    
    Returns (N, 3) array of 3D point positions
    """
    points = []
    
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            parts = line.split()
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            points.append((x, y, z))
    
    return np.array(points, dtype=np.float32) if points else np.empty((0, 3), dtype=np.float32)


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to 3x3 rotation matrix
    """
    w, x, y, z = quat
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float32)


def compute_view_matrix(quat: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Compute 4x4 view matrix from Blender-exported COLMAP data.
    
    The data is in Blender world-space format and needs B2C transform.
    Matches the reference shader's getViewMatrix exactly:
    - R = transpose(B2C @ R_blender)
    - t_colmap = B2C @ t
    - Matrix has t in last row (before final transpose in shader)
    """
    R_blender = quat_to_rotation_matrix(quat)
    t = translation
    
    # Blender to COLMAP transform (negate Y and Z)
    B2C = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]
    ], dtype=np.float32)
    
    # Match shader: R = transpose(B2C @ R_blender), t_colmap = B2C @ t
    R = (B2C @ R_blender).T
    t_colmap = B2C @ t
    
    # Build matrix with t in last row (shader format before transpose)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = R
    view[3, :3] = t_colmap  # t in last ROW, not column
    
    return view


def compute_projection_matrix(
    fx: float, fy: float,
    cx: float, cy: float,
    width: int, height: int,
    near: float = 0.01,
    far: float = 1000.0
) -> np.ndarray:
    """
    Compute 4x4 projection matrix from camera intrinsics
    
    Based on getProjectionMatrix in prepare_colmap_iteration_data.slang
    """
    proj = np.zeros((4, 4), dtype=np.float32)
    
    # Match the shader's projection matrix format
    proj[0, 0] = 2.0 * fx / width
    proj[0, 2] = 1.0 - 2.0 * cx / width
    proj[1, 1] = 2.0 * fy / height
    proj[1, 2] = 2.0 * cy / height - 1.0
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    
    return proj


def load_colmap_cameras(colmap_path: Path) -> Tuple[Dict[int, dict], List[dict]]:
    """
    Load COLMAP camera data
    
    Args:
        colmap_path: Path to COLMAP output directory (containing sparse/0/)
        
    Returns:
        cameras: Dict mapping camera_id to camera intrinsics + projection matrix
        images: List of image data with view matrices
    """
    sparse_path = colmap_path / "sparse" / "0" / "text"
    
    # Try binary first, then text
    if (sparse_path / "cameras.bin").exists():
        cameras = read_cameras_binary(sparse_path / "cameras.bin")
        images = read_images_binary(sparse_path / "images.bin")
    elif (sparse_path / "cameras.txt").exists():
        cameras = read_cameras_text(sparse_path / "cameras.txt")
        images = read_images_text(sparse_path / "images.txt")
    else:
        raise FileNotFoundError(f"No COLMAP camera files found in {sparse_path}")
    
    # Compute projection matrices for each camera
    for cam_id, cam in cameras.items():
        params = cam['params']
        width, height = cam['width'], cam['height']
        
        # Extract focal length and principal point based on model
        if cam['model_id'] in [0]:  # SIMPLE_PINHOLE
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif cam['model_id'] in [1, 2, 3, 4]:  # PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
        else:
            # Default to first 4 params
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
        
        cam['projection_matrix'] = compute_projection_matrix(fx, fy, cx, cy, width, height)
    
    # Compute view matrices for each image
    for img in images:
        img['view_matrix'] = compute_view_matrix(img['quat'], img['translation'])
    
    return cameras, images


def load_colmap_points(colmap_path: Path) -> np.ndarray:
    """
    Load COLMAP 3D points
    
    Args:
        colmap_path: Path to COLMAP output directory
        
    Returns:
        (N, 3) array of 3D point positions
    """
    sparse_path = colmap_path / "sparse" / "0" / "text"
    
    # Try binary first, then text
    if (sparse_path / "points3D.bin").exists():
        return read_points3D_binary(sparse_path / "points3D.bin")
    elif (sparse_path / "points3D.txt").exists():
        return read_points3D_text(sparse_path / "points3D.txt")
    else:
        # Try without /text subdirectory
        sparse_path = colmap_path / "sparse" / "0"
        if (sparse_path / "points3D.bin").exists():
            return read_points3D_binary(sparse_path / "points3D.bin")
        elif (sparse_path / "points3D.txt").exists():
            return read_points3D_text(sparse_path / "points3D.txt")
    
    return np.empty((0, 3), dtype=np.float32)


def load_colmap_images(
    colmap_path: Path,
    images: List[dict],
    target_width: int,
    target_height: int
) -> List[np.ndarray]:
    """
    Load and resize COLMAP images
    
    Args:
        colmap_path: Path to COLMAP output directory
        images: List of image metadata from load_colmap_cameras
        target_width: Target image width
        target_height: Target image height
        
    Returns:
        List of (H, W, 4) uint8 RGBA arrays
    """
    images_path = colmap_path / "images"
    
    image_arrays = []
    for img_data in images:
        img_path = images_path / img_data['name']
        
        if not img_path.exists():
            # Try common subdirectories
            for subdir in ['', 'images', 'input']:
                test_path = colmap_path / subdir / img_data['name']
                if test_path.exists():
                    img_path = test_path
                    break
        
        if not img_path.exists():
            print(f"Warning: Image not found: {img_data['name']}")
            # Create placeholder
            img_array = np.zeros((target_height, target_width, 4), dtype=np.uint8)
            img_array[:, :, 3] = 255
        else:
            img = Image.open(img_path)
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            
            # Convert to RGBA
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            img_array = np.array(img, dtype=np.uint8)
        
        image_arrays.append(img_array)
    
    return image_arrays


def load_colmap_points(colmap_path: Path) -> np.ndarray:
    """
    Load COLMAP 3D points for gaussian initialization
    
    Args:
        colmap_path: Path to COLMAP output directory
        
    Returns:
        (N, 3) array of 3D point positions
    """
    # Try sparse/0/text first (common structure)
    sparse_path = colmap_path / "sparse" / "0" / "text"
    
    if (sparse_path / "points3D.bin").exists():
        return read_points3D_binary(sparse_path / "points3D.bin")
    elif (sparse_path / "points3D.txt").exists():
        return read_points3D_text(sparse_path / "points3D.txt")
    
    # Try sparse/0 without text subdirectory
    sparse_path = colmap_path / "sparse" / "0"
    if (sparse_path / "points3D.bin").exists():
        return read_points3D_binary(sparse_path / "points3D.bin")
    elif (sparse_path / "points3D.txt").exists():
        return read_points3D_text(sparse_path / "points3D.txt")
    
    return np.empty((0, 3), dtype=np.float32)


def get_scene_bounds(points: np.ndarray, padding: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box of point cloud
    
    Args:
        points: (N, 3) array of 3D positions
        padding: Fractional padding to add to bounds
        
    Returns:
        scene_min, scene_max: (3,) arrays of bounds
    """
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    
    extent = max_pt - min_pt
    min_pt -= extent * padding
    max_pt += extent * padding
    
    return min_pt, max_pt
