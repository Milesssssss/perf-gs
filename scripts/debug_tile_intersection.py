"""Debug tile intersection — visual output.

For each test case, outputs 3 images:
  {name}_render.png   — the raw rendered image
  {name}_tiles.png    — white = tile has gaussians, black = empty tile
  {name}_errors.png   — green = correct coverage, RED = missed (visible pixel in empty tile)

Usage:
    .venv/Scripts/python.exe scripts/debug_tile_intersection.py
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image

from trainer.dataset import simple_camera
from trainer.device import create_device
from trainer.pipeline import Pipeline


TILE_SIZE = 16
OUT_DIR = Path("outputs/debug_tiles")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80, 80)))


def run_test(name, means, log_scales, rot_quats, opacities, color_logits, W, H, camera, device):
    print(f"\n  {name}:", end=" ")

    n_g = means.shape[0]
    pipeline = Pipeline(device=device, width=W, height=H, max_splats=max(64, n_g * 2))
    pipeline.lambda_ssim = 0.0
    pipeline.w_alpha = 0.0
    pipeline.set_gaussians_from_numpy(
        means=means, log_scales=log_scales, rot_quats=rot_quats,
        opacities=opacities, color_logits=color_logits,
    )

    pipeline.render(camera)
    rendered = pipeline.read_output_image()
    rgb = np.clip(rendered[..., :3], 0, 1)

    pipeline.device.wait_for_idle()
    n_live = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])

    # Tile coverage map.
    tile_start_raw = pipeline.buffers.tile_start.to_numpy()
    tile_end_raw = pipeline.buffers.tile_end.to_numpy()
    tile_starts = np.frombuffer(tile_start_raw.tobytes(), dtype=np.uint32)
    tile_ends = np.frombuffer(tile_end_raw.tobytes(), dtype=np.uint32)
    num_tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
    num_tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE

    tile_has_gaussians = np.zeros((num_tiles_y, num_tiles_x), dtype=bool)
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            tid = ty * num_tiles_x + tx
            if tid < len(tile_starts) and tile_ends[tid] > tile_starts[tid]:
                tile_has_gaussians[ty, tx] = True

    # CPU ground-truth: which pixels have alpha > 1/255?
    proj_raw = pipeline.buffers.projected_2d.to_numpy()
    proj_floats = np.frombuffer(proj_raw.tobytes(), dtype=np.float32).reshape(-1, 6)[:n_live]

    gt_has_alpha = np.zeros((H, W), dtype=bool)
    for gi in range(n_live):
        mean_2d = proj_floats[gi, :2]
        cov2d = proj_floats[gi, 2:5]
        depth = proj_floats[gi, 5]
        if depth <= 0:
            continue
        cov00, cov01, cov11 = cov2d
        det = cov00 * cov11 - cov01 * cov01
        if det < 1e-20:
            continue
        inv_det = 1.0 / det

        opa_logit = float(np.log(opacities[gi] / (1.0 - opacities[gi])))
        opa = sigmoid(opa_logit)

        max_var = max(cov00, cov11)
        radius = 4.0 * np.sqrt(max_var)
        px_min = max(0, int(mean_2d[0] - radius))
        px_max = min(W, int(mean_2d[0] + radius) + 1)
        py_min = max(0, int(mean_2d[1] - radius))
        py_max = min(H, int(mean_2d[1] + radius) + 1)

        for py in range(py_min, py_max):
            for px in range(px_min, px_max):
                d = np.array([px + 0.5 - mean_2d[0], py + 0.5 - mean_2d[1]])
                power = -0.5 * (d[0]*d[0]*cov11*inv_det
                                - 2*d[0]*d[1]*cov01*inv_det
                                + d[1]*d[1]*cov00*inv_det)
                alpha = opa * np.exp(power)
                if alpha > 1.0 / 255.0:
                    gt_has_alpha[py, px] = True

    # Find missed pixels: has alpha but tile is empty.
    errors = np.zeros((H, W), dtype=bool)
    for py in range(H):
        for px in range(W):
            if gt_has_alpha[py, px]:
                tx, ty = px // TILE_SIZE, py // TILE_SIZE
                if not tile_has_gaussians[ty, tx]:
                    errors[py, px] = True

    n_missed = errors.sum()

    # --- Save images ---
    # 1. Render
    render_srgb = (np.power(rgb, 1.0/2.2) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(render_srgb, mode="RGB").save(OUT_DIR / f"{name}_render.png")

    # 2. Tile coverage: white = has gaussians, black = empty
    tile_img = np.zeros((H, W), dtype=np.uint8)
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            if tile_has_gaussians[ty, tx]:
                y0, y1 = ty * TILE_SIZE, min((ty+1) * TILE_SIZE, H)
                x0, x1 = tx * TILE_SIZE, min((tx+1) * TILE_SIZE, W)
                tile_img[y0:y1, x0:x1] = 255
    Image.fromarray(tile_img, mode="L").save(OUT_DIR / f"{name}_tiles.png")

    # 3. Error map: green = correctly covered, RED = missed
    error_img = np.zeros((H, W, 3), dtype=np.uint8)
    error_img[gt_has_alpha & ~errors] = [0, 200, 0]   # green = good
    error_img[errors] = [255, 0, 0]                    # red = MISSED
    Image.fromarray(error_img, mode="RGB").save(OUT_DIR / f"{name}_errors.png")

    status = f"MISSED {n_missed} px" if n_missed > 0 else "OK"
    print(f"{status} (n_g={n_live}, tiles_with_data={tile_has_gaussians.sum()})")
    return n_missed


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = create_device(backend="vulkan")
    W, H = 256, 256

    camera = simple_camera(
        width=W, height=H,
        eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
        fovy_degrees=60.0,
    )

    print("Running tile intersection tests...")
    total_missed = 0

    total_missed += run_test("01_isotropic",
        means=np.array([[0.0, 0.0, 3.0]], dtype=np.float32),
        log_scales=np.log(np.array([[0.2, 0.2, 0.2]], dtype=np.float32)),
        rot_quats=np.array([[1, 0, 0, 0]], dtype=np.float32),
        opacities=np.array([0.9], dtype=np.float32),
        color_logits=np.array([[2, -2, -2]], dtype=np.float32),
        W=W, H=H, camera=camera, device=device)

    total_missed += run_test("02_elongated_45deg",
        means=np.array([[0.0, 0.0, 3.0]], dtype=np.float32),
        log_scales=np.log(np.array([[0.5, 0.02, 0.02]], dtype=np.float32)),
        rot_quats=np.array([[0.924, 0, 0, 0.383]], dtype=np.float32),
        opacities=np.array([0.95], dtype=np.float32),
        color_logits=np.array([[-2, 2, -2]], dtype=np.float32),
        W=W, H=H, camera=camera, device=device)

    total_missed += run_test("03_elongated_steep",
        means=np.array([[0.0, 0.0, 3.0]], dtype=np.float32),
        log_scales=np.log(np.array([[0.6, 0.01, 0.01]], dtype=np.float32)),
        rot_quats=np.array([[0.707, 0, 0, 0.707]], dtype=np.float32),  # 90° Z
        opacities=np.array([0.99], dtype=np.float32),
        color_logits=np.array([[2, 2, -2]], dtype=np.float32),
        W=W, H=H, camera=camera, device=device)

    total_missed += run_test("04_edge",
        means=np.array([[0.7, 0.5, 2.5]], dtype=np.float32),
        log_scales=np.log(np.array([[0.15, 0.15, 0.15]], dtype=np.float32)),
        rot_quats=np.array([[1, 0, 0, 0]], dtype=np.float32),
        opacities=np.array([0.85], dtype=np.float32),
        color_logits=np.array([[-2, -2, 2]], dtype=np.float32),
        W=W, H=H, camera=camera, device=device)

    total_missed += run_test("05_very_close",
        means=np.array([[0.0, 0.0, 1.2]], dtype=np.float32),
        log_scales=np.log(np.array([[0.3, 0.3, 0.3]], dtype=np.float32)),
        rot_quats=np.array([[1, 0, 0, 0]], dtype=np.float32),
        opacities=np.array([0.9], dtype=np.float32),
        color_logits=np.array([[2, 2, -2]], dtype=np.float32),
        W=W, H=H, camera=camera, device=device)

    total_missed += run_test("06_low_opacity",
        means=np.array([[0.0, 0.0, 3.0]], dtype=np.float32),
        log_scales=np.log(np.array([[0.3, 0.3, 0.3]], dtype=np.float32)),
        rot_quats=np.array([[1, 0, 0, 0]], dtype=np.float32),
        opacities=np.array([0.01], dtype=np.float32),
        color_logits=np.array([[2, -2, 2]], dtype=np.float32),
        W=W, H=H, camera=camera, device=device)

    total_missed += run_test("07_multiple",
        means=np.array([
            [0.0, 0.0, 3.0], [0.3, 0.2, 2.5],
            [-0.2, -0.3, 3.5], [0.0, 0.0, 2.0],
        ], dtype=np.float32),
        log_scales=np.log(np.array([
            [0.15, 0.15, 0.15], [0.2, 0.05, 0.1],
            [0.1, 0.3, 0.1], [0.1, 0.1, 0.1],
        ], dtype=np.float32)),
        rot_quats=np.array([
            [1, 0, 0, 0], [0.924, 0, 0, 0.383],
            [0.866, 0.5, 0, 0], [0.966, 0, 0.259, 0],
        ], dtype=np.float32),
        opacities=np.array([0.8, 0.9, 0.7, 0.85], dtype=np.float32),
        color_logits=np.array([
            [2, -2, -2], [-2, 2, -2], [-2, -2, 2], [2, 2, -2],
        ], dtype=np.float32),
        W=W, H=H, camera=camera, device=device)

    total_missed += run_test("08_tiny",
        means=np.array([[0.0, 0.0, 3.0]], dtype=np.float32),
        log_scales=np.log(np.array([[0.001, 0.001, 0.001]], dtype=np.float32)),
        rot_quats=np.array([[1, 0, 0, 0]], dtype=np.float32),
        opacities=np.array([0.99], dtype=np.float32),
        color_logits=np.array([[2, 2, 2]], dtype=np.float32),
        W=W, H=H, camera=camera, device=device)

    print(f"\n{'='*40}")
    if total_missed == 0:
        print("ALL PASSED — zero missed pixels.")
    else:
        print(f"FAILED — {total_missed} total missed pixels. Check *_errors.png for red areas.")
    print(f"Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
