"""Interactive viewer for live training inspection.

Has its own Pipeline (separate buffers/scratch) and copies the gaussian
buffer + count from the training pipeline every N iterations. Completely
decoupled — no shared mutable state, no locks needed.

Usage:
    from trainer.viewer import Viewer
    viewer = Viewer(device, width, height, max_splats)
    viewer.start()
    # In training loop every N iters:
    viewer.update_gaussians(train_pipeline)
    # At end:
    viewer.stop()
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import slangpy as spy

from trainer.dataset import Camera
from trainer.pipeline import Pipeline


def _build_view_matrix(eye: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = forward / np.linalg.norm(forward)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = r
    M[1, :3] = u
    M[2, :3] = f
    M[0, 3] = -np.dot(r, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = -np.dot(f, eye)
    return M


def _build_proj_matrix(fovy_deg: float, aspect: float,
                       z_near: float = 0.01, z_far: float = 100.0) -> np.ndarray:
    fovy = math.radians(fovy_deg)
    fy = 1.0 / math.tan(fovy * 0.5)
    fx = fy / aspect
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = fx
    M[1, 1] = fy
    M[2, 2] = z_far / (z_far - z_near)
    M[2, 3] = -z_far * z_near / (z_far - z_near)
    M[3, 2] = 1.0
    return M


class FlyCamera:
    def __init__(self, eye=None, yaw=0.0, pitch=0.0, speed=1.0):
        self.eye = np.array(eye if eye is not None else [0.0, 0.0, -2.0], dtype=np.float64)
        self.yaw = yaw
        self.pitch = pitch
        self.speed = speed
        self.sensitivity = 0.2

    @property
    def forward(self) -> np.ndarray:
        yaw_r = math.radians(self.yaw)
        pitch_r = math.radians(self.pitch)
        return np.array([
            math.cos(pitch_r) * math.sin(yaw_r),
            -math.sin(pitch_r),
            math.cos(pitch_r) * math.cos(yaw_r),
        ], dtype=np.float64)

    @property
    def right(self) -> np.ndarray:
        f = self.forward
        up = np.array([0.0, 1.0, 0.0])
        r = np.cross(f, up)
        n = np.linalg.norm(r)
        return r / n if n > 1e-8 else np.array([1.0, 0.0, 0.0])

    def move(self, keys: set, dt: float):
        vel = self.speed * dt
        if "w" in keys:
            self.eye += self.forward * vel
        if "s" in keys:
            self.eye -= self.forward * vel
        if "a" in keys:
            self.eye -= self.right * vel
        if "d" in keys:
            self.eye += self.right * vel
        if "q" in keys:
            self.eye[1] -= vel
        if "e" in keys:
            self.eye[1] += vel

    def rotate(self, dx: float, dy: float):
        self.yaw += -dx * self.sensitivity
        self.pitch += -dy * self.sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def get_camera(self, width: int, height: int, fovy: float = 60.0) -> Camera:
        view = _build_view_matrix(
            self.eye.astype(np.float32),
            self.forward.astype(np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        proj = _build_proj_matrix(fovy, width / height)
        return Camera(view_matrix=view, proj_matrix=proj, image_size=(width, height))


class Viewer:
    def __init__(self, device: spy.Device, width: int = 960, height: int = 540,
                 max_splats: int = 500_000, fovy: float = 60.0):
        self.device = device
        self.width = width
        self.height = height
        self.max_splats = max_splats
        self.fovy = fovy
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Viewer has its own pipeline for rendering (separate buffers).
        self._pipeline = Pipeline(
            device=device, width=width, height=height, max_splats=max_splats,
        )
        self._pipeline.lambda_ssim = 0.0
        self._pipeline.w_alpha = 0.0
        self._has_data = False

    def update_gaussians(self, train_pipeline: Pipeline):
        """Copy gaussian buffer + count from trainer. Call from training thread."""
        train_pipeline.device.wait_for_idle()
        n_raw = train_pipeline.buffers.num_gaussians_counter.to_numpy()
        n = int(np.frombuffer(n_raw.tobytes(), dtype=np.uint32)[0])
        g_raw = train_pipeline.buffers.gaussians.to_numpy()
        self._pipeline.buffers.num_gaussians_counter.copy_from_numpy(
            np.array([n], dtype=np.uint32))
        self._pipeline.buffers.gaussians.copy_from_numpy(g_raw)
        self._has_data = True

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self):
        try:
            self._run_inner()
        except Exception as e:
            print(f"[Viewer] crashed: {e}")

    def _run_inner(self):
        window = spy.Window(width=self.width, height=self.height, title="3DGS Viewer")

        surface = self.device.create_surface(window)
        surface.configure(width=self.width, height=self.height)
        needs_reconfigure = False

        cam = FlyCamera()
        keys_held: set = set()
        mouse_captured = False
        last_mouse_x, last_mouse_y = 0.0, 0.0
        last_time = time.perf_counter()

        key_map = {
            spy.KeyCode.w: "w", spy.KeyCode.a: "a",
            spy.KeyCode.s: "s", spy.KeyCode.d: "d",
            spy.KeyCode.q: "q", spy.KeyCode.e: "e",
        }

        def on_key(event):
            nonlocal keys_held, mouse_captured
            k = key_map.get(event.key)
            if event.is_key_press() or event.is_key_repeat():
                if k:
                    keys_held.add(k)
                if event.key == spy.KeyCode.escape:
                    mouse_captured = False
            elif event.is_key_release():
                if k:
                    keys_held.discard(k)

        def on_mouse(event):
            nonlocal mouse_captured, last_mouse_x, last_mouse_y
            if event.is_button_down():
                mouse_captured = True
                last_mouse_x = event.pos[0]
                last_mouse_y = event.pos[1]
            elif event.is_button_up():
                mouse_captured = False
            elif event.is_move() and mouse_captured:
                dx = event.pos[0] - last_mouse_x
                dy = event.pos[1] - last_mouse_y
                last_mouse_x = event.pos[0]
                last_mouse_y = event.pos[1]
                cam.rotate(dx, dy)

        def on_resize(w, h):
            nonlocal needs_reconfigure
            needs_reconfigure = True

        window.on_keyboard_event = on_key
        window.on_mouse_event = on_mouse
        window.on_resize = on_resize

        while not window.should_close() and not self._stop_event.is_set():
            window.process_events()

            if needs_reconfigure:
                needs_reconfigure = False
                new_w, new_h = window.width, window.height
                if new_w > 0 and new_h > 0:
                    try:
                        self.device.wait_for_idle()
                        surface.unconfigure()
                        surface.configure(width=new_w, height=new_h)
                    except Exception:
                        pass
                continue

            now = time.perf_counter()
            dt = min(now - last_time, 0.1)
            last_time = now
            cam.move(keys_held, dt)

            if not self._has_data:
                time.sleep(0.016)
                continue

            camera = cam.get_camera(self.width, self.height, self.fovy)
            try:
                self._pipeline.render(camera)
                surface_tex = surface.acquire_next_image()
                encoder = self.device.create_command_encoder()
                encoder.blit(surface_tex, self._pipeline.buffers.output_image)
                cmd = encoder.finish()
                self.device.submit_command_buffer(cmd)
                surface.present()
            except Exception:
                time.sleep(0.016)

        try:
            window.close()
        except Exception:
            pass
