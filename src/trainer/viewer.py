"""Interactive viewer for live training inspection.

Has its own Pipeline (separate buffers/scratch) and copies the gaussian
buffer + count from the training pipeline every N iterations. Completely
decoupled — no shared mutable state, no locks needed.

Shader hot-reload: file changes are polled and device.reload_all_programs()
is called to update kernels in-place.

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
import os
import threading
import time

import numpy as np
import slangpy as spy

from trainer.dataset import Camera
from trainer.device import SHADERS_DIR
from trainer.pipeline import Pipeline


def _build_view_matrix(eye: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    # RIGHT-HANDED basis (det +1), matching the COLMAP view matrices the
    # splats were trained with. The previous r = cross(forward, up) is the
    # negation of the right-handed r = cross(up, forward), which flipped the
    # X axis (det -1) and rendered the scene left-right MIRRORED relative to
    # the trainer's own renders. Using cross(up, forward) fixes the mirror.
    f = forward / np.linalg.norm(forward)
    r = np.cross(up, f)
    r = r / np.linalg.norm(r)
    u = np.cross(f, r)
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
        # cross(up, forward) to match the right-handed _build_view_matrix
        # basis; otherwise D-strafe would move opposite the camera's right.
        f = self.forward
        up = np.array([0.0, 1.0, 0.0])
        r = np.cross(up, f)
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
            self.eye[1] += vel
        if "e" in keys:
            self.eye[1] -= vel

    def rotate(self, dx: float, dy: float):
        # +dx on yaw: in the right-handed basis, screen-right is view +x, and
        # forward gains +x as yaw increases, so drag-right must INCREASE yaw
        # ("look right"). The old -dx was tuned to the pre-fix left-handed
        # (mirrored) basis and now reads as inverted. Pitch is unchanged by the
        # handedness fix (Y axis untouched), so its sign is left as-is.
        self.yaw += dx * self.sensitivity
        self.pitch += -dy * self.sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def set_from_view_matrix(self, view: np.ndarray) -> None:
        """Snap this fly-cam to a world->view matrix (e.g. a dataset camera
        pose). Recovers eye + yaw/pitch so subsequent WASD/look continue
        smoothly from the snapped pose.

        Our view matrix rows are (right, up, forward) and translation is
        -R @ eye, so eye = -R^T @ t and the world-space forward is row 2.
        The forward parameterization is
          f = (cos(p)sin(y), -sin(p), cos(p)cos(y)),
        which inverts to pitch = asin(-f_y), yaw = atan2(f_x, f_z)."""
        view = np.asarray(view, dtype=np.float64)
        R = view[:3, :3]
        t = view[:3, 3]
        self.eye = (-R.T @ t).astype(np.float64)
        f = R[2] / (np.linalg.norm(R[2]) + 1e-12)  # world-space forward
        self.pitch = math.degrees(math.asin(max(-1.0, min(1.0, -f[1]))))
        self.yaw = math.degrees(math.atan2(f[0], f[2]))

    def get_camera(self, width: int, height: int, fovy: float = 45.0,
                   aspect: float | None = None) -> Camera:
        """Build a Camera at render resolution (width, height). ``aspect``
        overrides the projection aspect ratio when the display surface has a
        different aspect than the render buffer (e.g. a fixed-size render blit
        into a resized window) — passing the window aspect here keeps geometry
        undistorted instead of stretched."""
        view = _build_view_matrix(
            self.eye.astype(np.float32),
            self.forward.astype(np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        proj = _build_proj_matrix(fovy, aspect if aspect is not None else width / height)
        return Camera(view_matrix=view, proj_matrix=proj, image_size=(width, height))


class _ShaderWatcher:
    """Polls .slang files for mtime changes. Calls device.reload_all_programs()
    when a change is detected (after a debounce period)."""

    DEBOUNCE_S = 0.4
    POLL_INTERVAL = 0.5

    def __init__(self, device: spy.Device, pipeline: Pipeline):
        self._device = device
        self._pipeline = pipeline
        self._mtimes: dict[str, float] = {}
        self._last_poll = 0.0
        self._change_at: float | None = None
        self._scan(initial=True)

    def _scan(self, initial: bool = False) -> bool:
        changed = False
        for root, _, files in os.walk(SHADERS_DIR):
            for f in files:
                if not f.endswith(".slang"):
                    continue
                path = os.path.join(root, f)
                try:
                    mt = os.path.getmtime(path)
                except OSError:
                    continue
                if not initial and self._mtimes.get(path) != mt:
                    changed = True
                self._mtimes[path] = mt
        return changed

    def poll(self):
        """Call each frame. Triggers reload_all_programs after debounce."""
        now = time.monotonic()
        if now - self._last_poll >= self.POLL_INTERVAL:
            self._last_poll = now
            if self._scan():
                self._change_at = now

        if self._change_at is not None and now - self._change_at >= self.DEBOUNCE_S:
            self._change_at = None
            self._device.wait_for_idle()
            self._device.reload_all_programs()
            try:
                self._pipeline.on_hot_reload()
                print("[Viewer] shaders reloaded")
            except Exception as e:
                print(f"[Viewer] on_hot_reload failed: {e}")


class Viewer:
    def __init__(self, device: spy.Device, width: int = 960, height: int = 540,
                 max_splats: int = 500_000, fovy: float = 45.0,
                 hot_reload: bool = False, max_sort_pairs: int | None = None,
                 gpu_lock=None):
        self.device = device
        self.width = width
        self.height = height
        self.max_splats = max_splats
        self.fovy = fovy
        self.hot_reload = hot_reload
        # Shared GPU-queue lock (see Pipeline.gpu_lock). The viewer render loop
        # runs on a background thread and MUST hold this lock around every
        # queue operation (submit / present / wait_for_idle / acquire) so it
        # never races the training thread's submits on the shared VkQueue.
        import contextlib
        self.gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # The viewer only RENDERS a single static view (never trains/densifies),
        # so a splat covers far fewer tiles than the training pipeline's
        # max_splats*16 headroom assumes. Sizing the viewer's sort-pair buffers
        # at that headroom doubles the whole allocation (~2.5 GB of sort buffers
        # alone at 6.5M splats) and OOMs the device on top of the training
        # pipeline — surfacing as a generic "Vulkan returned a failure" at
        # startup. Cap the viewer at a modest per-splat tile budget.
        if max_sort_pairs is None:
            max_sort_pairs = max(2_000_000, max_splats * 4)
        self._pipeline = Pipeline(
            device=device, width=width, height=height, max_splats=max_splats,
            max_sort_pairs=max_sort_pairs,
        )
        self._pipeline.lambda_ssim = 0.0
        self._pipeline.w_alpha = 0.0
        self._has_data = False
        self._update_pending = False

    def update_gaussians(self, train_pipeline: Pipeline):
        """GPU blit gaussian data from training pipeline to viewer pipeline.

        Both pipelines share the same device, so this is a fast GPU→GPU
        copy with no CPU round-trip. Called from training thread.
        """
        src_g = train_pipeline.buffers.gaussians.storage
        dst_g = self._pipeline.buffers.gaussians.storage
        src_n = train_pipeline.buffers.num_gaussians_counter.storage
        dst_n = self._pipeline.buffers.num_gaussians_counter.storage
        encoder = self.device.create_command_encoder()
        encoder.copy_buffer(dst_g, 0, src_g, 0, src_g.size)
        encoder.copy_buffer(dst_n, 0, src_n, 0, src_n.size)
        cmd = encoder.finish()
        with self.gpu_lock:
            self.device.submit_command_buffer(cmd)
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

        configured = True
        watcher = _ShaderWatcher(self.device, self._pipeline) if self.hot_reload else None

        while not window.should_close() and not self._stop_event.is_set():
            window.process_events()
            if watcher:
                watcher.poll()

            if needs_reconfigure:
                needs_reconfigure = False
                configured = False
                new_w, new_h = window.width, window.height
                if new_w > 0 and new_h > 0:
                    try:
                        with self.gpu_lock:
                            self.device.wait_for_idle()
                            surface.unconfigure()
                            self.device.wait_for_idle()
                            surface.configure(width=new_w, height=new_h)
                        configured = True
                    except Exception as e:
                        print(f"[Viewer] reconfigure failed: {e}")
                continue

            if not configured:
                time.sleep(0.016)
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
                # Hold the shared GPU-queue lock across the ENTIRE frame: the
                # forward pass issues many immediate per-kernel submits (its
                # _dispatch runs with no encoder), and the blit/acquire/present
                # are queue ops too. All of it must be serialized against the
                # training thread's submits on the shared VkQueue.
                with self.gpu_lock:
                    self._pipeline._run_forward(camera)
                    surface_tex = surface.acquire_next_image()
                    encoder = self.device.create_command_encoder()
                    encoder.blit(surface_tex, self._pipeline.buffers.output_image)
                    cmd = encoder.finish()
                    self.device.submit_command_buffer(cmd)
                    surface.present()
            except Exception as e:
                if not getattr(self, '_err_printed', False):
                    print(f"[Viewer] frame error: {e}")
                    self._err_printed = True
                time.sleep(0.016)

        try:
            window.close()
        except Exception:
            pass