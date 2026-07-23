"""Interactive in-process viewer sharing the trainer's GPU device and queue."""
from __future__ import annotations

import math
import os
import threading
import time

import numpy as np
import slangpy as spy

from trainer.dataset import Camera
from trainer.device import SHADERS_DIR
from trainer.gpu import GpuRuntime
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
    """Free-fly camera that stores an explicit orthonormal basis
    (right, up, forward) instead of world-up-locked Euler angles.

    Every rotation is applied incrementally IN THE CAMERA'S CURRENT FRAME:
    mouse-look yaws about the current ``up`` and pitches about the current
    ``right``; roll (R/T) spins about the current ``forward``. Because pan uses
    the live ``up`` — the SAME axis Q/E translate along — a roll stays "baked
    in": the global up reference becomes the camera's local up, so after rolling
    the pan and the up/down movement remain in sync (no separate world-up axis
    to drift out of alignment).

    The constructor still accepts yaw/pitch/roll (for callers and the camera
    tests); those seed the initial basis with the historical world-up
    convention, after which only the basis is tracked.
    """

    def __init__(self, eye=None, yaw=0.0, pitch=0.0, roll=0.0, speed=1.0):
        self.eye = np.array(eye if eye is not None else [0.0, 0.0, -2.0], dtype=np.float64)
        self.speed = speed
        self.sensitivity = 0.2
        self._set_basis_from_euler(yaw, pitch, roll)

    # --- basis construction / maintenance -------------------------------
    def _set_basis_from_euler(self, yaw: float, pitch: float, roll: float) -> None:
        """Seed the (right, up, forward) basis from Euler angles, matching the
        old world-up convention so a given (yaw, pitch, roll) yields the same
        frame it used to."""
        yaw_r = math.radians(yaw)
        pitch_r = math.radians(pitch)
        f = np.array([
            math.cos(pitch_r) * math.sin(yaw_r),
            -math.sin(pitch_r),
            math.cos(pitch_r) * math.cos(yaw_r),
        ], dtype=np.float64)
        world_up = np.array([0.0, 1.0, 0.0])
        r0 = np.cross(world_up, f)
        n = np.linalg.norm(r0)
        # Near-vertical forward makes right ill-defined vs world up; fall back
        # to the x-axis so the basis stays finite (roll is degenerate here).
        r0 = r0 / n if n > 1e-8 else np.array([1.0, 0.0, 0.0])
        u0 = np.cross(f, r0)
        a = math.radians(roll)
        self._fwd = f
        self._right = math.cos(a) * r0 + math.sin(a) * u0
        self._up = math.cos(a) * u0 - math.sin(a) * r0
        self._orthonormalize()

    def _orthonormalize(self) -> None:
        """Re-establish a right-handed orthonormal basis (right x up = forward)
        from the current forward + right, killing accumulated float drift."""
        f = self._fwd / (np.linalg.norm(self._fwd) + 1e-12)
        r = self._right - f * np.dot(self._right, f)
        r = r / (np.linalg.norm(r) + 1e-12)
        u = np.cross(f, r)          # right x up = forward  ->  up = forward x right
        self._fwd, self._right, self._up = f, r, u

    @staticmethod
    def _rotate_about(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rodrigues rotation of ``v`` about unit ``axis`` by ``angle`` (rad)."""
        c, s = math.cos(angle), math.sin(angle)
        return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)

    @property
    def forward(self) -> np.ndarray:
        return self._fwd

    @property
    def right(self) -> np.ndarray:
        return self._right

    @property
    def up(self) -> np.ndarray:
        return self._up

    def roll_by(self, degrees: float) -> None:
        """Roll about the current forward axis (R/T). Positive rolls the view's
        right toward its up; the up vector rolls WITH the camera, so subsequent
        pan and Q/E keep using this rolled up (they stay in sync)."""
        a = math.radians(degrees)
        self._right = self._rotate_about(self._right, self._fwd, a)
        self._up = self._rotate_about(self._up, self._fwd, a)
        self._orthonormalize()

    def move(self, keys: set, dt: float):
        vel = self.speed * dt
        if "w" in keys:
            self.eye += self._fwd * vel
        if "s" in keys:
            self.eye -= self._fwd * vel
        if "a" in keys:
            self.eye -= self._right * vel
        if "d" in keys:
            self.eye += self._right * vel
        # Q/E translate along the CURRENT (roll-aware) up, so they track roll.
        if "q" in keys:
            self.eye += self._up * vel
        if "e" in keys:
            self.eye -= self._up * vel

    def rotate(self, dx: float, dy: float):
        """Mouse-look in the camera's CURRENT frame: yaw about the live ``up``,
        pitch about the live ``right``. Both axes are the actual basis (roll
        included), so look-around is screen-relative at any roll and stays in
        sync with the Q/E up/down axis — panning after a roll no longer fights a
        separate world-up reference."""
        yaw = math.radians(dx * self.sensitivity)     # drag-right -> look right
        pitch = math.radians(-dy * self.sensitivity)  # drag-up -> look up
        # Yaw about current up.
        self._right = self._rotate_about(self._right, self._up, yaw)
        self._fwd = self._rotate_about(self._fwd, self._up, yaw)
        # Pitch about (new) current right.
        self._up = self._rotate_about(self._up, self._right, pitch)
        self._fwd = self._rotate_about(self._fwd, self._right, pitch)
        self._orthonormalize()

    def set_from_view_matrix(self, view: np.ndarray) -> None:
        """Snap to a world->view matrix (e.g. a dataset camera pose) so
        subsequent WASD/look/roll continue from it. Our view rows are
        (right, up, forward) and the translation is -R @ eye, so eye = -R^T @ t
        and the basis is read straight off the rows — roll included, no Euler
        round-trip needed."""
        view = np.asarray(view, dtype=np.float64)
        R = view[:3, :3]
        t = view[:3, 3]
        self.eye = (-R.T @ t).astype(np.float64)
        self._right = R[0].astype(np.float64)
        self._up = R[1].astype(np.float64)
        self._fwd = R[2].astype(np.float64)
        self._orthonormalize()

    def get_camera(self, width: int, height: int, fovy: float = 45.0,
                   aspect: float | None = None) -> Camera:
        """Build a Camera at render resolution (width, height). ``aspect``
        overrides the projection aspect ratio when the display surface has a
        different aspect than the render buffer (e.g. a fixed-size render blit
        into a resized window) — passing the window aspect here keeps geometry
        undistorted instead of stretched."""
        view = _build_view_matrix(
            self.eye.astype(np.float32),
            self._fwd.astype(np.float32),
            self._up.astype(np.float32),
        )
        proj = _build_proj_matrix(fovy, aspect if aspect is not None else width / height)
        return Camera(view_matrix=view, proj_matrix=proj, image_size=(width, height))


class _ShaderWatcher:
    """Polls .slang files for mtime changes. Calls device.reload_all_programs()
    when a change is detected (after a debounce period)."""

    DEBOUNCE_S = 0.4
    POLL_INTERVAL = 0.5

    def __init__(self, device: spy.Device, pipeline: Pipeline, runtime: GpuRuntime):
        self._device = device
        self._pipeline = pipeline
        self._runtime = runtime
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
            with self._runtime.queue_scope():
                self._runtime.wait_for_idle()
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
        self.runtime = GpuRuntime(device, self.gpu_lock)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Set by the in-panel "Start Training" button. The trainer blocks on
        # start_requested() before entering its loop when it wants the user to
        # inspect the initialized gaussians first. Same-process handshake, so a
        # plain Event — no shared-memory flag needed (cf. the old two-process
        # splat stream).
        self._start_event = threading.Event()
        # Reflects the trainer's state so the panel can show "training" vs
        # "inspecting init". The trainer sets this True once it starts looping.
        self._training_active = False
        # The viewer only RENDERS a single static view (never trains/densifies),
        # so a splat covers far fewer tiles than the training pipeline's
        # max_splats*16 headroom assumes. Sizing the viewer's sort-pair buffers
        # at that headroom doubles the whole allocation (~2.5 GB of sort buffers
        # alone at 6.5M splats) and OOMs the device on top of the training
        # pipeline — surfacing as a generic "Vulkan returned a failure" at
        # startup. So we cap below max_splats*16.
        #
        # NOTE ON OVERFLOW SAFETY: bin_emit now hard-guards its writes against
        # this cap and prepareDispatchArgs clamps scan_total to it, so a
        # viewpoint that emits MORE pairs than this simply drops the excess
        # (a few splats go missing from that one frame) instead of overrunning
        # the sort buffers into tile_start/tile_end — which previously produced
        # view-dependent alternating BLACK TILES from certain camera positions
        # (close-ups where many large splats each cover many tiles). The cap is
        # therefore a quality/VRAM knob, not a crash boundary.
        #
        # The old hard 4M ceiling was too tight for close-ups of multi-million
        # splat scenes (a near view can put most splats on-screen, each
        # covering many tiles). Scale generously with max_splats — ~6 pairs per
        # splat, which comfortably covers realistic close-up coverage — with an
        # 8M ceiling so a 6.5M-splat scene gets ~8M pairs (~192 MB of sort
        # buffers) rather than being starved at 4M. Pass an explicit
        # max_sort_pairs to override for tight-VRAM setups.
        if max_sort_pairs is None:
            max_sort_pairs = min(5_000_000, max(2_000_000, max_splats * 6))
        self._pipeline = Pipeline(
            device=device, width=width, height=height, max_splats=max_splats,
            max_sort_pairs=max_sort_pairs,
            gpu_lock=self.gpu_lock,
            # Skip the ~3.5 GB of training-only buffers (grads/adam/densify
            # scratch/loss textures) — critical to avoid OOMing the device
            # when the viewer + trainer share it. Also skips compiling ~30
            # training-only kernels, speeding up viewer startup.
            viewer_only=True,
        )
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
        batch = self.runtime.command_batch("viewer_update")
        encoder = batch.encoder
        encoder.copy_buffer(dst_g, 0, src_g, 0, src_g.size)
        encoder.copy_buffer(dst_n, 0, src_n, 0, src_n.size)
        self.runtime.submit(batch)
        self._has_data = True

    def start_requested(self) -> bool:
        """True once the user pressed 'Start Training' in the viewer panel."""
        return self._start_event.is_set()

    def wait_for_start(self, poll: float = 0.05) -> None:
        """Block until 'Start Training' is pressed or the viewer window closes
        (so a closed viewer can't hang the trainer forever)."""
        while not self._start_event.is_set():
            if self._stop_event.is_set() or (self._thread and not self._thread.is_alive()):
                break
            time.sleep(poll)

    def set_training_active(self, active: bool = True) -> None:
        """Trainer tells the viewer it has begun (drives the panel label)."""
        self._training_active = active

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

        # --- ImGui panel: status + Start Training button ---
        ui = spy.ui.Context(self.device)
        ui_panel = spy.ui.Window(ui.screen, "Trainer", position=spy.float2(10, 10),
                                 size=spy.float2(240, 0))
        spy.ui.Text(ui_panel, "WASD move | QE up/down | drag look | R/T roll")
        status_text = spy.ui.Text(ui_panel, "inspecting init")

        def _on_start():
            self._start_event.set()
        spy.ui.Button(ui_panel, "Start Training", callback=_on_start)

        cam = FlyCamera()
        keys_held: set = set()
        mouse_captured = False
        last_mouse_x, last_mouse_y = 0.0, 0.0
        last_time = time.perf_counter()

        # Q/E move along the camera's own up vector (roll-aware), so up/down
        # follows whatever roll R/T has applied. R/T roll the view about the
        # forward axis.
        key_map = {
            spy.KeyCode.w: "w", spy.KeyCode.a: "a",
            spy.KeyCode.s: "s", spy.KeyCode.d: "d",
            spy.KeyCode.q: "q", spy.KeyCode.e: "e",
            spy.KeyCode.r: "r", spy.KeyCode.t: "t",  # roll left / right
        }
        roll_speed = 60.0  # degrees/sec while R/T held

        def on_key(event):
            nonlocal keys_held, mouse_captured
            if ui.handle_keyboard_event(event):
                return  # UI consumed it (typing in a widget)
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
            if ui.handle_mouse_event(event):
                mouse_captured = False  # don't drive the camera from panel clicks
                return
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
        watcher = _ShaderWatcher(
            self.device, self._pipeline, self.runtime) if self.hot_reload else None

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
                        with self.runtime.queue_scope():
                            self.runtime.wait_for_idle()
                            surface.unconfigure()
                            self.runtime.wait_for_idle()
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
            if "r" in keys_held:
                cam.roll_by(+roll_speed * dt)
            if "t" in keys_held:
                cam.roll_by(-roll_speed * dt)

            status_text.text = ("training" if self._training_active
                                 else "inspecting init — press Start Training")

            if not self._has_data:
                # Still pump the UI so the Start button is clickable before the
                # first gaussian snapshot arrives.
                try:
                    with self.runtime.queue_scope():
                        surface_tex = surface.acquire_next_image()
                        if surface_tex is not None:
                            batch = self.runtime.command_batch("viewer_ui")
                            encoder = batch.encoder
                            ui.begin_frame(window.width, window.height)
                            ui.end_frame(surface_tex, encoder)
                            self.runtime.submit(batch, after_submit=surface.present)
                except Exception:
                    pass
                time.sleep(0.016)
                continue

            camera = cam.get_camera(self.width, self.height, self.fovy)
            try:
                # One encoder per frame → ONE VkQueue submit per frame. Setting
                # pipeline._encoder routes every _run_forward dispatch (~40
                # kernels: clear/bin/scan/radix×8/tile_ranges/bucket/forward)
                # through this shared encoder instead of each doing its own
                # submit. Before this collapse the viewer thread issued ~40
                # submits per frame and held gpu_lock for the full frame.
                # Blit + UI share the encoder, so the whole frame is one
                # commit.
                #
                # acquire_next_image is Vulkan-swapchain-only (NOT a queue
                # submit), so it can — and must — happen outside gpu_lock;
                # otherwise vsync wait pins the lock for ~16ms per frame
                # while the queue is idle, starving the trainer thread.
                surface_tex = surface.acquire_next_image()
                batch = self.runtime.command_batch("viewer_frame")
                encoder = batch.encoder
                self._pipeline._encoder = encoder
                try:
                    self._pipeline._run_forward(camera)
                finally:
                    self._pipeline._encoder = None
                if surface_tex is not None:
                    encoder.blit(surface_tex, self._pipeline.buffers.output_image)
                    # Draw the panel on top, same swapchain image + encoder.
                    ui.begin_frame(window.width, window.height)
                    ui.end_frame(surface_tex, encoder)
                # submit_command_buffer is async (queues work + returns in
                # ~30us); present() is a queue op too and MUST be under
                # gpu_lock (Vulkan requires external synchronization of all
                # queue ops on the same VkQueue).
                present = surface.present if surface_tex is not None else None
                submit = self.runtime.submit(batch, after_submit=present)
                # Backpressure: keep at most ONE viewer frame in flight on
                # the shared VkQueue. Without this, at 60 fps the viewer
                # queues 5+ frames between trainer iterations; Vulkan
                # executes the queue in-order, so each trainer iter waits
                # for all queued viewer frames to drain before its own
                # work can begin — head-of-line-blocking the trainer down
                # to ~5 it/s. Waiting on this submit before queueing the
                # next viewer frame lets the trainer's submits interleave
                # cleanly (viewer takes ~1 frame slot; trainer takes the
                # next). Also naturally caps viewer FPS at min(vsync, GPU
                # capability while sharing with trainer).
                self.runtime.wait_for_submit(submit)
            except Exception as e:
                if not getattr(self, '_err_printed', False):
                    print(f"[Viewer] frame error: {e}")
                    self._err_printed = True
                time.sleep(0.016)

        try:
            window.close()
        except Exception:
            pass