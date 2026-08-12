"""Interactive in-process viewer sharing the trainer's GPU device and queue."""
from __future__ import annotations

import math
import os
import threading
import time

import numpy as np
import slangpy as spy

from trainer.data.dataset import Camera
from trainer.backend.compute.device import SHADERS_DIR
from trainer.backend.compute.gpu import GpuRuntime
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.config.config import AppearanceConfig, RasterConfig
from trainer.frontend.viewer.viewer_ui import InputMap, ViewerUI


def _build_view_matrix(eye: np.ndarray, forward: np.ndarray, up: np.ndarray) -> np.ndarray:
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


def _fovy_from_proj(proj: np.ndarray) -> float | None:
    """Recover vertical FOV (degrees) from a projection matrix.

    Both ``_build_proj_matrix`` here and the dataset/COLMAP pinhole builders
    put ``cot(fovy/2)`` in the (1,1) entry (for COLMAP that equals
    ``2*fy_pixels/H``). Snapping to a test camera reuses its own vertical FOV so
    the viewer framing is comparable to the eval render at that camera. Returns
    None if the entry is non-positive (degenerate/unset projection)."""
    m11 = float(proj[1, 1])
    if m11 <= 1e-6:
        return None
    return math.degrees(2.0 * math.atan(1.0 / m11))


def _linear_present_format(surface: spy.Surface) -> spy.Format:
    """Choose a NON-sRGB swapchain format for presentation.

    ``output_image`` already holds sRGB-ENCODED color (the renderer composites
    in sRGB — the same values the stored PNGs are written from). Presenting it
    through an sRGB surface makes the blit re-apply the sRGB OETF, double-
    encoding the color -> a washed-out viewer while the PNGs look correct.
    Configuring the surface with the linear twin of its preferred format writes
    those sRGB values to the display verbatim, matching the PNG path."""
    def name(f) -> str:
        return getattr(f, "name", str(f)).lower()

    info = surface.info
    try:
        formats = list(info.formats)
    except Exception:
        formats = []
    preferred = info.preferred_format
    if "srgb" not in name(preferred):
        return preferred  # already linear — nothing to fix

    twins = {
        "bgra8_unorm_srgb": "bgra8_unorm",
        "rgba8_unorm_srgb": "rgba8_unorm",
        "bgrx8_unorm_srgb": "bgrx8_unorm",
    }
    want = twins.get(name(preferred))
    if want is not None:
        for f in formats:
            if name(f) == want:
                return f
    # Fallback: any supported non-sRGB 8-bit unorm, then any non-sRGB format.
    for f in formats:
        if "srgb" not in name(f) and name(f) in ("bgra8_unorm", "rgba8_unorm", "bgrx8_unorm"):
            return f
    for f in formats:
        if "srgb" not in name(f):
            return f
    return preferred


class FlyCamera:

    def __init__(self, eye=None, yaw=0.0, pitch=0.0, roll=0.0, speed=1.0):
        self.eye = np.array(eye if eye is not None else [0.0, 0.0, -2.0], dtype=np.float64)
        self.speed = speed
        self.sensitivity = 0.2
        self._set_basis_from_euler(yaw, pitch, roll)
        # Reference "up" for roll-free mouse-look: yaw rotates about this axis
        # and right is always rebuilt perpendicular to it, so looking around
        # never rolls. R/T rotate it (persistent roll); Y (set_ref_up_to_current)
        # resets it to the current view. Scene-agnostic — NOT tied to world +Y.
        self._ref_up = self._up.copy()

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

    def _rebuild_from_ref(self) -> None:
        """Rebuild (right, up) from the current forward and ``self._ref_up``,
        forcing right PERPENDICULAR to ref_up (horizon level w.r.t. ref_up).
        All roll lives in ref_up; mouse-look never tilts right out of the ref_up
        plane, so no drift accumulates no matter the drag path."""
        f = self._fwd / (np.linalg.norm(self._fwd) + 1e-12)
        r = np.cross(self._ref_up, f)
        n = np.linalg.norm(r)
        if n < 1e-6:
            # forward ~parallel to ref_up: keep previous right, projected off f.
            r = self._right - f * np.dot(self._right, f)
            n = np.linalg.norm(r)
            if n < 1e-6:
                r = np.array([1.0, 0.0, 0.0]) - f * float(f[0])
                n = np.linalg.norm(r)
        r = r / (n + 1e-12)
        self._fwd, self._right, self._up = f, r, np.cross(f, r)

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
        """Roll about the forward axis (R/T). Rotates the reference up so the
        tilt PERSISTS through subsequent mouse-look (which levels to ref_up)."""
        ref = self._rotate_about(self._ref_up, self._fwd, math.radians(degrees))
        self._ref_up = ref / (np.linalg.norm(ref) + 1e-12)
        self._rebuild_from_ref()

    def set_ref_up_to_current(self) -> None:
        """Y key: adopt the current view's up as the new up reference. Mouse-look
        then keeps the horizon level to what you're looking at right now, and any
        roll in the current view becomes the new zero-roll reference."""
        self._ref_up = self._up / (np.linalg.norm(self._up) + 1e-12)

    def move(self, keys: set, dt: float):
        # Move at the mouse-wheel-configured speed (see adjust_speed).
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

    def adjust_speed(self, steps: float) -> None:
        """Scale the movement speed by mouse-wheel ``steps`` (~25% per notch,
        multiplicative), clamped to a sane range. All WASD/QE motion uses this
        one speed."""
        if steps:
            self.speed *= 1.25 ** steps
            self.speed = float(max(1e-3, min(self.speed, 1e5)))

    def rotate(self, dx: float, dy: float):
        """Mouse-look about the reference up (self._ref_up): yaw about ref_up,
        pitch about the current right, then rebuild right PERPENDICULAR to
        ref_up. Looking around can never roll the view no matter the drag path
        (right stays in the ref_up plane) — only R/T and Y change the roll
        reference. Levels to the SCENE's up (whatever ref_up holds), not +Y."""
        yaw = math.radians(dx * self.sensitivity)     # drag-right -> look right
        pitch = math.radians(-dy * self.sensitivity)  # drag-up -> look up
        f = self._rotate_about(self._fwd, self._ref_up, yaw)
        r = np.cross(self._ref_up, f)
        n = np.linalg.norm(r)
        r = r / n if n > 1e-6 else self._right
        f_pitched = self._rotate_about(f, r, pitch)
        # Stop pitch from crossing the ref_up pole (keeps right well-defined).
        if np.linalg.norm(np.cross(self._ref_up, f_pitched)) > math.sin(math.radians(0.5)):
            f = f_pitched
        self._fwd = f
        self._rebuild_from_ref()

    def set_from_view_matrix(self, view: np.ndarray) -> None:
        view = np.asarray(view, dtype=np.float64)
        R = view[:3, :3]
        t = view[:3, 3]
        self.eye = (-R.T @ t).astype(np.float64)
        self._right = R[0].astype(np.float64)
        self._up = R[1].astype(np.float64)
        self._fwd = R[2].astype(np.float64)
        self._orthonormalize()
        # Level subsequent mouse-look to this camera's up (the scene's up as it
        # sees it), so looking around from a snapped test cam stays upright.
        self._ref_up = self._up.copy()

    def get_camera(self, width: int, height: int, fovy: float = 45.0,
                   aspect: float | None = None) -> Camera:
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

    def __init__(self, device: spy.Device, pipeline: TrainingPipeline, runtime: GpuRuntime):
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
                 gpu_lock=None, sh_degree: int = 0,
                 mip_enabled: bool = False, mip_kernel_size: float = 0.3,
                 test_cameras: list[tuple[str, Camera]] | None = None):
        self.device = device
        self.width = width
        self.height = height
        self.max_splats = max_splats
        self.fovy = fovy
        self.hot_reload = hot_reload
        self.sh_degree = sh_degree
        # Held-out test cameras as (name, Camera) for the C/V snap feature;
        # empty when the viewer runs without a scene (standalone / tests).
        self._test_cameras: list[tuple[str, Camera]] = list(test_cameras or [])

        import contextlib
        self.gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()
        self.runtime = GpuRuntime(device, self.gpu_lock)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._start_event = threading.Event()
        # Set == training paused; the trainer blocks on wait_while_paused() each
        # iteration. Toggled by the viewer's Pause/Resume button.
        self._pause_event = threading.Event()
        # Reflects the trainer's state so the panel can show "training" vs
        # "inspecting init". The trainer sets this True once it starts looping.
        self._training_active = False

        # NOTE can cut out splats
        if max_sort_pairs is None:
            max_sort_pairs = min(12_000_000, max(2_000_000, max_splats * 6))
        self._pipeline = TrainingPipeline(
            device=device, width=width, height=height, max_splats=max_splats,
            max_sort_pairs=max_sort_pairs,
            gpu_lock=self.gpu_lock,
            viewer_only=True,
            appearance_config=AppearanceConfig(sh_degree=sh_degree),
            # Mirror the trainer's Mip-Splatting raster policy so the viewer
            # low-passes the same tiny high-frequency gaussians eval does;
            # without this those splats render raw and show aggressive
            # higher-order SH color the eval render suppresses.
            raster_config=RasterConfig(mip_enabled=mip_enabled,
                                       kernel_size=mip_kernel_size),
        )

        if sh_degree > 0:
            self._pipeline.set_active_sh_degree(sh_degree)

        self._has_data = False
        self._update_pending = False

    def update_gaussians(self, train_pipeline: TrainingPipeline):
        src_g = train_pipeline.buffers.gaussians.storage
        dst_g = self._pipeline.buffers.gaussians.storage
        src_n = train_pipeline.buffers.num_gaussians_counter.storage
        dst_n = self._pipeline.buffers.num_gaussians_counter.storage
        batch = self.runtime.command_batch("viewer_update")
        encoder = batch.encoder
        encoder.copy_buffer(dst_g, 0, src_g, 0, src_g.size)
        encoder.copy_buffer(dst_n, 0, src_n, 0, src_n.size)
        if train_pipeline.buffers.sh_enabled and self._pipeline.buffers.sh_enabled:
            src_r = train_pipeline.buffers.features_rest.storage
            dst_r = self._pipeline.buffers.features_rest.storage
            encoder.copy_buffer(dst_r, 0, src_r, 0, min(src_r.size, dst_r.size))
        # Mip-Splatting: copy the per-gaussian 3D smoothing filter so the viewer
        # applies the SAME low-pass eval does. filter_3D depends on gaussian
        # positions, so it must be refreshed with every snapshot (post-densify).
        if self._pipeline.raster_config.mip_enabled:
            src_f = getattr(train_pipeline.buffers, "filter_3D", None)
            dst_f = getattr(self._pipeline.buffers, "filter_3D", None)
            if src_f is not None and dst_f is not None:
                encoder.copy_buffer(dst_f.storage, 0, src_f.storage, 0,
                                    min(src_f.storage.size, dst_f.storage.size))
        self.runtime.submit(batch)
        # Mirror the trainer's CURRENT active SH degree so the viewer shows the
        # same bands as eval. The degree schedule ramps during training; pinning
        # the viewer at max over-shows half-trained higher bands eval (at the
        # scheduled degree) does not.
        self._pipeline.set_active_sh_degree(train_pipeline._active_sh_degree)
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

    def is_paused(self) -> bool:
        """True while the Pause button has training suspended."""
        return self._pause_event.is_set()

    def toggle_pause(self) -> None:
        """Flip pause/resume (called from the panel button)."""
        (self._pause_event.clear if self._pause_event.is_set()
         else self._pause_event.set)()

    def wait_while_paused(self, poll: float = 0.05) -> None:
        """Block the caller (the trainer) while paused, releasing if the viewer
        window closes so a paused+closed viewer can't hang training forever."""
        while self._pause_event.is_set():
            if self._stop_event.is_set() or (self._thread and not self._thread.is_alive()):
                break
            time.sleep(poll)

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
        # Present in a NON-sRGB format: output_image already holds sRGB-encoded
        # color (renderer composites in sRGB, like the stored PNGs), so the blit
        # must not re-apply the sRGB OETF. A default/sRGB surface double-encodes
        # -> washed-out viewer while the PNGs look correct.
        present_format = _linear_present_format(surface)
        print(f"[Viewer] surface preferred={surface.info.preferred_format} "
              f"-> presenting as {present_format}")
        surface.configure(width=self.width, height=self.height, format=present_format)
        needs_reconfigure = False

        cam = FlyCamera()

        # Interaction state shared by the input callbacks and the render loop.
        #   snap_idx : current test-camera index, -1 == free-fly
        #   snapped  : exact-proj Camera to render while parked (None => fly-cam)
        #   grab     : request to save the next frame to a PNG (P key)
        #   syncing  : guard so programmatic combo updates don't re-fire select
        st = {"snap_idx": -1, "snapped": None, "grab": False, "syncing": False}
        test_cams = self._test_cameras
        cam_combo = None  # assigned in the panel build below when cameras exist

        # --- Actions driven FROM the UI / keyboard. ---
        def apply_snap(idx: int) -> None:
            """Snap to test-camera ``idx`` (-1 => free-fly). Single source of
            truth for C/V, the camera dropdown, and 'left on manual nav'; keeps
            the dropdown selection in sync."""
            if idx < 0 or not test_cams:
                st["snap_idx"], st["snapped"] = -1, None
                ui.set("camera", "Free-fly")
            else:
                idx %= len(test_cams)
                name, tcam = test_cams[idx]
                cam.set_from_view_matrix(tcam.view_matrix)  # so WASD continues here
                fovy = _fovy_from_proj(tcam.proj_matrix)
                if fovy is not None:
                    self.fovy = fovy
                # Render the test camera EXACTLY (view+proj) when its resolution
                # matches the render target; else fall back to the fly-cam
                # perspective. Exact proj removes the last raster-side difference
                # vs eval (principal point, exact fx/fy).
                st["snap_idx"] = idx
                st["snapped"] = (tcam if tuple(tcam.image_size) == (self.width, self.height)
                                 else None)
                ui.set("camera", f"Cam {idx + 1}/{len(test_cams)}: {name}")
                print(f"[Viewer] snapped to test cam {idx + 1}/{len(test_cams)} "
                      f"({name}), exact_proj={st['snapped'] is not None}, fovy={self.fovy:.1f} deg")
            if cam_combo is not None:  # reflect state in the dropdown (no re-fire)
                st["syncing"] = True
                cam_combo.value = st["snap_idx"] + 1
                st["syncing"] = False

        def snap_delta(step: int) -> None:
            if not test_cams:
                print("[Viewer] no test cameras available to snap to")
                return
            apply_snap((st["snap_idx"] + step) % len(test_cams))

        def leave_snap() -> None:
            """Manual navigation drops the 'parked on a test cam' state."""
            if st["snap_idx"] != -1:
                apply_snap(-1)

        def on_cam_select(value: int) -> None:
            if not st["syncing"]:            # ignore our own programmatic updates
                apply_snap(int(value) - 1)   # 0 => free-fly, k => camera k-1

        def on_pause_toggle() -> None:
            self.toggle_pause()
            pause_btn.label = "Resume training" if self.is_paused() else "Pause training"

        def set_up_ref() -> None:
            cam.set_ref_up_to_current()
            print("[Viewer] up reference set to current view")

        def save_frame() -> None:
            """Write the last rendered frame to a PNG for offline diffing against
            the eval output (outputs/<run>/cams/cam_XX/...)."""
            try:
                from PIL import Image
                img = self._pipeline.read_output_image()[..., :3]
                png = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
                if st["snap_idx"] != -1:
                    name = test_cams[st["snap_idx"]][0]
                    label = "testcam_" + name.replace("/", "_").replace("\\", "_")
                else:
                    name, label = "(free-fly)", "freefly"
                path = os.path.abspath(f"viewer_grab_{label}.png")
                Image.fromarray(png, mode="RGB").save(path)
                print(f"[Viewer] saved frame for {name} -> {path}")
            except Exception as e:
                print(f"[Viewer] frame grab failed: {e}")

        # --- Panel. Real widgets (dropdown / buttons) + live fields; EXTEND by
        #     adding a control here and, if it shows state, a ui.set(...) call. ---
        ui = ViewerUI(self.device, title="3DGS Viewer", anchor="left", thickness=300.0)
        ui.text("CAMERA")
        if test_cams:
            items = ["Free-fly"] + [f"{i + 1:02d}  {n}" for i, (n, _) in enumerate(test_cams)]
            cam_combo = ui.combo("snap", items, 0, on_cam_select)
        else:
            ui.text("  (no test cameras)")
        ui.field("camera", "Free-fly")
        ui.field("speed", f"Move speed: {cam.speed:.2f}")
        ui.text("")
        ui.text("TRAINING")
        ui.field("status", "inspecting init")
        start_btn = ui.button("Start Training", self._start_event.set)
        pause_btn = ui.button("Pause training", on_pause_toggle)
        ui.text("")
        ui.text("CONTROLS")
        ui.text("  WASD / QE  move      drag  look")
        ui.text("  R / T      roll      Y     set up")
        ui.text("  C / V      snap cam  P     save PNG")
        ui.text("  wheel      move speed")

        # --- Keyboard bindings. EXTEND controls by adding a hold/press line. ---
        MOVE_TAGS = {"w", "a", "s", "d", "q", "e", "r", "t"}
        roll_speed = 60.0  # deg/sec while R/T held
        mouse = {"captured": False, "x": 0.0, "y": 0.0}
        keys = InputMap().hold_many([
            (spy.KeyCode.w, "w"), (spy.KeyCode.a, "a"), (spy.KeyCode.s, "s"),
            (spy.KeyCode.d, "d"), (spy.KeyCode.q, "q"), (spy.KeyCode.e, "e"),
            (spy.KeyCode.r, "r"), (spy.KeyCode.t, "t"),  # R/T roll
        ])
        keys.press(spy.KeyCode.c, lambda: snap_delta(+1))
        keys.press(spy.KeyCode.v, lambda: snap_delta(-1))
        keys.press(spy.KeyCode.p, lambda: st.__setitem__("grab", True))
        keys.press(spy.KeyCode.y, set_up_ref)
        keys.press(spy.KeyCode.escape, lambda: mouse.__setitem__("captured", False))

        # --- Mouse: look (drag) + scroll-to-speed. Camera control, not a widget,
        #     so it stays here; the UI only reports whether it ate the event. ---
        def on_key(event):
            if ui.handle_keyboard(event):
                return  # UI consumed it (typing in a widget)
            keys.handle(event)

        def on_mouse(event):
            if ui.handle_mouse(event):
                mouse["captured"] = False  # don't drive the camera from panel clicks
                return
            if event.is_scroll():
                cam.adjust_speed(float(event.scroll.y))
                ui.set("speed", f"Move speed: {cam.speed:.2f}")
                return
            if event.is_button_down():
                mouse.update(captured=True, x=event.pos[0], y=event.pos[1])
            elif event.is_button_up():
                mouse["captured"] = False
            elif event.is_move() and mouse["captured"]:
                dx = event.pos[0] - mouse["x"]
                dy = event.pos[1] - mouse["y"]
                mouse["x"], mouse["y"] = event.pos[0], event.pos[1]
                cam.rotate(dx, dy)
                leave_snap()

        def on_resize(w, h):
            nonlocal needs_reconfigure
            needs_reconfigure = True

        window.on_keyboard_event = on_key
        window.on_mouse_event = on_mouse
        window.on_resize = on_resize

        configured = True
        watcher = _ShaderWatcher(
            self.device, self._pipeline, self.runtime) if self.hot_reload else None
        last_time = time.perf_counter()

        while not window.should_close() and not self._stop_event.is_set():
            window.process_events()
            if watcher:
                watcher.poll()

            # Preserve the render's aspect ratio on resize. output_image is a
            # FIXED (self.width x self.height) render, and a full-extent blit
            # onto a surface with a different aspect is exactly what stretched
            # the image. Fit the render aspect inside the requested window and
            # SNAP the window to it, so the swapchain we blit into always shares
            # the render aspect (uniform scale, no distortion). The >1px guard
            # stops a resize/rounding feedback loop.
            if needs_reconfigure:
                needs_reconfigure = False
                configured = False
                req_w, req_h = window.width, window.height
                if req_w > 0 and req_h > 0:
                    ar = self.width / self.height
                    if req_w / req_h > ar:       # too wide -> clamp width to height
                        cw, ch = max(1, round(req_h * ar)), req_h
                    else:                         # too tall -> clamp height to width
                        cw, ch = req_w, max(1, round(req_w / ar))
                    try:
                        if abs(cw - req_w) > 1 or abs(ch - req_h) > 1:
                            window.resize(cw, ch)  # snap; re-triggers on_resize once
                        with self.runtime.queue_scope():
                            self.runtime.wait_for_idle()
                            surface.unconfigure()
                            self.runtime.wait_for_idle()
                            surface.configure(width=cw, height=ch,
                                              format=present_format)
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
            cam.move(keys.held, dt)
            if "r" in keys.held:
                cam.roll_by(+roll_speed * dt)
            if "t" in keys.held:
                cam.roll_by(-roll_speed * dt)
            if keys.held & MOVE_TAGS:
                leave_snap()

            paused = self.is_paused()
            ui.set("status", ("paused" if paused else "training") if self._training_active
                   else "inspecting init — press Start Training")
            start_btn.enabled = not self._training_active   # only before training
            pause_btn.enabled = self._training_active        # only during training
            ui.layout(window.width, window.height)  # keep the panel docked

            if not self._has_data:
                # Pump the UI so the Start button works before the first snapshot.
                try:
                    with self.runtime.queue_scope():
                        surface_tex = surface.acquire_next_image()
                        if surface_tex is not None:
                            batch = self.runtime.command_batch("viewer_ui")
                            encoder = batch.encoder
                            ui.begin(window.width, window.height)
                            ui.end(surface_tex, encoder)
                            self.runtime.submit(batch, after_submit=surface.present)
                except Exception:
                    pass
                time.sleep(0.016)
                continue

            # Parked on a test camera -> render it EXACTLY (pixel-faithful to
            # eval); otherwise the free-fly perspective.
            camera = (st["snapped"] if st["snapped"] is not None
                      else cam.get_camera(self.width, self.height, self.fovy))
            try:
                surface_tex = surface.acquire_next_image()
                batch = self.runtime.command_batch("viewer_frame")
                encoder = batch.encoder
                self._pipeline._encoder = encoder
                try:
                    self._pipeline.rasterize(camera)
                finally:
                    self._pipeline._encoder = None
                if surface_tex is not None:
                    encoder.blit(surface_tex, self._pipeline.buffers.output_image)
                    # Draw the panel on top, same swapchain image + encoder.
                    ui.begin(window.width, window.height)
                    ui.end(surface_tex, encoder)

                present = surface.present if surface_tex is not None else None
                submit = self.runtime.submit(batch, after_submit=present)

                self.runtime.wait_for_submit(submit)
                if st["grab"]:
                    st["grab"] = False
                    save_frame()
            except Exception as e:
                if not getattr(self, '_err_printed', False):
                    print(f"[Viewer] frame error: {e}")
                    self._err_printed = True
                time.sleep(0.016)

        try:
            window.close()
        except Exception:
            pass