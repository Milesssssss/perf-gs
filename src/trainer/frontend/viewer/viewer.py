"""Interactive in-process viewer sharing the trainer's GPU device and queue."""
from __future__ import annotations

import math
import os
import threading
import time
from collections.abc import Callable

import numpy as np
import slangpy as spy

from trainer.backend.compute.device import SHADERS_DIR
from trainer.backend.compute.gpu import GpuRuntime
from trainer.backend.render.pipeline import TrainingPipeline
from trainer.config.config import AppearanceConfig, RasterConfig
from trainer.data.dataset import Camera
from trainer.frontend.viewer.session import MODE_HUB, Session
from trainer.frontend.viewer.ui import InputMap, Panel
from trainer.frontend.viewer.ui.sections import (
    Actions,
    CameraControls,
    configuration_section,
    controls_section,
    dataset_section,
    diagnostics_section,
    header,
    loss_section,
    result_section,
    training_section,
    viewer_section,
)

# Backdrop behind the panel while there is no scene to show. Written straight
# into the (non-sRGB) present format, so this is already display-encoded color.
BACKDROP = spy.float4(0.055, 0.065, 0.080, 1.0)


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


def _copy_splats(encoder, src, dst, count: int, capacity: int) -> None:
    """Copy the first ``count`` splats of a per-splat buffer, not the allocation.

    The per-splat stride comes from the buffer's own size: every one of these is
    allocated as ``capacity`` (``--max-splats``) elements, and their strides differ
    (56 B of ``Gaussian3D``, the half-packed ``features_rest``, 4 B of
    ``filter_3D``), so deriving it here keeps the layouts out of this function.
    ``dst.size`` is respected too — the viewer may be configured with a different
    capacity than the trainer.
    """
    stride = src.size // max(1, capacity)
    size = min(count * stride, src.size, dst.size)
    if size > 0:
        encoder.copy_buffer(dst, 0, src, 0, size)


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


#: Key tags whose motion drops the "parked on a test camera" state.
MOVE_TAGS = frozenset({"w", "a", "s", "d", "q", "e", "r", "t"})
ROLL_SPEED_DEG_S = 60.0

# Scene re-renders per second while FREE-FLYING during training (see
# Viewer._scene_due). A viewer frame costs about what a training step does, so an
# unthrottled ~60 fps was taking most of the GPU from the trainer; 15 is still
# fluid to fly with. Parked on a test camera the rate is zero — the scene is
# re-rendered only when new gaussians arrive.
TRAINING_SCENE_FPS = 15.0


class _CameraRig:
    """Free-fly camera + held-out test-camera snapping.

    Owns what the render loop used to keep in an ad-hoc dict (which test camera
    we are parked on, the exact-projection camera to render while parked, the
    active vertical FOV). C/V, the panel dropdown and manual navigation all go
    through :meth:`snap_to`, and the panel binds its readouts to :meth:`names` /
    :meth:`selection` / :meth:`label` — so the selector and the camera can never
    disagree.
    """

    def __init__(self, *, fovy: float, render_size: tuple[int, int],
                 test_cameras: list[tuple[str, Camera]] | None = None,
                 on_change: Callable[[], None] | None = None) -> None:
        self.cam = FlyCamera()
        self.fovy = float(fovy)
        self._render_size = render_size
        self.test_cameras: list[tuple[str, Camera]] = list(test_cameras or [])
        self.index = -1                 # -1 == free-fly
        self.exact: Camera | None = None  # rendered verbatim while parked
        # Called when the PARKED selection changes. While training and parked, the
        # render loop only re-rasterizes on demand (see Viewer._scene_due), so a
        # snap that nothing announced would leave the previous camera on screen.
        self._on_change = on_change or (lambda: None)

    # ---- panel bindings ----
    def names(self) -> list[str]:
        return ["Free-fly"] + [f"{i + 1:02d}  {name}"
                               for i, (name, _) in enumerate(self.test_cameras)]

    def selection(self) -> int:
        """Dropdown index: 0 == free-fly, k == test camera k-1."""
        return self.index + 1

    def label(self) -> str:
        if self.index < 0 or not self.test_cameras:
            return "free-fly"
        name, _ = self.test_cameras[self.index]
        exact = "exact proj" if self.exact is not None else "fly proj"
        return f"cam {self.index + 1}/{len(self.test_cameras)}  {name}  ({exact})"

    # ---- state changes ----
    def set_test_cameras(self, cameras, snap: bool = False) -> None:
        """Replace the dropdown's camera list. ``snap`` parks on the first one
        instead of dropping to free-fly — the point-cloud preview needs it, since
        the fly camera starts at the origin and a scene reconstructed anywhere
        else would be off-screen with nothing to say the data had loaded."""
        self.test_cameras = list(cameras or [])
        self.snap_to(0 if snap and self.test_cameras else -1)

    def snap_to(self, index: int) -> None:
        """Park on test camera ``index`` (-1 => free-fly)."""
        if index < 0 or not self.test_cameras:
            self.index, self.exact = -1, None
            self._on_change()
            return
        index %= len(self.test_cameras)
        name, target = self.test_cameras[index]
        self.cam.set_from_view_matrix(target.view_matrix)  # WASD continues here
        fovy = _fovy_from_proj(target.proj_matrix)
        if fovy is not None:
            self.fovy = fovy
        # Render the test camera EXACTLY (view+proj) when its resolution matches
        # the render target, else fall back to the fly-cam perspective. Exact
        # proj removes the last raster-side difference vs eval (principal point,
        # exact fx/fy).
        self.index = index
        self.exact = (target if tuple(target.image_size) == self._render_size
                      else None)
        self._on_change()
        print(f"[Viewer] snapped to test cam {index + 1}/{len(self.test_cameras)} "
              f"({name}), exact_proj={self.exact is not None}, "
              f"fovy={self.fovy:.1f} deg")

    def step(self, delta: int) -> None:
        if not self.test_cameras:
            print("[Viewer] no test cameras available to snap to")
            return
        self.snap_to((self.index + delta) % len(self.test_cameras))

    def leave(self) -> None:
        """Manual navigation drops the parked state (keeps the current pose)."""
        if self.index != -1:
            self.snap_to(-1)

    def advance(self, held: set[str], dt: float) -> None:
        """Apply one frame of held-key motion (move + roll)."""
        self.cam.move(held, dt)
        if "r" in held:
            self.cam.roll_by(+ROLL_SPEED_DEG_S * dt)
        if "t" in held:
            self.cam.roll_by(-ROLL_SPEED_DEG_S * dt)
        if held & MOVE_TAGS:
            self.leave()

    def camera(self, width: int, height: int, aspect: float) -> Camera:
        if self.exact is not None:
            return self.exact
        return self.cam.get_camera(width, height, self.fovy, aspect=aspect)

    def current_name(self) -> str:
        if self.index < 0 or not self.test_cameras:
            return "(free-fly)"
        return self.test_cameras[self.index][0]


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
                 test_cameras: list[tuple[str, Camera]] | None = None,
                 state: Session | None = None):
        self.device = device
        self.width = width
        self.height = height
        self.max_splats = max_splats
        self.fovy = fovy
        self.hot_reload = hot_reload
        self.sh_degree = sh_degree
        # Session state shared with the orchestrator (phase, progress, log,
        # training stats, dataset commands). Always present, so every panel
        # binding has something to read even for a plain `train` run.
        self.state = state if state is not None else Session()
        # Hub mode adds the DATASET section (drag-drop + reconstruct/train).
        self._hub = self.state.mode == MODE_HUB
        # --- Scene-render budget WHILE TRAINING (see _scene_due) --------------
        # The panel keeps painting every frame; only the SCENE rasterize is
        # throttled, because that is the part that costs a training step's worth
        # of GPU (bin_count + radix sort + blend over the whole model).
        # Set == the model or the view changed, so the scene must be re-rendered
        # even if the budget would otherwise skip it. Written from the TRAINER
        # thread (update_gaussians), read/cleared on the render thread — an Event
        # for the same reason _suspend / _grab are. Initialized BEFORE the rig,
        # whose on_change callback writes it.
        self._scene_dirty = threading.Event()
        self._scene_dirty.set()
        self._last_scene_s = 0.0
        # The camera rig owns the fly camera and the test-camera snapping; the
        # panel binds to it directly, so the dropdown and the camera cannot
        # disagree and attach_scene needs no dirty-flag dance in the loop.
        self._rig = _CameraRig(fovy=fovy, render_size=(width, height),
                               test_cameras=test_cameras,
                               on_change=self.invalidate_scene)

        import contextlib
        self.gpu_lock = gpu_lock if gpu_lock is not None else contextlib.nullcontext()
        self.runtime = GpuRuntime(device, self.gpu_lock)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._start_event = threading.Event()
        # Set == training paused; the trainer blocks on wait_while_paused() each
        # iteration. Toggled by the viewer's Pause/Resume button.
        self._pause_event = threading.Event()
        # Set == the user asked to end the run early (Stop). The training loop's
        # continue condition polls it, so the run finalizes normally instead of
        # being torn down mid-iteration.
        self._stop_training = threading.Event()
        # Set == skip ALL GPU work in the render loop. The hub sets this while it
        # does heavy pipeline/target setup on the main thread, so those uploads
        # don't race the render thread on the shared queue.
        self._suspend = threading.Event()
        # Reflects the trainer's state so the panel can show "training" vs
        # "inspecting init". The trainer sets this True once it starts looping.
        self._training_active = False
        # Set == save the next rendered frame to a PNG (P key / programmatic).
        self._grab = threading.Event()

        # Serializes the main loop's paint against the one on_resize does from
        # inside the OS modal-resize loop — and against reconfigure_model, which
        # must not swap the pipeline out from under a frame.
        self._render_lock = threading.RLock()
        self._explicit_sort_pairs = max_sort_pairs
        self._mip_enabled = mip_enabled
        self._mip_kernel_size = mip_kernel_size
        self._pipeline = self._build_view_pipeline(
            max_splats, sh_degree, mip_enabled, mip_kernel_size, max_sort_pairs)

        self._has_data = False
        self._update_pending = False


    def _build_view_pipeline(self, max_splats: int, sh_degree: int,
                             mip_enabled: bool, mip_kernel_size: float,
                             max_sort_pairs: int | None) -> TrainingPipeline:
        """The viewer's OWN (render-only) pipeline, sized for the snapshots it will
        receive. Kept separate from ``__init__`` so :meth:`reconfigure_model` can
        rebuild it when the hub's TRAINING SETUP changes the model."""
        # NOTE can cut out splats
        if max_sort_pairs is None:
            max_sort_pairs = min(12_000_000, max(2_000_000, max_splats * 6))
        pipeline = TrainingPipeline(
            device=self.device, width=self.width, height=self.height,
            max_splats=max_splats,
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
            pipeline.set_active_sh_degree(sh_degree)
        return pipeline

    def reconfigure_model(self, *, max_splats: int, sh_degree: int,
                          mip_enabled: bool, mip_kernel_size: float) -> bool:
        """Rebuild the viewer's pipeline for a model the hub just reconfigured.

        ``max_splats`` sizes the snapshot buffers, ``sh_degree`` decides whether a
        ``features_rest`` buffer exists at all, and the Mip settings pick the raster
        path plus the ``filter_3D`` buffer — all fixed at pipeline construction. So
        when the panel's TRAINING SETUP changes any of them, the viewer's pipeline
        has to be rebuilt or the snapshot copy would target the wrong buffers.

        Call with rendering suspended (the hub already is, while it builds the run).
        Returns True if a rebuild happened. Cheap no-op when nothing changed.
        """
        if (max_splats == self.max_splats and sh_degree == self.sh_degree
                and mip_enabled == self._mip_enabled
                and mip_kernel_size == self._mip_kernel_size):
            return False
        with self._render_lock:      # no frame in flight while we swap pipelines
            with self.runtime.queue_scope():
                self.runtime.wait_for_idle()   # old buffers may still be in use
            self.max_splats = max_splats
            self.sh_degree = sh_degree
            self._mip_enabled = mip_enabled
            self._mip_kernel_size = mip_kernel_size
            self._pipeline = self._build_view_pipeline(
                max_splats, sh_degree, mip_enabled, mip_kernel_size,
                self._explicit_sort_pairs)
            # The new buffers are empty: don't rasterize until a snapshot lands.
            self._has_data = False
        print(f"[Viewer] model reconfigured: max_splats={max_splats:,} "
              f"sh_degree={sh_degree} mip={mip_enabled}")
        return True

    def live_splat_count(self, train_pipeline: TrainingPipeline) -> int:
        """The trainer's active splat count, read back to the host.

        ONE uint32. It costs a readback (a staging copy queued behind the work in
        flight, hence the queue scope), and it buys the snapshot below the right to
        copy the ACTIVE range instead of the whole allocation — which is sized for
        ``--max-splats`` and is mostly untouched tail for most of a run.
        """
        with self.runtime.queue_scope():
            counter = train_pipeline.buffers.num_gaussians_counter.to_numpy()
        return int(counter.view(np.uint32)[0])

    def update_gaussians(self, train_pipeline: TrainingPipeline):
        src, dst = train_pipeline.buffers, self._pipeline.buffers
        count = min(self.live_splat_count(train_pipeline), self.max_splats)
        batch = self.runtime.command_batch("viewer_update")
        encoder = batch.encoder
        _copy_splats(encoder, src.gaussians.storage, dst.gaussians.storage,
                     count, src.max_splats)
        if src.sh_enabled and dst.sh_enabled:
            _copy_splats(encoder, src.features_rest.storage,
                         dst.features_rest.storage, count, src.max_splats)
        # Mip-Splatting: copy the per-gaussian 3D smoothing filter so the viewer
        # applies the SAME low-pass eval does. filter_3D depends on gaussian
        # positions, so it must be refreshed with every snapshot (post-densify).
        if self._pipeline.raster_config.mip_enabled:
            src_f = getattr(src, "filter_3D", None)
            dst_f = getattr(dst, "filter_3D", None)
            if src_f is not None and dst_f is not None:
                _copy_splats(encoder, src_f.storage, dst_f.storage,
                             count, src.max_splats)
        # The viewer's count is what we COPIED, written INSIDE this batch rather
        # than copied from the trainer's counter: the readback above can be a
        # densify behind by the time we get here, and a count ahead of the copied
        # range would have the rasterizer walk uninitialized tail. Same batch means
        # the viewer's next frame sees the count and its data together — one queue,
        # so a submitted batch is atomic with respect to frames.
        encoder.upload_buffer_data(dst.num_gaussians_counter.storage, 0,
                                   np.array([count], dtype=np.uint32))
        self.runtime.submit(batch)
        # Mirror the trainer's CURRENT active SH degree so the viewer shows the
        # same bands as eval. The degree schedule ramps during training; pinning
        # the viewer at max over-shows half-trained higher bands eval (at the
        # scheduled degree) does not.
        self._pipeline.set_active_sh_degree(train_pipeline._active_sh_degree)
        self._has_data = count > 0
        # New model: the last scene image is stale, so the next frame must
        # re-rasterize even while the training budget is throttling (parked on a
        # test camera this is the ONLY thing that triggers a render).
        self.invalidate_scene()

    def preview_points(self, initial_points: dict) -> int:
        """Show a COLMAP-derived initial point cloud in the viewer's own pipeline.

        The hub calls this once a dataset is ready but before Train is pressed, so
        there is no training pipeline to snapshot from yet — the points go straight
        into the view pipeline's buffers instead. Returns how many were uploaded.

        SH is pinned to band 0 for the preview: an SfM cloud only carries DC
        color, and the higher bands in these buffers belong to whatever was shown
        before (a previous preview, or nothing at all).
        """
        means = np.asarray(initial_points["means"], dtype=np.float32)
        count = min(means.shape[0], self.max_splats)
        if count < means.shape[0]:
            print(f"[Viewer] preview clipped to {count:,} of {means.shape[0]:,} "
                  f"SfM points (viewer capacity)")
        with self._render_lock:      # no frame reading the buffers mid-upload
            with self.runtime.queue_scope():
                self._pipeline.set_gaussians(
                    means=means[:count],
                    log_scales=initial_points["log_scales"][:count],
                    rot_quats=initial_points["rot_quats"][:count],
                    opacities=initial_points["opacities"][:count],
                    color_logits=initial_points["color_logits"][:count],
                )
            self._pipeline.set_active_sh_degree(0)
            self._has_data = count > 0
        self.invalidate_scene()
        return count

    def clear_view(self) -> None:
        """Blank the display: no gaussians, no camera list (the hub's Clear).

        The buffers keep their allocation — ``_has_data`` is what gates
        rasterization — so the next preview or snapshot fills them straight back
        in. What this DOES drop is the last run's picture, which would otherwise
        sit on screen next to a panel that says there is no dataset.
        """
        with self._render_lock:      # no frame mid-read of the buffers
            self._has_data = False
        self._rig.set_test_cameras([])
        self._training_active = False
        print("[Viewer] view cleared")

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

    def invalidate_scene(self) -> None:
        """Mark the rendered scene stale, so the next frame re-rasterizes it.

        Anything that changes WHAT should be on screen — new gaussians, a camera
        snap, a resize, a pipeline swap — has to call this, because while training
        the render loop otherwise assumes the last image is still good.
        """
        self._scene_dirty.set()

    def _scene_due(self, now: float) -> bool:
        """Whether to re-rasterize the scene this frame.

        The viewer's frame does the same work as a training step, and unthrottled
        it takes roughly 60 of them per second — which is why attaching the viewer
        cost ~4x throughput early in a run, when a step is cheap and the model is
        small. So while the trainer is competing for the queue, the scene is
        rendered on a budget:

          * PARKED ON A TEST CAMERA — the view cannot move (any navigation calls
            ``_CameraRig.leave``), so there is nothing to animate: render only
            when new gaussians actually arrive. This is the mode for watching a
            held-out view converge, and it costs one render per snapshot.
          * FREE-FLY — the camera can move at any moment, so it must animate;
            capped at :data:`TRAINING_SCENE_FPS`, enough to fly by and far below
            the ~60 that starves the trainer.
          * NOT TRAINING — nothing to share the GPU with, so no budget at all.

        The UI pass is NOT gated by this: the panel, log and progress bar keep
        painting every frame, so the window stays responsive at 60Hz regardless.
        """
        if not self._training_active:
            return True
        if self._scene_dirty.is_set():
            return True
        if self._rig.index >= 0:            # parked: only new data moves the image
            return False
        return (now - self._last_scene_s) >= (1.0 / TRAINING_SCENE_FPS)

    def set_training_active(self, active: bool = True) -> None:
        """The trainer tells the viewer whether it is inside the loop.

        Only the Pause button needs this (there is nothing to pause otherwise);
        the run's STATE is published by the orchestrator, which is the only place
        that knows the difference between initializing, running and finalizing.
        """
        self._training_active = active
        if active:
            self.state.training_running()

    def attach_scene(self, test_cameras, snap: bool = False) -> None:
        """Populate the camera dropdown from a scene chosen at runtime (hub flow).
        Thread-safe hand-off: the combo's items/value are bound to the rig, so the
        next frame picks the new list up on its own. ``snap`` parks on the first
        camera (see ``_CameraRig.set_test_cameras``).
        """
        self._rig.set_test_cameras(test_cameras, snap=snap)

    def suspend_rendering(self) -> None:
        """Pause all GPU work in the render loop (hub uses this during setup).

        Also frees the GIL: a painting frame holds it through the submit wait and
        every panel binding, which starves CPU-heavy work on other threads (see
        ``ViewerSession.suspend_rendering``)."""
        self._suspend.set()

    def resume_rendering(self) -> None:
        self._suspend.clear()

    def is_paused(self) -> bool:
        """True while the Pause button has training suspended."""
        return self._pause_event.is_set()

    def pause_training(self) -> None:
        """Ask the trainer to pause, and say so NOW.

        The trainer notices at its next iteration boundary, so the button
        publishes the INTENT ("Pausing") and the trainer publishes the fact (see
        :meth:`wait_while_paused`). Without the first half the panel would look
        unchanged for up to an iteration.
        """
        if self._training_active and not self._pause_event.is_set():
            self._pause_event.set()
            self.state.begin_activity("Pausing")

    def resume_training(self) -> None:
        if self._pause_event.is_set():
            self._pause_event.clear()
            self.state.begin_activity("Resuming")

    def toggle_pause(self) -> None:
        """Flip pause/resume (keyboard shortcut; the panel has both buttons)."""
        (self.resume_training if self._pause_event.is_set()
         else self.pause_training)()

    def request_stop_training(self) -> None:
        """Stop the run EARLY, keeping what it has optimized so far.

        Not a cancel: the loop breaks at the next iteration boundary and the run
        finalizes normally, so the partially trained result is still a result the
        user can inspect and export. Clears any pause, or a paused run would never
        reach the check.
        """
        self._stop_training.set()
        self._pause_event.clear()
        self.state.training_stopping()

    def stop_requested(self) -> bool:
        """Polled by the training loop's continue condition."""
        return self._stop_training.is_set()

    def clear_stop_request(self) -> None:
        """Before a new run: a stop belongs to the run that was stopped."""
        self._stop_training.clear()

    def wait_while_paused(self, poll: float = 0.05) -> None:
        """Block the caller (the trainer) while paused, releasing if the viewer
        window closes so a paused+closed viewer can't hang training forever.

        Publishes PAUSED/RUNNING from HERE — the trainer thread is the only place
        that knows when the pause actually took effect.
        """
        if not self._pause_event.is_set():
            return
        self.state.training_paused()
        while self._pause_event.is_set():
            if self._stop_event.is_set() or (self._thread and not self._thread.is_alive()):
                break
            time.sleep(poll)
        if not self._stop_training.is_set():
            self.state.training_running()

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

    # ---- UI composition (nothing below this lives in the hot loop) --------

    def _set_up_ref(self) -> None:
        self._rig.cam.set_ref_up_to_current()
        print("[Viewer] up reference set to current view")

    def _save_frame(self) -> None:
        """Write the last rendered frame to a PNG for offline diffing against
        the eval output (outputs/<run>/cams/cam_XX/...)."""
        try:
            from PIL import Image
            img = self._pipeline.read_output_image()[..., :3]
            png = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            name = self._rig.current_name()
            label = ("testcam_" + name.replace("/", "_").replace("\\", "_")
                     if self._rig.index >= 0 else "freefly")
            path = os.path.abspath(f"viewer_grab_{label}.png")
            Image.fromarray(png, mode="RGB").save(path)
            print(f"[Viewer] saved frame for {name} -> {path}")
        except Exception as e:  # noqa: BLE001 - a failed grab must not stop the loop
            print(f"[Viewer] frame grab failed: {e}")

    def _build_input(self, mouse: dict) -> InputMap:
        """Key bindings AND their help text. Adding a control is one line here;
        ``describe()`` feeds the panel's CONTROLS section, so the on-screen help
        can never drift from the real bindings."""
        keys = InputMap().hold_many([
            (spy.KeyCode.w, "w"), (spy.KeyCode.a, "a"), (spy.KeyCode.s, "s"),
            (spy.KeyCode.d, "d"), (spy.KeyCode.q, "q"), (spy.KeyCode.e, "e"),
            (spy.KeyCode.r, "r"), (spy.KeyCode.t, "t"),  # R/T roll
        ])
        keys.press(spy.KeyCode.c, lambda: self._rig.step(+1))
        keys.press(spy.KeyCode.v, lambda: self._rig.step(-1))
        keys.press(spy.KeyCode.p, self._grab.set)
        keys.press(spy.KeyCode.y, self._set_up_ref)
        keys.press(spy.KeyCode.escape, lambda: mouse.update(captured=False))
        return (keys
                .document("WASD / QE", "move")
                .document("drag", "look around")
                .document("wheel", "move speed")
                .document("R / T", "roll view")
                .document("Y", "level horizon to view")
                .document("C / V", "next / prev test camera")
                .document("P", "save frame to PNG")
                .document("F1", "show / hide panel"))

    def _panel_actions(self) -> Actions:
        """Bind every panel action to a one-shot request on the session.

        Nothing here does work: a click posts intent and returns, so a slow
        operation can never block the render thread. The orchestrator picks the
        request up and drives the state machine (see ``TrainingApplication``).
        Pause/resume/stop are the exception — they are the viewer's own events, so
        they publish immediately, which is what makes them feel instant.
        """
        return Actions(
            browse=None,          # no file dialog in this windowing layer
            install_tools=lambda: self.state.request("install_tools"),
            reconstruct=lambda: self.state.request("reconstruct"),
            select_matcher=self.state.set_matcher_index,
            set_recon_cuda=self.state.set_recon_cuda,
            new_dataset=lambda: self.state.request("new_dataset"),
            clear_session=lambda: self.state.request("clear"),
            start_training=lambda: self.state.request("train"),
            restart_training=lambda: self.state.request("restart"),
            pause=self.pause_training,
            resume=self.resume_training,
            stop=self.request_stop_training,
            export=lambda: self.state.request("export"),
            set_export_name=self.state.set_export_name,
            reveal_export=lambda: self.state.request("reveal"),
            confirm=lambda: self.state.request("confirm"),
            confirm_alt=lambda: self.state.request("confirm_alt"),
            cancel=lambda: self.state.request("cancel"),
            on_config_change=self.state.mark_config_edited,
            toggle_log=self.state.toggle_log,
            clear_log=self.state.clear_log,
        )

    def _build_panel(self, view, keys: InputMap) -> Panel:
        """Compose the panel in LIFECYCLE order: state, dataset, training, result,
        configuration, viewer, controls, diagnostics.

        Every widget PULLS its value from a getter each frame, so a state
        transition changes the UI without rebuilding anything, and extending it
        means adding a line to a section in :mod:`.ui.sections`.
        """
        panel = Panel(self.device, title="Scatto-GS trainer", anchor="left",
                      thickness=430.0, min_thickness=380.0)
        actions = self._panel_actions()
        header(panel, view)
        if self._hub:
            # Outside the hub the dataset and the run are already chosen by the
            # time the window exists, so these would all be read-only.
            dataset_section(panel, view, actions)
        training_section(panel, view, actions)
        # Viewer-only by construction: the curve exists to be watched while a run
        # goes, which is the one thing a headless run cannot do.
        loss_section(panel, view)
        if self._hub:
            result_section(panel, view, actions)
            configuration_section(
                panel, view, self.state.setup, actions,
                show_all=lambda: view.show_all_config,
                toggle_all=self.state.toggle_config_detail)
        viewer_section(panel, CameraControls(
            names=self._rig.names,
            selection=self._rig.selection,
            select=lambda index: self._rig.snap_to(int(index) - 1),
            label=self._rig.label,
            speed=lambda: self._rig.cam.speed,
            fovy=lambda: self._rig.fovy,
        ))
        controls_section(panel, keys.describe())
        diagnostics_section(panel, view, actions, show=lambda: view.show_log)
        return panel

    def _run_inner(self):
        window = spy.Window(width=self.width, height=self.height, title="Scatto-GS")

        surface = self.device.create_surface(window)
        # Present in a NON-sRGB format: output_image already holds sRGB-encoded
        # color (renderer composites in sRGB, like the stored PNGs), so the blit
        # must not re-apply the sRGB OETF. A default/sRGB surface double-encodes
        # -> washed-out viewer while the PNGs look correct.
        present_format = _linear_present_format(surface)
        print(f"[Viewer] surface preferred={surface.info.preferred_format} "
              f"-> presenting as {present_format}")
        surface.configure(width=self.width, height=self.height, format=present_format)
        # The configured surface size is the SINGLE source of truth for a frame
        # (drives surface config, camera aspect, ui.layout, ui.begin, and the blit
        # target) — mixing this with the live window size mis-projected the panel.
        surf_w, surf_h = self.width, self.height
        render_lock = self._render_lock   # main-loop vs on_resize vs reconfigure

        # One consistent state snapshot per frame feeds EVERY binding (see
        # StateView): no two widgets can show state from different instants.
        view = self.state.view()
        mouse = {"captured": False, "x": 0.0, "y": 0.0}
        keys = self._build_input(mouse)
        ui = self._build_panel(view, keys)
        # ImGui puts a close box on the panel's title bar; F1 brings it back (and
        # gives an unobstructed look at the scene on demand).
        keys.press(spy.KeyCode.f1, ui.toggle)
        print(f"[Viewer] panel: {ui.dock.thickness_px:.0f}px wide "
              f"(display scale {ui.dock.scale:.2f}x)")

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
                self._rig.cam.adjust_speed(float(event.scroll.y))
                return
            if event.is_button_down():
                mouse.update(captured=True, x=event.pos[0], y=event.pos[1])
            elif event.is_button_up():
                mouse["captured"] = False
            elif event.is_move() and mouse["captured"]:
                dx = event.pos[0] - mouse["x"]
                dy = event.pos[1] - mouse["y"]
                mouse["x"], mouse["y"] = event.pos[0], event.pos[1]
                self._rig.cam.rotate(dx, dy)
                self._rig.leave()

        def on_drop(paths) -> None:
            if paths:
                self.state.drop(str(paths[0]))

        def _reconfigure(w, h):
            """Match the swapchain to the window. No window.resize / aspect-snap
            (that fought the drag). configure() rebuilds the swapchain; GPU must
            be idle first. Early-outs when the size is unchanged."""
            nonlocal surf_w, surf_h
            if w <= 0 or h <= 0 or (w, h) == (surf_w, surf_h):
                return
            with self.runtime.queue_scope():
                self.runtime.wait_for_idle()
                surface.configure(width=w, height=h, format=present_format)
            surf_w, surf_h = w, h
            # The camera aspect changed, so the last scene image no longer matches
            # the window — without this, resizing while parked on a test camera
            # would blit a stale frame into the new shape for as long as the
            # snapshot cadence takes.
            self.invalidate_scene()

        def render_frame():
            """Paint one frame. Uses surf_w/surf_h for EVERYTHING (ui.layout,
            ui.begin's ImGui DisplaySize, the camera aspect, the blit target) so
            the docked panel can never be mis-projected for a frame. The UI always
            draws; the scene is rasterized+blitted only once gaussians exist."""
            with render_lock:
                if self._suspend.is_set():   # hub setup owns the GPU queue
                    return
                w, h = surf_w, surf_h
                # layout THEN refresh: bindings wrap their text to the panel's
                # current content width, so a panel resize re-flows text live.
                ui.layout(w, h)
                ui.refresh()
                surface_tex = surface.acquire_next_image()
                if surface_tex is None:      # can happen transiently on resize
                    return
                batch = self.runtime.command_batch("viewer_frame")
                enc = batch.encoder
                if self._has_data:
                    # Re-rasterize only when the budget allows (see _scene_due);
                    # the blit happens either way, so a skipped frame re-presents
                    # the last scene image while the UI pass below stays live.
                    now = time.perf_counter()
                    if self._scene_due(now):
                        camera = self._rig.camera(self.width, self.height, w / h)
                        self._pipeline._encoder = enc
                        try:
                            self._pipeline.rasterize(camera)
                        finally:
                            self._pipeline._encoder = None
                        self._scene_dirty.clear()
                        self._last_scene_s = now
                    enc.blit(surface_tex, self._pipeline.buffers.output_image)
                else:
                    # Nothing else writes the target before the UI pass, and that
                    # pass must LOAD (else it would erase the scene blit above),
                    # so an un-cleared swapchain image shows whatever was in it
                    # LAST time it came round — which smeared several stale copies
                    # of the panel's edge across the hub window during a resize.
                    enc.clear_texture_float(surface_tex, clear_value=BACKDROP)
                ui.begin(w, h, window)
                ui.end(surface_tex, enc)
                submit = self.runtime.submit(batch, after_submit=surface.present)
                self.runtime.wait_for_submit(submit)
                if self._grab.is_set():
                    self._grab.clear()
                    self._save_frame()

        def on_resize(w, h):
            """Render LIVE during a resize. sgl fires on_resize on THIS render
            thread from inside window.process_events()'s Win32 modal-resize loop,
            so painting here is the only way to track the drag — the old debounce
            skipped rendering and let the OS stretch a stale frame (the glitch)."""
            if self._suspend.is_set():
                return
            try:
                _reconfigure(w, h)
                render_frame()
            except Exception as e:
                if not getattr(self, "_err_printed", False):
                    print(f"[Viewer] on_resize render failed: {e}")
                    self._err_printed = True

        window.on_keyboard_event = on_key
        window.on_mouse_event = on_mouse
        window.on_resize = on_resize
        if self._hub:
            window.on_drop_files = on_drop  # drag a dataset folder onto the window

        watcher = _ShaderWatcher(
            self.device, self._pipeline, self.runtime) if self.hot_reload else None
        last_time = time.perf_counter()

        while not window.should_close() and not self._stop_event.is_set():
            window.process_events()
            if self._suspend.is_set():  # hub setup owns the GPU queue right now
                time.sleep(0.02)
                continue
            if watcher:
                watcher.poll()

            # Keep the surface matched to the window (catches a resize that
            # arrived while suspended or that on_resize didn't service). Cheap —
            # early-outs when unchanged.
            _reconfigure(window.width, window.height)

            # --- per-frame STATE, once per loop iteration (NOT inside
            #     render_frame, so a paint triggered from on_resize can't advance
            #     the camera or re-sample state mid-drag). ---
            now = time.perf_counter()
            dt = min(now - last_time, 0.1)
            last_time = now
            self._rig.advance(keys.held, dt)
            view.sample()   # the frame's single, consistent state snapshot

            try:
                render_frame()
            except Exception as e:
                if not getattr(self, "_err_printed", False):
                    print(f"[Viewer] frame error: {e}")
                    self._err_printed = True
                time.sleep(0.016)

        try:
            window.close()
        except Exception:
            pass