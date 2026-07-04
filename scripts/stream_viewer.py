"""Standalone live viewer — renders gaussian snapshots streamed from a
running trainer over shared memory.

Fully independent of the trainer: its own process, its own GPU device, its
own window on its own main thread. It never touches the trainer's device or
buffers, so it cannot stall or crash the training run (and can be closed and
reopened freely).

Usage:
    # The trainer prints the stream name at startup, e.g.
    #   [SplatStream] serving on 'psm_ab12cd34' (capacity 2000000)
    .venv/Scripts/python.exe scripts/stream_viewer.py --stream psm_ab12cd34
    # or auto-discover the newest stream (reads outputs/.splat_stream):
    .venv/Scripts/python.exe scripts/stream_viewer.py
    # pass --scene to enable snapping to dataset camera poses:
    .venv/Scripts/python.exe scripts/stream_viewer.py --scene datasets/360_v2/bicycle

Controls:
    WASD          move   (Q/E = down/up)
    Shift         move faster (boost)
    Ctrl          move slower (precision)
    Mouse wheel   adjust base move speed
    Left-drag     look around
    UI panel      camera dropdown to snap to a dataset pose; speed slider
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import slangpy as spy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trainer.dataset import Camera
from trainer.device import create_device
from trainer.pipeline import Pipeline
from trainer.splat_stream import SplatStreamReader, STREAM_LOCK_FILE_DEFAULT
from trainer.viewer import FlyCamera, _build_proj_matrix

STREAM_LOCK_FILE = Path(STREAM_LOCK_FILE_DEFAULT)

SHIFT_BOOST = 8.0     # multiplier while Shift held
CTRL_SLOW = 0.15      # multiplier while Ctrl held
WHEEL_STEP = 1.25     # base-speed factor per wheel notch


def main() -> None:
    ap = argparse.ArgumentParser(description="Live viewer for a trainer splat stream")
    ap.add_argument("--stream", type=str, default=None,
                    help="shared-memory stream name (printed by the trainer). "
                         "If omitted, reads outputs/.splat_stream.")
    ap.add_argument("--scene", type=str, default=None,
                    help="dataset dir (e.g. datasets/360_v2/bicycle). If given, "
                         "loads the camera poses so you can snap to them from the UI.")
    ap.add_argument("--downsample", type=int, default=4,
                    help="downsample factor used when loading --scene poses")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--max-splats", type=int, default=None,
                    help="viewer buffer capacity; defaults to the stream capacity")
    ap.add_argument("--backend", type=str, default="vulkan", choices=["vulkan", "d3d12"])
    ap.add_argument("--fovy", type=float, default=45.0)
    args = ap.parse_args()

    stream_name = args.stream
    if stream_name is None:
        if not STREAM_LOCK_FILE.exists():
            print(f"no --stream given and {STREAM_LOCK_FILE} not found; "
                  f"is the trainer running with streaming enabled?")
            return
        stream_name = STREAM_LOCK_FILE.read_text().strip()

    # Attach to the trainer's stream, retrying in case the viewer launched a
    # moment before the trainer created the block.
    reader = None
    for _ in range(40):
        try:
            reader = SplatStreamReader(stream_name)
            break
        except (FileNotFoundError, ValueError):
            time.sleep(0.25)
    if reader is None:
        print(f"could not attach to stream {stream_name!r} (trainer not up?)")
        return
    print(f"attached to stream {stream_name!r} (capacity {reader.capacity})")

    # Optional: dataset camera poses for the snap-to-camera UI.
    cam_names: list[str] = []
    cam_views: list[np.ndarray] = []
    if args.scene:
        try:
            from trainer.colmap import load_colmap
            scene = load_colmap(args.scene, downsample=args.downsample)
            for i, e in enumerate(scene.images):
                cam_names.append(f"camera {i}: " + e.image_name)
                cam_views.append(np.asarray(e.camera.view_matrix, dtype=np.float64))
            print(f"loaded {len(cam_names)} camera poses from {args.scene}")
        except Exception as e:
            print(f"could not load --scene poses ({e}); camera snap disabled")

    cap = args.max_splats or reader.capacity
    device = create_device(backend=args.backend)
    pipeline = Pipeline(device=device, width=args.width, height=args.height,
                        max_splats=cap)
    pipeline.lambda_ssim = 0.0
    pipeline.w_alpha = 0.0

    window = spy.Window(width=args.width, height=args.height,
                        title=f"3DGS Live — {stream_name}")
    surface = device.create_surface(window)
    surface.configure(width=args.width, height=args.height)

    # --- ImGui UI ---
    ui = spy.ui.Context(device)
    ui_panel = spy.ui.Window(ui.screen, "Viewer", position=spy.float2(10, 10),
                             size=spy.float2(260, 0))
    spy.ui.Text(ui_panel, "WASD move | Shift boost | Ctrl slow")
    spy.ui.Text(ui_panel, "wheel: speed | drag: look")
    status_text = spy.ui.Text(ui_panel, "waiting for data...")

    cam = FlyCamera()
    # Base move speed lives on the fly-cam; the slider edits it directly.
    speed_prop = spy.ui.SliderFloat(ui_panel, "speed", value=cam.speed,
                                    min=0.01, max=50.0)

    # When a dataset camera is selected we render its EXACT view matrix (roll
    # included) rather than the fly-cam's yaw/pitch approximation, so the
    # snapped view aligns pixel-for-pixel with GT. Any WASD/look clears it and
    # hands control back to the fly-cam (seeded from that pose).
    snapped_view = [None]  # holds the exact GT world->view matrix, or None

    if cam_names:
        # ListBox of dataset cameras; selecting one snaps to its exact pose.
        cam_box = spy.ui.ListBox(ui_panel, "camera", items=["(free fly)"] + cam_names,
                                 height_in_items=min(12, len(cam_names) + 1))

        def on_cam_select(selected: int):
            idx = selected - 1  # 0 == "(free fly)"
            if 0 <= idx < len(cam_views):
                # Seed the fly-cam (so leaving the snap continues from here)
                # AND record the exact view matrix for pixel-accurate display.
                cam.set_from_view_matrix(cam_views[idx])
                snapped_view[0] = cam_views[idx]
            else:
                snapped_view[0] = None  # "(free fly)"
        cam_box.callback = on_cam_select
    else:
        cam_box = None

    keys_held: set = set()
    mouse_captured = False
    last_mouse = [0.0, 0.0]
    key_map = {spy.KeyCode.w: "w", spy.KeyCode.a: "a", spy.KeyCode.s: "s",
               spy.KeyCode.d: "d", spy.KeyCode.q: "q", spy.KeyCode.e: "e"}
    mods = {"shift": False, "ctrl": False}

    def on_key(event):
        nonlocal mouse_captured
        if ui.handle_keyboard_event(event):
            return  # UI consumed it (e.g. typing in a widget)
        mods["shift"] = event.has_modifier(spy.KeyModifier.shift)
        mods["ctrl"] = event.has_modifier(spy.KeyModifier.ctrl)
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
        nonlocal mouse_captured
        if ui.handle_mouse_event(event):
            # Don't also drive the camera when interacting with the panel.
            mouse_captured = False
            return
        if event.is_scroll():
            # Wheel adjusts base speed multiplicatively (feels linear in log).
            factor = WHEEL_STEP ** float(event.scroll[1])
            cam.speed = float(np.clip(cam.speed * factor, 0.01, 50.0))
            speed_prop.value = cam.speed
        elif event.is_button_down():
            mouse_captured = True
            last_mouse[0], last_mouse[1] = event.pos[0], event.pos[1]
        elif event.is_button_up():
            mouse_captured = False
        elif event.is_move() and mouse_captured:
            dx = event.pos[0] - last_mouse[0]
            dy = event.pos[1] - last_mouse[1]
            last_mouse[0], last_mouse[1] = event.pos[0], event.pos[1]
            if dx or dy:
                snapped_view[0] = None  # looking around leaves the snapped pose
            cam.rotate(dx, dy)

    needs_reconfigure = False

    def on_resize(w, h):
        nonlocal needs_reconfigure
        needs_reconfigure = True

    window.on_keyboard_event = on_key
    window.on_mouse_event = on_mouse
    window.on_resize = on_resize

    have_data = False
    last_iter = -1
    last_count = 0
    last_time = time.perf_counter()
    poll_accum = 0.0
    configured = True
    err_printed = False

    while not window.should_close():
        window.process_events()
        if reader.is_closed():
            print("trainer closed the stream; exiting")
            break

        if needs_reconfigure:
            needs_reconfigure = False
            configured = False
            new_w, new_h = window.width, window.height
            if new_w > 0 and new_h > 0:
                try:
                    device.wait_for_idle()
                    surface.unconfigure()
                    surface.configure(width=new_w, height=new_h)
                    configured = True
                except Exception as e:
                    print(f"[viewer] reconfigure failed: {e}")
            continue
        if not configured:
            time.sleep(0.016)
            continue

        now = time.perf_counter()
        dt = min(now - last_time, 0.1)
        last_time = now

        # Speed slider can be dragged; keep the fly-cam in sync both ways.
        if abs(speed_prop.value - cam.speed) > 1e-6:
            cam.speed = float(speed_prop.value)
        boost = SHIFT_BOOST if mods["shift"] else (CTRL_SLOW if mods["ctrl"] else 1.0)
        base_speed = cam.speed
        cam.speed = base_speed * boost
        if keys_held:
            snapped_view[0] = None  # moving leaves the snapped pose
        cam.move(keys_held, dt)
        cam.speed = base_speed  # restore un-boosted base for the slider

        poll_accum += dt
        if poll_accum >= 0.1:
            poll_accum = 0.0
            snap = reader.latest()
            if snap is not None:
                floats, count, iteration = snap
                pipeline.set_gaussians_raw(floats, count)
                have_data = True
                last_iter, last_count = iteration, count

        status_text.text = (f"iter {last_iter}  |  {last_count} splats  |  spd {cam.speed:.2f}"
                            if have_data else "waiting for data...")

        if not have_data:
            time.sleep(0.02)
            continue

        # Render at the pipeline's FIXED size (the final blit rescales into
        # the resized swapchain). Using the live window size here would
        # mismatch the pipeline's output_image resolution and fail the
        # camera-resolution check in _run_forward. Take the projection aspect
        # from the WINDOW so geometry stays undistorted after the blit stretch.
        win_aspect = window.width / max(window.height, 1)
        if snapped_view[0] is not None:
            # Use the dataset camera's EXACT view matrix (preserves roll) so the
            # snapped view matches GT; only the projection is rebuilt at the
            # viewer's render resolution/aspect.
            proj = _build_proj_matrix(args.fovy, win_aspect)
            camera = Camera(view_matrix=snapped_view[0].astype(np.float32),
                            proj_matrix=proj,
                            image_size=(pipeline.width, pipeline.height))
        else:
            camera = cam.get_camera(pipeline.width, pipeline.height, args.fovy,
                                    aspect=win_aspect)
        try:
            pipeline._run_forward(camera)
            surface_tex = surface.acquire_next_image()
            if surface_tex is None:
                needs_reconfigure = True
                continue
            encoder = device.create_command_encoder()
            encoder.blit(surface_tex, pipeline.buffers.output_image)
            # Draw the UI on top, into the same swapchain image + encoder.
            ui.begin_frame(window.width, window.height)
            ui.end_frame(surface_tex, encoder)
            device.submit_command_buffer(encoder.finish())
            surface.present()
            err_printed = False
        except Exception as e:
            if not err_printed:
                print(f"[viewer] frame error (will reconfigure): {e}")
                err_printed = True
            needs_reconfigure = True
            time.sleep(0.02)

    reader.close()
    try:
        window.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
