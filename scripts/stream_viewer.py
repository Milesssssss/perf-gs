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
    # or auto-discover the newest stream the trainer wrote to a lock file:
    .venv/Scripts/python.exe scripts/stream_viewer.py

Controls: WASD move, QE up/down, hold left-mouse + drag to look.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import slangpy as spy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trainer.device import create_device
from trainer.pipeline import Pipeline
from trainer.splat_stream import SplatStreamReader, STREAM_LOCK_FILE_DEFAULT
from trainer.viewer import FlyCamera

# Written by the trainer so the viewer can auto-discover the active stream.
STREAM_LOCK_FILE = Path(STREAM_LOCK_FILE_DEFAULT)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live viewer for a trainer splat stream")
    ap.add_argument("--stream", type=str, default=None,
                    help="shared-memory stream name (printed by the trainer). "
                         "If omitted, reads outputs/.splat_stream.")
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

    # Attach to the trainer's stream, retrying briefly in case the viewer
    # launched a moment before the trainer created the block.
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

    cam = FlyCamera()
    keys_held: set = set()
    mouse_captured = False
    last_mouse = [0.0, 0.0]
    key_map = {spy.KeyCode.w: "w", spy.KeyCode.a: "a", spy.KeyCode.s: "s",
               spy.KeyCode.d: "d", spy.KeyCode.q: "q", spy.KeyCode.e: "e"}

    def on_key(event):
        nonlocal mouse_captured
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
        if event.is_button_down():
            mouse_captured = True
            last_mouse[0], last_mouse[1] = event.pos[0], event.pos[1]
        elif event.is_button_up():
            mouse_captured = False
        elif event.is_move() and mouse_captured:
            dx = event.pos[0] - last_mouse[0]
            dy = event.pos[1] - last_mouse[1]
            last_mouse[0], last_mouse[1] = event.pos[0], event.pos[1]
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
    last_time = time.perf_counter()
    poll_accum = 0.0
    configured = True

    while not window.should_close():
        window.process_events()
        if reader.is_closed():
            print("trainer closed the stream; exiting")
            break

        # A window resize invalidates the swapchain; the surface must be
        # reconfigured to the new size before acquire_next_image will succeed
        # again (otherwise it fails with SLANG_FAIL every frame). The render
        # pipeline stays at its fixed size — the final blit rescales into the
        # resized swapchain image.
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
        cam.move(keys_held, dt)

        # Poll the stream a few times a second (not every frame — the trainer
        # only pushes every N iters, and uploading is the only real cost).
        poll_accum += dt
        if poll_accum >= 0.1:
            poll_accum = 0.0
            snap = reader.latest()
            if snap is not None:
                floats, count, iteration = snap
                pipeline.set_gaussians_raw(floats, count)
                have_data = True
                if iteration != last_iter:
                    last_iter = iteration
                    window.title = f"3DGS Live — iter {iteration}, {count} splats"

        if not have_data:
            time.sleep(0.02)
            continue

        camera = cam.get_camera(args.width, args.height, args.fovy)
        try:
            pipeline._run_forward(camera)
            surface_tex = surface.acquire_next_image()
            if surface_tex is None:
                # Swapchain not ready (e.g. minimized or mid-resize) — retry.
                needs_reconfigure = True
                continue
            encoder = device.create_command_encoder()
            encoder.blit(surface_tex, pipeline.buffers.output_image)
            device.submit_command_buffer(encoder.finish())
            surface.present()
            main._err_printed = False  # healthy frame — re-arm error logging
        except Exception as e:
            # Most frame errors here are a stale swapchain after a resize;
            # flag a reconfigure so the next loop heals it instead of failing
            # every frame. Throttle the message so it doesn't spam.
            if not getattr(main, "_err_printed", False):
                print(f"[viewer] frame error (will reconfigure): {e}")
                main._err_printed = True
            needs_reconfigure = True
            time.sleep(0.02)

    reader.close()
    try:
        window.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
