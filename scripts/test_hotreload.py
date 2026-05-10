"""Minimal test: does SlangPy hot-reload work with a window loop?

Renders a simple compute kernel to a texture, blits to surface, and
prints diagnostics on each frame. Save the test shader to trigger reload.

Run: .venv/Scripts/python.exe scripts/test_hotreload.py
Then edit shaders/test_hotreload.slang (change the color) and save.
Watch stdout for what happens.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import slangpy as spy

SHADERS_DIR = Path(__file__).resolve().parents[1] / "shaders"
TEST_SHADER = SHADERS_DIR / "test_hotreload.slang"

# Write a trivial shader we can edit.
TEST_SHADER.write_text("""\
[shader("compute")]
[numthreads(16, 16, 1)]
void fillColor(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,
    uniform uint2 frame_dim)
{
    if (dtid.x >= frame_dim.x || dtid.y >= frame_dim.y) return;
    output_image[dtid.xy] = float4(1.0, 0.0, 0.0, 1.0); // RED
}
""")

WIDTH, HEIGHT = 320, 240

device = spy.create_device(type=spy.DeviceType.vulkan, include_paths=[SHADERS_DIR])

# Create kernel
program = device.load_program(str(TEST_SHADER), ["fillColor"])
kernel = device.create_compute_kernel(program)

# Create output texture
output_tex = device.create_texture(
    type=spy.TextureType.texture_2d,
    format=spy.Format.rgba32_float,
    width=WIDTH, height=HEIGHT,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    label="test_output",
)

# Window + surface
window = spy.Window(width=WIDTH, height=HEIGHT, title="Hot Reload Test")
surface = device.create_surface(window)
surface.configure(width=WIDTH, height=HEIGHT)

print("Running. Edit shaders/test_hotreload.slang and save to test hot-reload.")
print("Close window to exit.")

frame = 0
while not window.should_close():
    window.process_events()

    try:
        kernel.dispatch(
            thread_count=[WIDTH, HEIGHT, 1],
            output_image=output_tex,
            frame_dim=spy.uint2(WIDTH, HEIGHT),
        )

        surface_tex = surface.acquire_next_image()
        encoder = device.create_command_encoder()
        encoder.blit(surface_tex, output_tex)
        cmd = encoder.finish()
        device.submit_command_buffer(cmd)
        surface.present()
    except Exception as e:
        print(f"frame {frame}: EXCEPTION: {e}")
        time.sleep(0.1)

    frame += 1
    if frame % 60 == 0:
        print(f"frame {frame}: alive")

window.close()
print("done")
