"""Test: use device.reload_all_programs() for hot-reload."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import slangpy as spy

SHADERS_DIR = Path(__file__).resolve().parents[1] / "shaders"
TEST_SHADER = SHADERS_DIR / "test_hotreload.slang"

WIDTH, HEIGHT = 4, 4

TEST_SHADER.write_text("""\
[shader("compute")]
[numthreads(4, 4, 1)]
void fillColor(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,
    uniform uint2 frame_dim)
{
    if (dtid.x >= frame_dim.x || dtid.y >= frame_dim.y) return;
    output_image[dtid.xy] = float4(1.0, 0.0, 0.0, 1.0);
}
""")

device = spy.create_device(type=spy.DeviceType.vulkan, include_paths=[SHADERS_DIR])

# Print docs
print("hot_reload_check doc:", device.hot_reload_check.__doc__)
print("reload_all_programs doc:", device.reload_all_programs.__doc__)
print("set_hot_reload_delay doc:", device.set_hot_reload_delay.__doc__)
print("register_shader_hot_reload_callback doc:", device.register_shader_hot_reload_callback.__doc__)
print()

program = device.load_program(str(TEST_SHADER), ["fillColor"])
kernel = device.create_compute_kernel(program)

output_tex = device.create_texture(
    type=spy.TextureType.texture_2d,
    format=spy.Format.rgba32_float,
    width=WIDTH, height=HEIGHT,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
)


def dispatch_and_read():
    kernel.dispatch(
        thread_count=[WIDTH, HEIGHT, 1],
        output_image=output_tex,
        frame_dim=spy.uint2(WIDTH, HEIGHT),
    )
    device.wait_for_idle()
    return output_tex.to_numpy()[0, 0]


pixel = dispatch_and_read()
print(f"1. Initial: {pixel}")

# Edit shader to GREEN
TEST_SHADER.write_text("""\
[shader("compute")]
[numthreads(4, 4, 1)]
void fillColor(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,
    uniform uint2 frame_dim)
{
    if (dtid.x >= frame_dim.x || dtid.y >= frame_dim.y) return;
    output_image[dtid.xy] = float4(0.0, 1.0, 0.0, 1.0);
}
""")

# Try reload_all_programs
print("2. Calling device.reload_all_programs()...")
try:
    device.reload_all_programs()
    print("   reload_all_programs() returned OK")
except Exception as e:
    print(f"   reload_all_programs() raised: {e}")

pixel = dispatch_and_read()
print(f"   After reload_all_programs: {pixel}")
if pixel[1] > 0.9:
    print("   PASS: got GREEN after reload_all_programs")
else:
    print("   Still RED")

# Try hot_reload_check
print("\n3. Editing to BLUE and trying hot_reload_check...")
TEST_SHADER.write_text("""\
[shader("compute")]
[numthreads(4, 4, 1)]
void fillColor(
    uint3 dtid: SV_DispatchThreadID,
    RWTexture2D<float4> output_image,
    uniform uint2 frame_dim)
{
    if (dtid.x >= frame_dim.x || dtid.y >= frame_dim.y) return;
    output_image[dtid.xy] = float4(0.0, 0.0, 1.0, 1.0);
}
""")

import time
time.sleep(0.5)

print("   Calling device.hot_reload_check()...")
try:
    result = device.hot_reload_check()
    print(f"   hot_reload_check() returned: {result}")
except Exception as e:
    print(f"   hot_reload_check() raised: {e}")

pixel = dispatch_and_read()
print(f"   After hot_reload_check: {pixel}")
if pixel[2] > 0.9:
    print("   PASS: got BLUE after hot_reload_check")
else:
    print(f"   Got: {pixel}")

# Stability: dispatch 10 more times
print("\n4. Stability check...")
for i in range(10):
    try:
        pixel = dispatch_and_read()
    except Exception as e:
        print(f"   FAIL at iteration {i}: {e}")
        sys.exit(1)
print(f"   10 dispatches OK, last pixel: {pixel}")

TEST_SHADER.unlink(missing_ok=True)
print("\nALL DONE")
