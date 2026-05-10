"""Test: try different approaches to force shader reload."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import slangpy as spy

SHADERS_DIR = Path(__file__).resolve().parents[1] / "shaders"
TEST_SHADER = SHADERS_DIR / "test_hotreload.slang"

WIDTH, HEIGHT = 4, 4

# Write initial shader (RED)
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

output_tex = device.create_texture(
    type=spy.TextureType.texture_2d,
    format=spy.Format.rgba32_float,
    width=WIDTH, height=HEIGHT,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
)


def make_kernel():
    program = device.load_program(str(TEST_SHADER), ["fillColor"])
    return device.create_compute_kernel(program)


def dispatch_and_read(k):
    k.dispatch(
        thread_count=[WIDTH, HEIGHT, 1],
        output_image=output_tex,
        frame_dim=spy.uint2(WIDTH, HEIGHT),
    )
    device.wait_for_idle()
    return output_tex.to_numpy()[0, 0]


kernel = make_kernel()
pixel = dispatch_and_read(kernel)
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

# Approach A: Try reload_all_programs if it exists
print("\n--- Approach A: check for reload method ---")
for attr in dir(device):
    if 'reload' in attr.lower() or 'hot' in attr.lower() or 'invalidate' in attr.lower():
        print(f"  found: device.{attr}")

# Approach B: Create a new device
print("\n--- Approach B: new device ---")
device2 = spy.create_device(type=spy.DeviceType.vulkan, include_paths=[SHADERS_DIR])
output_tex2 = device2.create_texture(
    type=spy.TextureType.texture_2d,
    format=spy.Format.rgba32_float,
    width=WIDTH, height=HEIGHT,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
)
program2 = device2.load_program(str(TEST_SHADER), ["fillColor"])
kernel2 = device2.create_compute_kernel(program2)
kernel2.dispatch(
    thread_count=[WIDTH, HEIGHT, 1],
    output_image=output_tex2,
    frame_dim=spy.uint2(WIDTH, HEIGHT),
)
device2.wait_for_idle()
pixel = output_tex2.to_numpy()[0, 0]
print(f"  New device result: {pixel}")
if pixel[1] > 0.9:
    print("  PASS: new device picks up new shader")
else:
    print("  FAIL: even new device returns stale code")
del kernel2, program2, output_tex2, device2

# Approach C: load_program with source string
print("\n--- Approach C: load from source string ---")
src = TEST_SHADER.read_text()
print(f"  Source has 'float4(0.0, 1.0, 0.0, 1.0)': {'0.0, 1.0, 0.0' in src}")

# Check if Module API works differently
print("\n--- Approach D: spy.Module.load_from_source ---")
try:
    mod = spy.Module.load_from_source(device, "test_reload", src)
    print(f"  Module loaded: {mod}")
    print(f"  Module dir: {[a for a in dir(mod) if not a.startswith('_')]}")
except Exception as e:
    print(f"  Module.load_from_source failed: {e}")

# Approach E: Check device.load_program kwargs
print("\n--- Approach E: load_program signature ---")
import inspect
try:
    sig = inspect.signature(device.load_program)
    print(f"  signature: {sig}")
except:
    pass
help_text = device.load_program.__doc__
if help_text:
    print(f"  doc: {help_text[:500]}")

TEST_SHADER.unlink(missing_ok=True)
print("\nDONE")
