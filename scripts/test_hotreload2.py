"""Test: can we load_program the same file again to get updated code?"""
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

program = device.load_program(str(TEST_SHADER), ["fillColor"])
kernel = device.create_compute_kernel(program)

output_tex = device.create_texture(
    type=spy.TextureType.texture_2d,
    format=spy.Format.rgba32_float,
    width=WIDTH, height=HEIGHT,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
)


def dispatch_and_read(k):
    k.dispatch(
        thread_count=[WIDTH, HEIGHT, 1],
        output_image=output_tex,
        frame_dim=spy.uint2(WIDTH, HEIGHT),
    )
    device.wait_for_idle()
    return output_tex.to_numpy()[0, 0]


pixel = dispatch_and_read(kernel)
print(f"1. Initial dispatch: {pixel}")

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

# Test A: load_program again with old kernel still alive
print("2. Calling load_program on same file (old kernel still alive)...")
try:
    program2 = device.load_program(str(TEST_SHADER), ["fillColor"])
    kernel2 = device.create_compute_kernel(program2)
    pixel = dispatch_and_read(kernel2)
    print(f"   Result: {pixel}")
    if pixel[1] > 0.9:
        print("   PASS: new kernel returns GREEN")
    else:
        print("   FAIL: new kernel still returns RED")
except Exception as e:
    print(f"   FAIL: load_program raised: {e}")
    sys.exit(1)

# Test B: old kernel still works?
print("3. Old kernel still works?")
try:
    pixel = dispatch_and_read(kernel)
    print(f"   Old kernel result: {pixel}")
except Exception as e:
    print(f"   Old kernel raised: {e}")

# Test C: delete old kernel, then use new one
print("4. Delete old kernel, use new one...")
del kernel
del program
try:
    pixel = dispatch_and_read(kernel2)
    print(f"   New kernel after delete: {pixel}")
    print("   PASS")
except Exception as e:
    print(f"   FAIL: {e}")
    sys.exit(1)

TEST_SHADER.unlink(missing_ok=True)
print("ALL DONE")
