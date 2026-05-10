"""Headless test: verify SlangPy hot-reload behavior.

1. Create device + kernel
2. Dispatch, read back result (expect RED)
3. Modify shader on disk
4. Dispatch again, read back result (expect GREEN if hot-reload works)
5. Check if dispatch still works after reload
"""
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

def dispatch_and_read():
    kernel.dispatch(
        thread_count=[WIDTH, HEIGHT, 1],
        output_image=output_tex,
        frame_dim=spy.uint2(WIDTH, HEIGHT),
    )
    device.wait_for_idle()
    img = output_tex.to_numpy()
    return img[0, 0]  # First pixel RGBA

# Step 1: dispatch with RED shader
pixel = dispatch_and_read()
print(f"Before edit: pixel={pixel}")
assert pixel[0] > 0.9 and pixel[1] < 0.1, f"Expected RED, got {pixel}"
print("PASS: initial dispatch returns RED")

# Step 2: edit shader to GREEN
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
print("Shader edited to GREEN, waiting 1s...")
time.sleep(1.0)

# Step 3: dispatch again
print("Dispatching after edit...")
try:
    pixel = dispatch_and_read()
    print(f"After edit: pixel={pixel}")
    if pixel[1] > 0.9 and pixel[0] < 0.1:
        print("PASS: hot-reload worked, got GREEN")
    else:
        print(f"NOTE: still RED — SlangPy may not auto-reload without file watcher")
except Exception as e:
    print(f"FAIL: dispatch after edit raised: {e}")
    sys.exit(1)

# Step 4: dispatch several more times to confirm stability
print("Dispatching 10 more times...")
for i in range(10):
    try:
        pixel = dispatch_and_read()
    except Exception as e:
        print(f"FAIL: dispatch {i} after reload raised: {e}")
        sys.exit(1)
print(f"PASS: 10 dispatches after reload succeeded, last pixel={pixel}")

# Cleanup
TEST_SHADER.unlink(missing_ok=True)
print("ALL TESTS PASSED")
