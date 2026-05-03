"""Ultra-minimal dispatch smoke: does RWTexture2D write reach CPU readback?"""

from __future__ import annotations

import numpy as np
import slangpy as spy

from trainer.device import create_device, SHADERS_DIR


def main() -> None:
    W, H = 32, 32
    device = create_device(backend="vulkan")
    program = device.load_program(str(SHADERS_DIR / "common" / "debug_write.slang"), ["debugWrite"])
    kernel = device.create_compute_kernel(program)

    output = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=W, height=H,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )

    # Dispatch one thread per pixel.
    kernel.dispatch(thread_count=[W, H, 1], output_image=output)
    device.wait_for_idle()

    img = output.to_numpy()
    print("shape:", img.shape, "dtype:", img.dtype)
    # Try multiple indices to check where the green pixels landed.
    for (y, x) in [(0, 0), (0, 1), (1, 0), (10, 10), (15, 15), (16, 16), (31, 31)]:
        print(f"  img[{y},{x}] = {img[y, x]}")
    print("min/max:", img.min(), img.max())
    print("total nonzero:", (img[..., :3].sum(axis=-1) > 1e-3).sum())


if __name__ == "__main__":
    main()
