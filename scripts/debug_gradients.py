"""Debug gradient health: check for overflow/underflow in fixed-point accumulators.

Runs a few training steps and reads back the raw int32 gradient buffers,
reporting statistics on saturation and zero-fraction.

Usage:
    .venv/Scripts/python.exe scripts/debug_gradients.py [--scene datasets/scans/cactus]
"""
from __future__ import annotations

import argparse
import numpy as np

from trainer.colmap import load_colmap, load_image
from trainer.device import create_device
from trainer.pipeline import AdamLRs, Pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, default="datasets/synthetic/suzanne_blender")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--backend", type=str, default="vulkan")
    ap.add_argument("--n-iters", type=int, default=10)
    ap.add_argument("--max-splats", type=int, default=800_000)
    args = ap.parse_args()

    scene = load_colmap(args.scene, width=args.width, height=args.height)
    n_cams = len(scene.images)
    n_init = scene.initial_points["means"].shape[0]
    print(f"loaded {n_cams} cameras, {n_init} init points")

    device = create_device(backend=args.backend)
    pipeline = Pipeline(device=device, width=args.width, height=args.height,
                        max_splats=args.max_splats)
    pipeline.set_gaussians_from_numpy(**scene.initial_points)

    targets_srgb = [load_image(e, scene.image_dir, args.width, args.height)
                    for e in scene.images]
    target_textures = [pipeline.create_target_texture(t) for t in targets_srgb]

    rng = np.random.default_rng(42)
    from trainer.buffers import GRAD_SCALAR_COMPONENTS

    INT32_MAX = 2147483647
    INT32_MIN = -2147483648
    FIXED_SCALE = 65536.0  # must match types.slang

    for it in range(1, args.n_iters + 1):
        cam_idx = int(rng.integers(0, n_cams))
        entry = scene.images[cam_idx]
        pipeline.set_target_texture(target_textures[cam_idx])

        # Run forward + loss + backward WITHOUT adam (so we can inspect raw grads).
        pipeline._clear_grads()
        pipeline._run_forward(entry.camera)
        pipeline._run_loss()
        pipeline._run_backward()
        pipeline._run_regularization()
        pipeline.device.wait_for_idle()

        n_live = int(pipeline.buffers.num_gaussians_counter.to_numpy().view(np.uint32)[0])

        print(f"\n{'='*70}")
        print(f"ITER {it}, cam={cam_idx}, n_live={n_live}")
        print(f"{'='*70}")
        print(f"{'component':<16s} {'min':>12s} {'max':>12s} {'mean':>12s} "
              f"{'|max|/INT32':>12s} {'zero%':>7s} {'sat%':>6s}")
        print("-" * 80)

        any_saturated = False
        for name in GRAD_SCALAR_COMPONENTS:
            raw = pipeline.buffers.grad_buffers[name].to_numpy()
            ints = np.frombuffer(raw.tobytes(), dtype=np.int32)[:n_live]

            i_min = int(ints.min())
            i_max = int(ints.max())
            i_mean = float(ints.mean())
            abs_max = max(abs(i_min), abs(i_max))
            frac_of_max = abs_max / INT32_MAX
            zero_frac = (ints == 0).mean() * 100
            sat_count = ((ints == INT32_MAX) | (ints == INT32_MIN)).sum()
            sat_frac = sat_count / max(n_live, 1) * 100

            flag = ""
            if frac_of_max > 0.5:
                flag = " !! HIGH"
                any_saturated = True
            if frac_of_max > 0.9:
                flag = " ** OVERFLOW RISK **"

            print(f"{name:<16s} {i_min:>12d} {i_max:>12d} {i_mean:>12.1f} "
                  f"{frac_of_max:>12.4f} {zero_frac:>6.1f}% {sat_frac:>5.2f}%{flag}")

        # Also check float equivalents.
        print(f"\nFloat gradient magnitudes (after fromFixed):")
        print(f"{'component':<16s} {'min':>12s} {'max':>12s} {'absmax':>12s}")
        print("-" * 55)
        for name in GRAD_SCALAR_COMPONENTS:
            raw = pipeline.buffers.grad_buffers[name].to_numpy()
            ints = np.frombuffer(raw.tobytes(), dtype=np.int32)[:n_live]
            floats = ints.astype(np.float64) / FIXED_SCALE
            print(f"{name:<16s} {floats.min():>12.4f} {floats.max():>12.4f} "
                  f"{np.abs(floats).max():>12.4f}")

        if any_saturated:
            print("\nWARNING: GRADIENTS ARE APPROACHING INT32 OVERFLOW!")
        else:
            print("\nOK: Gradient magnitudes look healthy.")

        # Run adam to advance training state for next iter.
        pipeline._run_adam(AdamLRs())


if __name__ == "__main__":
    main()
