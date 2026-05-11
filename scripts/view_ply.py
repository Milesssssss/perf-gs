"""Load a .ply file and display it in the interactive viewer.

Usage:
    .venv/Scripts/python.exe scripts/view_ply.py path/to/output.ply
    .venv/Scripts/python.exe scripts/view_ply.py path/to/output.ply --width 1280 --height 720
"""

from __future__ import annotations

import argparse

import numpy as np

from trainer.dataset import load_ply
from trainer.device import create_device
from trainer.pipeline import Pipeline
from trainer.viewer import Viewer

def main() -> None:
    ap = argparse.ArgumentParser(description="View a .ply gaussian splat file")
    ap.add_argument("ply", type=str, help="path to the .ply file to view", default=r"C:\Users\sscolari\Desktop\slang_3dgs_trainer\datasets\ply_files\debug\Debug.ply")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--max-splats", type=int, default=1_000_000)
    ap.add_argument("--backend", type=str, default="vulkan", choices=["vulkan", "d3d12"])
    args = ap.parse_args()

    data = load_ply(args.ply)
    n = data["means"].shape[0]
    print(f"loaded {n} gaussians from {args.ply}")

    device = create_device(backend=args.backend)
    pipeline = Pipeline(device=device, width=args.width, height=args.height,
                        max_splats=max(args.max_splats, n))
    pipeline.set_gaussians_from_numpy(**data)

    viewer = Viewer(device, width=args.width, height=args.height,
                    max_splats=max(args.max_splats, n), hot_reload=True)
    viewer.update_gaussians(pipeline)
    viewer.start()

    print("viewer running — close the window to exit")
    viewer._thread.join()


if __name__ == "__main__":
    main()
