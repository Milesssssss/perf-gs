# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The final held-out PSNR curve: ``psnr_curve.json`` plus a best-effort PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trainer.frontend.reporting.error_log import report_exception


def write_psnr_curve(records: list[dict], *, args: argparse.Namespace, run_dir: Path,
                     holdout_stride: int, n_eval_cams: int) -> None:
    curve = {
        "scene": args.scene,
        "run_name": run_dir.name,
        "eval_holdout": bool(args.eval_holdout),
        "eval_holdout_stride": holdout_stride,
        "n_eval_cams": n_eval_cams,
        "psnr_curve_every": args.psnr_curve_every,
        "iters": [record["iter"] for record in records],
        "psnr_srgb": [record["psnr_srgb"] for record in records],
        "n_gaussians": [record["n_gaussians"] for record in records],
    }
    (run_dir / "psnr_curve.json").write_text(json.dumps(curve, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(10, 6), dpi=120)
        axis.plot(curve["iters"], curve["psnr_srgb"], "-o", ms=3, lw=1.6, color="#1f77b4")
        axis.set_xlabel("iteration")
        axis.set_ylabel("test PSNR sRGB (dB)")
        scene_name = Path(args.scene).name or args.scene
        axis.set_title(f"PSNR vs iteration — {scene_name} ({run_dir.name})")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(run_dir / "psnr_curve.png")
        plt.close(figure)
        print(f"[psnr-curve] wrote {run_dir / 'psnr_curve.json'} + "
              f"psnr_curve.png ({len(curve['iters'])} points)")
    except Exception as error:  # noqa: BLE001 - plotting is best-effort
        report_exception("[psnr-curve] PNG creation failed", error)
        print(f"[psnr-curve] wrote psnr_curve.json but the PNG failed "
              f"({error}); `.venv/Scripts/python.exe -m pip install "
              "matplotlib` to enable it.")
