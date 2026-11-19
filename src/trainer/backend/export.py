# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Export trained Gaussians to PLY."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trainer.backend.render.pipeline import TrainingPipeline
from trainer.data.ply import write_gaussian_ply


def save_ply(pipeline: TrainingPipeline, path: str | Path) -> None:
    """Export the live gaussians as a 3DGS-format .ply (inverse of ``load_ply``)."""
    gaussians = pipeline.primitives.read()
    if len(gaussians) == 0:
        raise ValueError("pipeline has 0 gaussians; nothing to save")
    features_rest = (pipeline.primitives.read_features_rest(len(gaussians))
                     if pipeline.buffers.sh_enabled else None)
    config = pipeline.rasterizer.config
    footprints = (pipeline.buffers.primitive.filter_3d.to_numpy().view(np.float32)[:len(gaussians)]
                  if config.mip_enabled else None)
    write_gaussian_ply(gaussians, path, features_rest=features_rest, filter_3d=footprints,
                       mip_kernel_size=config.kernel_size, active_sh_degree=pipeline.active_sh_degree,
                       natural_vignetting=pipeline.rasterizer.natural_vignetting)
