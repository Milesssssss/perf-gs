# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Run the unmodified SADGS reference ``train.py`` and time only its training loop (the ``train_wall_s`` span).

Env: REF_PIN_SH0=1 pins SH degree 0; REF_ST_BATCH caps views per structure-tensor call; REF_TIMING_OUT gets the JSON.
    SADGS_ROOT=/path/to/SADGS REF_TIMING_OUT=out/timing.json python timed_train.py -s <scene> -i images_4 --eval ...
"""
from __future__ import annotations

import builtins
import json
import os
import runpy
import sys
import time

import torch
from tqdm import tqdm

SADGS_ROOT = os.environ["SADGS_ROOT"]
marks: dict[str, float] = {}
_print = builtins.print
_close = tqdm.close


def _synced_now() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _print_hook(*args, **kwargs) -> None:
    text = " ".join(str(arg) for arg in args)
    if text.startswith("Pre-computing Structure Tensors took"):
        marks["loop_start"] = _synced_now()
        marks["setup_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
    elif text.startswith("Gaussian number:"):
        marks["num_gaussians"] = int(text.split(":")[1])
    _print(*args, **kwargs)


def _close_hook(bar) -> None:
    if "loop_start" in marks and "loop_end" not in marks and bar.total and bar.n >= bar.total:
        marks["loop_end"] = _synced_now()
    _close(bar)


def _chunked(function, chunk: int):
    """train.py's 100-view structure-tensor call overruns VRAM; the tensor is per image, so ``chunk``-view batches match."""
    def run(images, *args, **kwargs):
        return torch.cat([function(images[i:i + chunk], *args, **kwargs)
                          for i in range(0, images.shape[0], chunk)])
    return run


def main() -> None:
    sys.path.insert(0, SADGS_ROOT)
    os.chdir(SADGS_ROOT)
    if os.environ.get("REF_PIN_SH0") == "1":
        from scene.gaussian_model import GaussianModel
        GaussianModel.oneupSHdegree = lambda self: None
    from utils import loss_utils
    chunk = int(os.environ.get("REF_ST_BATCH", "16"))
    for name in ("get_multiscale_structure_tensor_v1", "get_multiscale_structure_tensor_v2"):
        setattr(loss_utils, name, _chunked(getattr(loss_utils, name), chunk))
    builtins.print = _print_hook
    tqdm.close = _close_hook
    script = os.path.join(SADGS_ROOT, "train.py")
    sys.argv = [script, *sys.argv[1:]]
    runpy.run_path(script, run_name="__main__")
    builtins.print = _print

    iterations = int(sys.argv[sys.argv.index("--iterations") + 1])
    record = {
        "iterations": iterations,
        "loop_wall_s": marks["loop_end"] - marks["loop_start"],
        "num_gaussians": marks.get("num_gaussians"),
        "peak_reserved_bytes": max(marks["setup_peak_reserved_bytes"],
                                   torch.cuda.max_memory_reserved()),
        "loop_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "loop_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "pin_sh0": os.environ.get("REF_PIN_SH0") == "1",
        "gpu": torch.cuda.get_device_name(0),
    }
    with open(os.environ["REF_TIMING_OUT"], "w") as handle:
        json.dump(record, handle, indent=1)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
