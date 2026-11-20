# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Optional LPIPS (Zhang et al. 2018) for held-out evaluation, lazily loaded on the CPU
(the GPU runs the workload being measured); ``None`` when torch/lpips are absent."""
from __future__ import annotations

import numpy as np

# Module-level: building the network reads ~233 MB of weights and a sweep calls this per view.
_NET = None
_LOADED_NET = ""
_STATE = "unloaded"          # unloaded | ready | unavailable
_REASON = ""


def availability() -> dict:
    """What this module can currently do, for the run report's metadata."""
    return {"state": _STATE, "reason": _REASON,
            "net": (_LOADED_NET or DEFAULT_NET), "input_range": "[-1,1] from sRGB [0,1]",
            "resolution": "full evaluation resolution (no resize)"}


# VGG: the variant the papers report; AlexNet values are not interchangeable (~1.3x lower).
DEFAULT_NET = "vgg"


def load(net: str = DEFAULT_NET) -> bool:
    """Load the LPIPS network once; returns False, and stays quiet afterwards, if unavailable."""
    global _NET, _LOADED_NET, _STATE, _REASON
    # Keyed on the net: a cached AlexNet must never answer a VGG request.
    if _STATE == "ready" and _LOADED_NET == net:
        return True
    if _STATE == "unavailable":
        return False
    try:
        import torch
        import lpips as lpips_module

        torch.set_grad_enabled(False)
        _NET = lpips_module.LPIPS(net=net, verbose=False).eval()
        _LOADED_NET = net
        _STATE = "ready"
        version = getattr(torch, "__version__", "?")
        print(f"[lpips] {net} ready on CPU (torch {version}); "
              f"~50 ms/view at evaluation resolution", flush=True)
        return True
    except Exception as error:                             # noqa: BLE001
        _STATE = "unavailable"
        _REASON = f"{type(error).__name__}: {error}"
        print(f"[lpips] unavailable ({_REASON}); LPIPS columns will be null. "
              f"Install with: pip install --index-url "
              f"https://download.pytorch.org/whl/cpu torch torchvision && "
              f"pip install lpips", flush=True)
        return False


def lpips(rendered: np.ndarray, target: np.ndarray,
          net: str = DEFAULT_NET) -> float | None:
    """Per-image LPIPS (lower is better) of HxWx3 sRGB in [0, 1]; ``None`` if unavailable."""
    if not load(net):
        return None
    import torch

    def to_tensor(image: np.ndarray) -> "torch.Tensor":
        array = np.ascontiguousarray(
            np.clip(image[..., :3], 0.0, 1.0).astype(np.float32))
        # HWC [0,1] -> NCHW [-1,1], the reference implementation's convention.
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor * 2.0 - 1.0

    with torch.no_grad():
        value = _NET(to_tensor(rendered), to_tensor(target))
    return float(value.reshape(-1)[0])
