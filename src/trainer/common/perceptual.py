"""LPIPS (Zhang et al. 2018) for held-out evaluation, as an OPTIONAL capability.

PSNR and SSIM are numpy-only and always available; LPIPS needs torch plus the
``lpips`` package's pretrained AlexNet. That is a heavyweight dependency for a
trainer whose entire point is not needing one, so it is loaded lazily, on the
CPU, and only when a caller asks for it. Everything degrades to ``None`` if it is
absent — a missing perceptual metric must never fail a training run, and it must
never be silently substituted with something else either.

Why CPU: the GPU is running the thing being measured. A second framework holding
a context on the same device would contend with the workload whose speed is the
experiment. At the evaluation resolution this costs ~50 ms per view, which is
already small next to the render plus SSIM in the same sweep.

Convention follows the reference implementation: inputs are RGB in [0, 1],
scaled to [-1, 1], NCHW. The metric is computed at full evaluation resolution
(no resize), so it is comparable across regimes but not necessarily against a
paper that scored at a different resolution.
"""
from __future__ import annotations

import numpy as np

# Module-level, because constructing the network reads 233 MB of weights and a
# sweep calls this once per view.
_NET = None
_LOADED_NET = ""
_STATE = "unloaded"          # unloaded | ready | unavailable
_REASON = ""


def availability() -> dict:
    """What this module can currently do, for the run report's metadata."""
    return {"state": _STATE, "reason": _REASON,
            "net": (_LOADED_NET or DEFAULT_NET), "input_range": "[-1,1] from sRGB [0,1]",
            "resolution": "full evaluation resolution (no resize)"}


# VGG, not AlexNet. 3DGS, Mip-NeRF 360 and essentially every NeRF/splatting paper
# report the VGG variant, and the two are NOT interchangeable: measured on this
# trainer's held-out bicycle renders, AlexNet reads 0.1662 where VGG reads 0.2194 —
# 1.32x lower. Defaulting to AlexNet made our numbers look better than published
# 30k-iteration results at 3000 iterations, which is how the discrepancy was
# noticed. Keep VGG as the default so a reported LPIPS is comparable to the
# literature; 'alex' stays selectable for a cheap in-loop signal.
DEFAULT_NET = "vgg"


def load(net: str = DEFAULT_NET) -> bool:
    """Load the LPIPS network once. Returns False (and stays quiet afterwards)
    when the dependency is missing, so callers can just check the return."""
    global _NET, _LOADED_NET, _STATE, _REASON
    # Keyed on the net: returning a cached AlexNet for a VGG request would be the
    # same apples-to-oranges error this default exists to prevent.
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


def is_available() -> bool:
    return _STATE == "ready"


def lpips(rendered: np.ndarray, target: np.ndarray,
          net: str = DEFAULT_NET) -> float | None:
    """Per-image LPIPS for HxWx3 sRGB arrays in [0, 1]; ``None`` if unavailable.

    Lower is better. Not comparable to PSNR/SSIM in sign — the tables must say
    so, and the aggregation code must not average it in the same direction as
    the others by accident.
    """
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
