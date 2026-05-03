"""Device creation for the trainer.

Thin wrapper around ``slangpy.create_device`` that fixes the shader include
path to the repo's ``shaders/`` directory and restricts backend selection to
the two we care about in v0: Vulkan (primary) and D3D12 (cross-check).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import slangpy as spy

Backend = Literal["vulkan", "d3d12"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
SHADERS_DIR = _REPO_ROOT / "shaders"

_BACKEND_MAP: dict[Backend, spy.DeviceType] = {
    "vulkan": spy.DeviceType.vulkan,
    "d3d12": spy.DeviceType.d3d12,
}


def create_device(backend: Backend = "vulkan", enable_debug: bool = False) -> spy.Device:
    """Create a SlangPy device for the requested backend.

    :param backend: ``"vulkan"`` or ``"d3d12"``.
    :param enable_debug: turn on backend debug layers (Vulkan validation, D3D12
        debug layer). Useful during development; off by default because the
        layer-enabled path can pull in extra loader work.
    :return: a configured :class:`slangpy.Device` with ``shaders/`` on the
        Slang include path.
    """
    if backend not in _BACKEND_MAP:
        raise ValueError(f"unknown backend {backend!r}; expected one of {list(_BACKEND_MAP)}")

    return spy.create_device(
        type=_BACKEND_MAP[backend],
        enable_debug_layers=enable_debug,
        include_paths=[SHADERS_DIR],
    )
