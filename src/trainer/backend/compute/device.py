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

_REPO_ROOT = Path(__file__).resolve().parents[4]
SHADERS_DIR = _REPO_ROOT / "shaders"

_BACKEND_MAP: dict[Backend, spy.DeviceType] = {
    "vulkan": spy.DeviceType.vulkan,
    "d3d12": spy.DeviceType.d3d12,
}


def create_device(
    backend: Backend = "vulkan",
    enable_debug: bool = False,
    shader_debug_info: bool = False,
    defines: dict[str, str] | None = None,
) -> spy.Device:
    """Create a SlangPy device for the requested backend.

    :param backend: ``"vulkan"`` or ``"d3d12"``.
    :param enable_debug: turn on backend debug layers (Vulkan validation, D3D12
        debug layer). Useful during development; off by default because the
        layer-enabled path can pull in extra loader work.
    :param shader_debug_info: compile shaders with source-level debug info so
        Nsight Graphics can correlate counters/SASS back to Slang source lines.
        On Vulkan there are no PDB files (a D3D concept); the equivalent is
        ``NonSemantic.Shader.DebugInfo`` embedded in the SPIR-V, which Slang
        emits when the ``debug_info`` compiler option is set. Optimization is
        left at the default level so the profiled code is still the real,
        optimized code — only source-line mapping is added. Off by default
        because it enlarges shader binaries and is only useful under a profiler.
    :param defines: preprocessor macros injected into every shader compile
        (e.g. ``{"BACKWARD_PIXEL_STRIDE": "2"}``). Compile-time knobs the host
        wants to sweep without editing shader source. Forces the
        ``compiler_options`` path (like ``shader_debug_info``).
    :return: a configured :class:`slangpy.Device` with ``shaders/`` on the
        Slang include path.
    """
    if backend not in _BACKEND_MAP:
        raise ValueError(f"unknown backend {backend!r}; expected one of {list(_BACKEND_MAP)}")

    if not shader_debug_info and not defines:
        return spy.create_device(
            type=_BACKEND_MAP[backend],
            enable_debug_layers=enable_debug,
            include_paths=[SHADERS_DIR],
        )

    # spy.create_device() doesn't expose compiler_options, so drop to the
    # underlying spy.Device to inject debug_info / defines while replicating the
    # convenience wrapper's include-path setup (slangpy's builtin SHADER_PATH
    # first, then our shaders/). Hot reload stays on to match the normal path.
    compiler_options: dict = {
        "include_paths": [spy.SHADER_PATH, str(SHADERS_DIR)],
    }
    if shader_debug_info:
        compiler_options["debug_info"] = spy.SlangDebugInfoLevel.standard
    if defines:
        compiler_options["defines"] = dict(defines)
    return spy.Device(
        type=_BACKEND_MAP[backend],
        enable_debug_layers=enable_debug,
        enable_hot_reload=True,
        compiler_options=compiler_options,
    )
