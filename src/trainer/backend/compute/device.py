# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Device creation: a SlangPy device (Vulkan primary, D3D12 cross-check, Metal on
Apple Silicon) with the packaged or source ``shaders/`` directory on the Slang include path."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from typing import Literal

import slangpy as spy

Backend = Literal["vulkan", "d3d12", "metal"]
# Slang 2026.5.2, the compiler shipped in this wheel, is what the Metal sources were checked against.
METAL_SLANGPY_VERSION = "0.42.0"

_REPO_ROOT = Path(__file__).resolve().parents[4]
SHADERS_DIR = Path(str(files("trainer").joinpath("shaders")))
if not SHADERS_DIR.is_dir():
    SHADERS_DIR = _REPO_ROOT / "shaders"  # editable/source checkout

_BACKEND_MAP: dict[Backend, spy.DeviceType] = {
    "vulkan": spy.DeviceType.vulkan,
    "d3d12": spy.DeviceType.d3d12,
    "metal": spy.DeviceType.metal,
}


def apple_silicon() -> bool:
    """True on Apple Silicon macOS, where the trainer's default device is Metal."""
    return sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def default_backend() -> Backend:
    """Metal on Apple Silicon; Vulkan everywhere else, including Intel Macs."""
    return "metal" if apple_silicon() else "vulkan"


def _warn_metal_slangpy_version() -> None:
    try:
        installed = version("slangpy")
    except PackageNotFoundError:
        return
    # 0.42.0 is the known-good published wheel (Metal indirect is CPU-replayed).
    # A source-built 0.43.0 that bundles slang-rhi PR #800 uses hardware indirect.
    # Official 0.43.1 wheels pin an older slang-rhi without that Metal implementation.
    if installed in (METAL_SLANGPY_VERSION, "0.43.0"):
        return
    if installed == "0.43.1":
        print(
            "warning: official slangpy 0.43.1 wheels pin a slang-rhi without Metal "
            "dispatchComputeIndirect; use 0.42.0 or a source build of slangpy main",
            file=sys.stderr,
        )
        return
    print(
        f"warning: the Metal backend was checked with slangpy {METAL_SLANGPY_VERSION} "
        f"(Slang 2026.5.2) and a source-built 0.43.0; this environment has slangpy {installed}",
        file=sys.stderr,
    )


def create_device(
    backend: Backend | None = None,
    enable_debug: bool = False,
    shader_debug_info: bool = False,
    defines: dict[str, str] | None = None,
) -> spy.Device:
    """Create a SlangPy device with ``shaders/`` on the include path.

    ``backend`` defaults to Metal on Apple Silicon and Vulkan elsewhere.
    ``shader_debug_info`` adds source-line mapping for Nsight without lowering optimization;
    ``defines`` are preprocessor macros injected into every shader compile.
    """
    if backend is None:
        backend = default_backend()
    if backend not in _BACKEND_MAP:
        raise ValueError(f"unknown backend {backend!r}; expected one of {list(_BACKEND_MAP)}")
    if backend == "metal" and sys.platform == "darwin":
        _warn_metal_slangpy_version()

    if not shader_debug_info and not defines:
        return spy.create_device(
            type=_BACKEND_MAP[backend],
            enable_debug_layers=enable_debug,
            include_paths=[SHADERS_DIR],
        )

    # spy.create_device() hides compiler_options; replicate its include paths and hot reload.
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


def create_slang_session(device: spy.Device, defines: dict[str, str] | None) -> spy.SlangSession:
    """A session compiling like the device's own, but with ``defines`` as its macros."""
    # Copied field by field: desc.compiler_options is the device's live object, not a copy.
    own = device.slang_session.desc.compiler_options
    return device.create_slang_session(compiler_options=spy.SlangCompilerOptions({
        "include_paths": own.include_paths,
        "debug_info": own.debug_info,
        "defines": dict(defines or {}),
    }))
