# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Shared pytest setup for the GPU math tests and standalone scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"

for _sub in ("running", "debug"):
    _p = _SCRIPTS_DIR / _sub
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))


@pytest.fixture(scope="module", params=["vulkan", "d3d12"])
def math_module(request: pytest.FixtureRequest):
    import slangpy as spy

    from trainer.backend.compute.device import create_device

    try:
        device = create_device(backend=request.param)
    except Exception as exc:
        pytest.skip(f"backend {request.param!r} unavailable: {exc}")
    try:
        yield spy.Module.load_from_file(device, str(Path(__file__).with_name("shaders") / "math_tests.slang"))
    finally:
        device.close()
