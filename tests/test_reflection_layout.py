# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Reflection/layout freeze: the Python <-> shader struct contract (slice 5).

buffers.py sizes every persistent NDBuffer from slangpy reflection of the shared
shader structs, so their byte layout is a hard Python<->shader contract. This
freezes the reflected size/stride/alignment of each struct AND proves the
neutral shared module (``common/types.slang`` — now the reflection source used
by Pipeline) agrees byte-for-byte with the optimizer module that previously
provided them (``optimizers/adam.slang``, which merely ``#include``s
types.slang). That makes the reflection repoint provably layout-preserving.
"""

from __future__ import annotations

import pytest
import slangpy as spy

from trainer.backend.memory.buffers import _reflect_types
from trainer.backend.compute.device import SHADERS_DIR, create_device

BACKENDS = ["vulkan", "d3d12"]

# Frozen byte size per struct — all tightly packed float32 (stride == size):
# Gaussian3D=14, Projected2DGaussian=7, BlendData=10, AdamState=28 floats.
FROZEN_SIZES = {
    "Gaussian3D": 56,
    "Projected2DGaussian": 28,
    "BlendData": 40,
    "AdamState": 112,
}


def _device(backend: str) -> spy.Device:
    try:
        return create_device(backend=backend)
    except Exception as exc:  # backend unavailable on host
        pytest.skip(f"backend {backend!r} unavailable: {exc}")


def _types_module(device: spy.Device) -> spy.Module:
    return spy.Module.load_from_file(device, str(SHADERS_DIR / "common" / "types.slang"))


@pytest.mark.parametrize("backend", BACKENDS)
def test_types_module_reflects_all_layout_structs(backend: str) -> None:
    """common/types.slang is a valid neutral reflection source for all 4 structs
    buffers.py needs (this is exactly what Pipeline now loads)."""
    t = _reflect_types(_types_module(_device(backend)))  # raises if any missing
    assert all(x is not None for x in (t.gaussian, t.projected, t.blend, t.adam))


@pytest.mark.parametrize("backend", BACKENDS)
def test_frozen_struct_sizes(backend: str) -> None:
    m = _types_module(_device(backend))
    for name, size in FROZEN_SIZES.items():
        bl = m.layout.find_type_by_name(name).buffer_layout
        assert bl.size == size, f"{name} size {bl.size} != frozen {size}"
        assert bl.stride == size, f"{name} stride {bl.stride} != frozen {size}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_types_matches_adam_reflection(backend: str) -> None:
    """The neutral module (types.slang) and the optimizer module (adam.slang,
    which #includes it) reflect byte-identical layouts, so repointing the
    reflection source from adam.slang to types.slang cannot change any buffer."""
    d = _device(backend)
    m_types = _types_module(d)
    m_adam = spy.Module.load_from_file(d, str(SHADERS_DIR / "optimizers" / "adam.slang"))
    for name in FROZEN_SIZES:
        a = m_adam.layout.find_type_by_name(name).buffer_layout
        t = m_types.layout.find_type_by_name(name).buffer_layout
        assert (t.size, t.stride, t.alignment) == (a.size, a.stride, a.alignment), name
