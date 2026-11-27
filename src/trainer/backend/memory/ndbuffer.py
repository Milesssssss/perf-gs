# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""NDBuffer constructor compatibility for slangpy 0.42 and 0.43+.

slangpy 0.43 removed the legacy ``NDBuffer`` / ``NativeNDBuffer`` types in favor
of ``Tensor``. The trainer still talks about typed GPU arrays the same way; this
module hides the constructor difference so call sites stay identical in meaning.
"""

from __future__ import annotations

from typing import Any

import slangpy as spy

_DEFAULT_USAGE = spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access

if hasattr(spy, "NDBuffer"):
    NDBuffer = spy.NDBuffer  # type: ignore[attr-defined]
else:
    NDBuffer = spy.Tensor


def make_ndbuffer(
    device: spy.Device,
    dtype: Any,
    shape,
    usage: spy.BufferUsage = _DEFAULT_USAGE,
    memory_type: spy.MemoryType = spy.MemoryType.device_local,
    program_layout=None,
):
    """Allocate a typed GPU array. 0.42 returns ``NDBuffer``; 0.43+ returns ``Tensor``."""
    if isinstance(shape, int):
        shape = (shape,)
    if hasattr(spy, "NDBuffer"):
        return spy.NDBuffer(  # type: ignore[attr-defined]
            device=device, dtype=dtype, shape=shape, usage=usage,
            memory_type=memory_type, program_layout=program_layout)
    return spy.Tensor.empty(
        device, shape, dtype=dtype, usage=usage,
        memory_type=memory_type, program_layout=program_layout)


def make_labeled_ndbuffer(device: spy.Device, dtype: Any, shape, label: str,
                          usage: spy.BufferUsage):
    """Like ``make_ndbuffer``, but names the underlying buffer when the API allows it."""
    try:
        from slangpy.reflection.lookup import resolve_element_type, resolve_program_layout

        program_layout = resolve_program_layout(device, dtype, None)
        element_type = resolve_element_type(program_layout, dtype)
        element_count = 1
        for dim in shape:
            element_count *= dim
        size = element_count * element_type.buffer_layout.reflection.stride
        buffer = device.create_buffer(
            size=size, usage=usage,
            memory_type=spy.MemoryType.device_local, label=label)
        if hasattr(spy, "NDBuffer"):
            from slangpy.core.native import NativeNDBuffer, NativeNDBufferDesc, Shape
            shp = Shape(tuple(shape))
            desc = NativeNDBufferDesc()
            desc.usage = usage
            desc.memory_type = spy.MemoryType.device_local
            desc.shape = shp
            desc.strides = shp.calc_contiguous_strides()
            desc.dtype = element_type
            desc.element_layout = element_type.buffer_layout.reflection
            return NativeNDBuffer(device, desc, buffer)
        return spy.Tensor(buffer, element_type, shape)
    except Exception:
        return make_ndbuffer(device, dtype, shape, usage=usage)
