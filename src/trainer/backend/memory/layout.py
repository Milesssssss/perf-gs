# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Numpy dtypes for shader structs, read from slangpy reflection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import slangpy as spy

_Kind = spy.TypeReflection.Kind
_Scalar = spy.TypeReflection.ScalarType
_NUMPY_SCALARS = {
    _Scalar.float32: np.float32,
    _Scalar.float16: np.float16,
    _Scalar.int32: np.int32,
    _Scalar.uint32: np.uint32,
    _Scalar.int64: np.int64,
    _Scalar.uint64: np.uint64,
}


@dataclass(frozen=True)
class StructLayout:
    """A shader struct's byte layout as a structured numpy dtype."""

    name: str
    dtype: np.dtype

    @property
    def float_count(self) -> int:
        return self.dtype.itemsize // 4

    @classmethod
    def of(cls, slang_type) -> StructLayout:
        layout = slang_type.buffer_layout.reflection
        names, formats, offsets = [], [], []
        for field in layout.fields:
            field_type = field.type_layout.type
            scalar = _NUMPY_SCALARS[field_type.scalar_type]
            if field_type.kind == _Kind.scalar:
                formats.append(scalar)
            elif field_type.kind == _Kind.vector:
                formats.append((scalar, (field_type.col_count,)))
            else:
                formats.append((scalar, (field_type.row_count, field_type.col_count)))
            names.append(field.name)
            offsets.append(field.offset)
        dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                          "itemsize": layout.stride})
        return cls(slang_type.name, dtype)
