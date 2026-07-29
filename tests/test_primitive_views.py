"""Tests for the non-owning primitive views (slice 7).

Freezes the crucial property that makes future zero-copy producers possible: a
view ALIASES the underlying buffers (same object identity), never copies them,
and the gradient view exposes exactly the 14 SoA scalar-gradient buffers. Uses a
lightweight fake buffers object so these stay fast (no GPU).
"""

from __future__ import annotations

import dataclasses

import pytest

from trainer.backend.memory.buffers import GRAD_SCALAR_COMPONENTS
from trainer.backend.primitives.primitive_types import DEFAULT_KEY
from trainer.backend.primitives.primitive_views import (
    PrimitiveGradientView,
    PrimitiveSetView,
    gaussian_gradient_view,
    gaussian_primitive_set_view,
)


class _FakeBuffers:
    """Minimal stand-in exposing only the attributes the view factories read.
    Sentinel objects stand in for NDBuffers; identity is what we assert on."""

    def __init__(self) -> None:
        self.max_splats = 123
        self.gaussians = object()
        self.num_gaussians_counter = object()
        self.filter_3D = object()
        self.grad_buffers = {name: object() for name in GRAD_SCALAR_COMPONENTS}


def test_set_view_aliases_buffers_zero_copy() -> None:
    fb = _FakeBuffers()
    v = gaussian_primitive_set_view(fb)
    assert isinstance(v, PrimitiveSetView)
    assert v.parameters is fb.gaussians          # same object -> no copy
    assert v.count is fb.num_gaussians_counter
    assert v.aux_filter_3d is fb.filter_3D
    assert v.capacity == 123
    assert v.layout == DEFAULT_KEY


def test_gradient_view_maps_all_14_zero_copy() -> None:
    fb = _FakeBuffers()
    gv = gaussian_gradient_view(fb)
    assert isinstance(gv, PrimitiveGradientView)
    assert set(gv.buffers) == set(GRAD_SCALAR_COMPONENTS)
    assert len(gv.buffers) == 14
    assert all(gv.buffers[n] is fb.grad_buffers[n] for n in GRAD_SCALAR_COMPONENTS)
    assert gv.layout == DEFAULT_KEY


def test_views_are_frozen() -> None:
    v = gaussian_primitive_set_view(_FakeBuffers())
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.capacity = 5  # type: ignore[misc]
