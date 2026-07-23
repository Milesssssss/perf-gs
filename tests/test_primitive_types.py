"""Contract tests for the primitive/blend specialization registry (slice 6).

Freezes: the DEFAULT specialization is 3D-Gaussian + alpha-over (today's
pipeline); its gradient schema matches the real ``buffers`` contracts; the two
Gaussian blend variants share one layout; and unknown pairs raise clearly.
"""

from __future__ import annotations

import pytest

from trainer.buffers import (
    GAUSSIAN_FLOAT_COUNT,
    GRAD_2D_SEED_COMPONENTS,
    GRAD_SCALAR_COMPONENTS,
)
from trainer.config_types import BlendKind, PrimitiveKind
from trainer.primitive_types import (
    DEFAULT_KEY,
    PrimitiveLayoutKey,
    default_specialization,
    get_specialization,
    registered_keys,
)


def test_default_is_gaussian_alpha_over() -> None:
    spec = default_specialization()
    assert spec.primitive == PrimitiveKind.GAUSSIAN_3D
    assert spec.blend == BlendKind.ALPHA_OVER
    assert DEFAULT_KEY == PrimitiveLayoutKey(PrimitiveKind.GAUSSIAN_3D,
                                             BlendKind.ALPHA_OVER)


def test_gradient_schema_matches_buffers() -> None:
    spec = default_specialization()
    assert spec.parameter_struct == "Gaussian3D"
    assert spec.param_float_count == GAUSSIAN_FLOAT_COUNT == 14
    assert spec.grad_field_names == GRAD_SCALAR_COMPONENTS
    assert len(spec.grad_field_names) == 14
    assert spec.grad_2d_seed_names == GRAD_2D_SEED_COMPONENTS
    assert spec.shader_dir == "primitives/gaussian"


def test_blend_variants_share_layout() -> None:
    alpha = get_specialization(
        PrimitiveLayoutKey(PrimitiveKind.GAUSSIAN_3D, BlendKind.ALPHA_OVER))
    hardened = get_specialization(
        PrimitiveLayoutKey(PrimitiveKind.GAUSSIAN_3D, BlendKind.HARDENED_EXTINCTION))
    # Only the blend law differs; parameter + gradient layout is identical.
    assert hardened.blend == BlendKind.HARDENED_EXTINCTION
    assert (hardened.parameter_struct, hardened.parameter_field_names,
            hardened.param_float_count, hardened.grad_field_names) == (
        alpha.parameter_struct, alpha.parameter_field_names,
        alpha.param_float_count, alpha.grad_field_names)


def test_unknown_pair_raises() -> None:
    with pytest.raises(KeyError):
        get_specialization(
            PrimitiveLayoutKey(PrimitiveKind.DISK_2D, BlendKind.ALPHA_OVER))


def test_registry_only_gaussian_today() -> None:
    keys = registered_keys()
    assert len(keys) == 2
    assert all(k.primitive == PrimitiveKind.GAUSSIAN_3D for k in keys)
