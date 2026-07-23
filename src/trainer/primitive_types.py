"""Primitive/blend specialization registry.

The rasterizer is generic over a *blendable differentiable primitive* + a
*blending law*. A concrete pair — e.g. (GAUSSIAN_3D, ALPHA_OVER) — selects a
:class:`PrimitiveSpecialization`: the reflected parameter struct, its field +
gradient-buffer names, and the shader bundle that projects/evaluates/backprops
it. This is a COMPILE-TIME registry (one specialized kernel bundle per pair),
not a runtime tagged mega-kernel.

Additive scaffolding: today only the 3D-Gaussian specialization is populated
(it is the DEFAULT), and nothing is wired into the rasterizer yet — that lands
in the rasterizer-extraction slice. The field/gradient names are pinned to the
real ``buffers`` contracts so the registry can't silently drift from the buffers
the pipeline actually allocates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from trainer.buffers import (
    GAUSSIAN_FLOAT_COUNT,
    GRAD_2D_SEED_COMPONENTS,
    GRAD_SCALAR_COMPONENTS,
)
from trainer.contract_schema import (
    DISK_FIELDS,
    DISK_FLOAT_COUNT,
    DISK_GRAD_SCALAR_COMPONENTS,
    GAUSSIAN_FIELDS,
)
from trainer.config_types import BlendKind, PrimitiveKind


@dataclass(frozen=True)
class PrimitiveLayoutKey:
    """Identifies one specialized rasterizer bundle."""
    primitive: PrimitiveKind
    blend: BlendKind


@dataclass(frozen=True)
class PrimitiveSpecialization:
    """Everything the pipeline needs to allocate + drive one (primitive, blend)
    specialization. Layout-bearing fields are pinned to the ``buffers`` module's
    real names/counts so a mismatch fails a test, not a training run."""
    key: PrimitiveLayoutKey
    parameter_struct: str                    # reflected Slang struct (buffers sizes from this)
    parameter_field_names: tuple[str, ...]   # semantic stored fields of the primitive
    param_float_count: int                   # flat optimizable scalar count
    grad_field_names: tuple[str, ...]        # per-scalar gradient buffers (SoA)
    grad_2d_seed_names: tuple[str, ...]      # projected 2D gradient seeds
    shader_dir: str                          # shaders/<shader_dir>/ holds its kernels
    pipeline_ready: bool                     # full Pipeline allocation/binning path exists

    @property
    def primitive(self) -> PrimitiveKind:
        return self.key.primitive

    @property
    def blend(self) -> BlendKind:
        return self.key.blend


# --- 3D Gaussian (the current, default specialization) ---------------------
# Parameter layout + gradient schema are identical across blend laws — only the
# compositor differs — so both blend variants share one layout, built once and
# re-keyed. mean(3)+log_scale(3)+rot_quat(4)+opacity(1)+sh0(3) = 14 scalars.
_GAUSSIAN_ALPHA_OVER = PrimitiveSpecialization(
    key=PrimitiveLayoutKey(PrimitiveKind.GAUSSIAN_3D, BlendKind.ALPHA_OVER),
    parameter_struct="Gaussian3D",
    parameter_field_names=tuple(field.name for field in GAUSSIAN_FIELDS),
    param_float_count=GAUSSIAN_FLOAT_COUNT,
    grad_field_names=GRAD_SCALAR_COMPONENTS,
    grad_2d_seed_names=GRAD_2D_SEED_COMPONENTS,
    shader_dir="primitives/gaussian",
    pipeline_ready=True,
)
_GAUSSIAN_HARDENED = replace(
    _GAUSSIAN_ALPHA_OVER,
    key=PrimitiveLayoutKey(PrimitiveKind.GAUSSIAN_3D, BlendKind.HARDENED_EXTINCTION),
)

_DISK_ALPHA_OVER = PrimitiveSpecialization(
    key=PrimitiveLayoutKey(PrimitiveKind.DISK_2D, BlendKind.ALPHA_OVER),
    parameter_struct="Disk2D",
    parameter_field_names=tuple(field.name for field in DISK_FIELDS),
    param_float_count=DISK_FLOAT_COUNT,
    grad_field_names=DISK_GRAD_SCALAR_COMPONENTS,
    grad_2d_seed_names=(),  # disk is already screen-space; no projection bridge
    shader_dir="primitives/disk",
    pipeline_ready=False,
)

_REGISTRY: dict[PrimitiveLayoutKey, PrimitiveSpecialization] = {
    spec.key: spec for spec in (
        _GAUSSIAN_ALPHA_OVER, _GAUSSIAN_HARDENED, _DISK_ALPHA_OVER)
}

# The current 3DGS pipeline's default (bit-identical alpha-over Gaussians).
DEFAULT_KEY = PrimitiveLayoutKey(PrimitiveKind.GAUSSIAN_3D, BlendKind.ALPHA_OVER)


def get_specialization(key: PrimitiveLayoutKey) -> PrimitiveSpecialization:
    """Look up the specialization for a (primitive, blend) pair, or raise with
    the set of registered pairs if it isn't implemented yet."""
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"no primitive specialization registered for {key}; "
            f"available: {sorted((k.primitive, k.blend) for k in _REGISTRY)}"
        ) from None


def default_specialization() -> PrimitiveSpecialization:
    """The default 3D-Gaussian + alpha-over specialization (today's pipeline)."""
    return _REGISTRY[DEFAULT_KEY]


def registered_keys() -> tuple[PrimitiveLayoutKey, ...]:
    return tuple(_REGISTRY)
