"""Dependency-free semantic schema for host<->shader contracts.

This module is the authoritative model. It deliberately imports no NumPy,
SlangPy, renderer code, or code-generation helpers, so tooling and future
consumers can inspect the schema without importing the trainer runtime.
``trainer.contracts`` consumes this model to generate Slang headers and build
NumPy storage adapters.
"""

from __future__ import annotations

from dataclasses import dataclass

_SCALAR_COUNTS = {"float": 1, "float2": 2, "float3": 3, "float4": 4}


@dataclass(frozen=True)
class FieldSpec:
    """One stored Gaussian field and its lifecycle policy.

    The schema owns names/order/types plus the semantic obligations that cannot
    be inferred from the packed struct alone. The implementation remains
    explicit shader/Python code; tests mechanically verify that each declared
    obligation has a corresponding runtime handler.
    """

    name: str
    dtype: str
    gradients: tuple[str, ...]
    optimizer_group: str
    moment_name: str
    channel_name: str
    initialization: str
    projection_role: str
    densification: str
    export_name: str
    normalization: str | None = None
    serialized: bool = True
    differentiable: bool = True
    doc: str = ""

    @property
    def scalar_count(self) -> int:
        return _SCALAR_COUNTS[self.dtype]


# ORDER IS THE STORAGE ABI, gradient-channel order, and Adam-channel order.
GAUSSIAN_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="mean", dtype="float3",
        gradients=("mean_x", "mean_y", "mean_z"),
        optimizer_group="position", moment_name="mean", channel_name="Mean",
        initialization="point_position", projection_role="geometry",
        densification="transform", export_name="position",
        doc="world-space position",
    ),
    FieldSpec(
        name="log_scale", dtype="float3",
        gradients=("log_scale_x", "log_scale_y", "log_scale_z"),
        optimizer_group="scale", moment_name="log_scale", channel_name="LogScale",
        initialization="knn_log_scale", projection_role="geometry",
        densification="transform", export_name="scale",
        doc="per-axis log scale; exp(log_scale) is the 3D gaussian scale",
    ),
    FieldSpec(
        name="rot_quat", dtype="float4",
        gradients=("rot_w", "rot_x", "rot_y", "rot_z"),
        optimizer_group="rotation", moment_name="rot", channel_name="Rotation",
        initialization="identity_quaternion", projection_role="geometry",
        densification="inherit", export_name="rotation",
        normalization="quaternion",
        doc="rotation quaternion (w, x, y, z)",
    ),
    FieldSpec(
        name="opacity", dtype="float",
        gradients=("opacity",),
        optimizer_group="opacity", moment_name="opacity", channel_name="Opacity",
        initialization="probability_to_logit", projection_role="blend",
        densification="inherit", export_name="opacity",
        doc="pre-sigmoid opacity logit; sigmoid(opacity) is the probability",
    ),
    FieldSpec(
        name="sh0", dtype="float3",
        gradients=("color_r", "color_g", "color_b"),
        optimizer_group="appearance", moment_name="color", channel_name="Color",
        initialization="rgb_to_sh0", projection_role="blend",
        densification="inherit", export_name="f_dc",
        doc="SH band-0 (DC) color coeff; linear RGB = 0.5 + C0*sh0",
    ),
)

# Minimal second primitive: screen-space disk. This intentionally has no
# covariance, quaternion, SH, or Gaussian optimizer state.
DISK_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="center", dtype="float2", gradients=("center_x", "center_y"),
        optimizer_group="position", moment_name="center", channel_name="Center",
        initialization="screen_center", projection_role="geometry",
        densification="inherit", export_name="center",
        doc="screen-space center in pixels",
    ),
    FieldSpec(
        name="log_radius", dtype="float", gradients=("log_radius",),
        optimizer_group="radius", moment_name="log_radius", channel_name="LogRadius",
        initialization="log_radius", projection_role="geometry",
        densification="inherit", export_name="radius",
        doc="logarithm of disk radius in pixels",
    ),
    FieldSpec(
        name="opacity", dtype="float", gradients=("opacity",),
        optimizer_group="opacity", moment_name="opacity", channel_name="Opacity",
        initialization="probability_to_logit", projection_role="blend",
        densification="inherit", export_name="opacity",
        doc="pre-sigmoid opacity logit",
    ),
    FieldSpec(
        name="color_value", dtype="float3",
        gradients=("color_r", "color_g", "color_b"),
        optimizer_group="appearance", moment_name="color", channel_name="Color",
        initialization="rgb", projection_role="blend",
        densification="inherit", export_name="color",
        doc="view-independent RGB color",
    ),
)

GAUSSIAN_FLOAT_COUNT: int = sum(field.scalar_count for field in GAUSSIAN_FIELDS)
GRAD_SCALAR_COMPONENTS: tuple[str, ...] = tuple(
    gradient for field in GAUSSIAN_FIELDS for gradient in field.gradients
)
ADAM_STATE_FLOAT_COUNT: int = sum(
    2 * field.scalar_count for field in GAUSSIAN_FIELDS if field.differentiable
)
DISK_FLOAT_COUNT: int = sum(field.scalar_count for field in DISK_FIELDS)
DISK_GRAD_SCALAR_COMPONENTS: tuple[str, ...] = tuple(
    gradient for field in DISK_FIELDS for gradient in field.gradients
)


# --- Spherical-harmonics appearance (optional view-dependent color) ---------
#
# The band-0 DC coefficient is the always-present ``Gaussian3D.sh0`` field. The
# higher bands (the "rest") are stored SEPARATELY, in their own buffer allocated
# only when ``sh_degree > 0``, so the default SH0 path keeps Gaussian3D at its
# 14-float ABI with zero extra storage or bandwidth. Degree d carries
# ``(d+1)**2 - 1`` rest coefficients per channel (x3 for RGB), activated one
# band at a time (classic 3DGS ``oneupSHdegree``).
SH_MAX_DEGREE: int = 3


def sh_rest_coeffs(degree: int) -> int:
    """Rest SH coefficients PER channel (excludes the DC term) at ``degree``."""
    if not 0 <= degree <= SH_MAX_DEGREE:
        raise ValueError(f"sh degree {degree} out of range [0, {SH_MAX_DEGREE}]")
    return (degree + 1) * (degree + 1) - 1


SH_REST_COEFFS_MAX: int = sh_rest_coeffs(SH_MAX_DEGREE)   # 15 per channel
SH_REST_FLOATS_MAX: int = SH_REST_COEFFS_MAX * 3          # 45 total


def sh_degree_from_rest_floats(rest_floats: int) -> int:
    """Infer the SH degree from a features_rest width (inverse of
    ``sh_rest_coeffs(d) * 3``): ``rest_floats == 3 * ((d+1)**2 - 1)``. Used to
    render an SH ``.ply`` at the degree it was trained with."""
    if rest_floats <= 0:
        return 0
    total = rest_floats // 3 + 1              # (d + 1) ** 2
    degree = int(round(total ** 0.5)) - 1
    return max(0, min(degree, SH_MAX_DEGREE))


def gaussian_field_offsets() -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for field in GAUSSIAN_FIELDS:
        offsets[field.name] = offset
        offset += field.scalar_count
    return offsets


def disk_field_offsets() -> dict[str, int]:
    offsets: dict[str, int] = {}
    offset = 0
    for field in DISK_FIELDS:
        offsets[field.name] = offset
        offset += field.scalar_count
    return offsets


def optimizer_groups() -> dict[str, tuple[int, int]]:
    groups: dict[str, tuple[int, int]] = {}
    offset = 0
    for field in GAUSSIAN_FIELDS:
        groups[field.optimizer_group] = (offset, field.scalar_count)
        offset += field.scalar_count
    return groups


def gaussian_field_matrix() -> tuple[dict[str, object], ...]:
    """Auditable field-by-system ownership table used by tests and reviews."""
    return tuple({
        "field": field.name,
        "storage": field.dtype,
        "allocation": "Gaussian3D",
        "initialization": field.initialization,
        "projection": field.projection_role,
        "gradients": field.gradients,
        "optimizer": field.optimizer_group,
        "compaction": "whole_struct",
        "sorting": "whole_struct",
        "densification": field.densification,
        "serialization": "included" if field.serialized else "excluded",
        "export": field.export_name,
        "normalization": field.normalization or "none",
    } for field in GAUSSIAN_FIELDS)


# Projected 2D gradient seeds: blend backward -> projection backward.
GRAD_2D_SEED_COMPONENTS: tuple[str, ...] = (
    "mean2d_x", "mean2d_y", "cov2d_a", "cov2d_b", "cov2d_c", "comp",
)

# Structure-aware densification multiview accumulators.
ETA_ACCUM_COMPONENTS: tuple[str, ...] = (
    "n_high", "n_low", "max_x", "max_y", "max_z",
)


@dataclass(frozen=True)
class ExecutionContract:
    """Independent algorithm inputs; dependent values are properties."""

    tile_size: int = 16
    cluster_size: int = 128
    radix_digits: int = 256
    radix_keys_per_thread: int = 8
    radix_passes: int = 8
    scan_block_size: int = 1024
    raster_bucket_size: int = 32

    @property
    def radix_part_size(self) -> int:
        return self.radix_digits * self.radix_keys_per_thread

    @property
    def radix_global_hist_size(self) -> int:
        return self.radix_digits * self.radix_passes


EXECUTION = ExecutionContract()
TILE_SIZE = EXECUTION.tile_size
CLUSTER_SIZE = EXECUTION.cluster_size
RADIX_PART_SIZE = EXECUTION.radix_part_size
RADIX_GLOBAL_HIST_SIZE = EXECUTION.radix_global_hist_size
SCAN_BLOCK_SIZE = EXECUTION.scan_block_size
RASTER_BUCKET_SIZE = EXECUTION.raster_bucket_size
