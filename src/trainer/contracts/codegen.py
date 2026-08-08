"""Code generation and host adapters for the host<->shader contract.

The dependency-free semantic source of truth is :mod:`trainer.contract_schema`.
This module consumes it to build NumPy storage adapters, generate Slang
artifacts, and expose a non-mutating ``--check`` command for CI.

``tests/test_contracts.py`` pins derived values, generated-file currency,
field-lifecycle coverage, and reflected struct sizes. The GitHub Actions
workflow runs ``--check`` without rewriting the worktree.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from trainer.contracts import schema as _schema


# Backward-compatible public facade over the dependency-free semantic model
FieldSpec = _schema.FieldSpec
ExecutionContract = _schema.ExecutionContract
GAUSSIAN_FIELDS = _schema.GAUSSIAN_FIELDS
DISK_FIELDS = _schema.DISK_FIELDS
GAUSSIAN_FLOAT_COUNT = _schema.GAUSSIAN_FLOAT_COUNT
DISK_FLOAT_COUNT = _schema.DISK_FLOAT_COUNT
GRAD_SCALAR_COMPONENTS = _schema.GRAD_SCALAR_COMPONENTS
DISK_GRAD_SCALAR_COMPONENTS = _schema.DISK_GRAD_SCALAR_COMPONENTS
ADAM_STATE_FLOAT_COUNT = _schema.ADAM_STATE_FLOAT_COUNT
SH_MAX_DEGREE = _schema.SH_MAX_DEGREE
SH_REST_COEFFS_MAX = _schema.SH_REST_COEFFS_MAX
SH_REST_FLOATS_MAX = _schema.SH_REST_FLOATS_MAX
sh_rest_coeffs = _schema.sh_rest_coeffs
sh_degree_from_rest_floats = _schema.sh_degree_from_rest_floats
GRAD_2D_SEED_COMPONENTS = _schema.GRAD_2D_SEED_COMPONENTS
ETA_ACCUM_COMPONENTS = _schema.ETA_ACCUM_COMPONENTS
EXECUTION = _schema.EXECUTION
TILE_WIDTH = _schema.TILE_WIDTH
TILE_HEIGHT = _schema.TILE_HEIGHT
TILE_AREA = _schema.TILE_AREA
CLUSTER_SIZE = _schema.CLUSTER_SIZE
RADIX_PART_SIZE = _schema.RADIX_PART_SIZE
RADIX_GLOBAL_HIST_SIZE = _schema.RADIX_GLOBAL_HIST_SIZE
SCAN_BLOCK_SIZE = _schema.SCAN_BLOCK_SIZE
RASTER_BUCKET_SIZE = _schema.RASTER_BUCKET_SIZE


def gaussian_field_offsets() -> dict[str, int]:
    return _schema.gaussian_field_offsets()


def disk_field_offsets() -> dict[str, int]:
    return _schema.disk_field_offsets()


def gaussian_numpy_dtype() -> np.dtype:
    """Structured NumPy dtype for one packed Gaussian3D (56 bytes, tightly packed
    like the Slang struct). Named-field access (``arr["mean"]``) replaces the
    hand-coded scalar-offset slices in CPU packing, PLY export, and densify
    readback, so a layout change is a one-place edit in this schema."""
    fields: list = []
    for f in GAUSSIAN_FIELDS:
        if f.scalar_count == 1:
            fields.append((f.name, np.float32))
        else:
            fields.append((f.name, np.float32, (f.scalar_count,)))
    return np.dtype(fields)


def optimizer_groups() -> dict[str, tuple[int, int]]:
    return _schema.optimizer_groups()


def gaussian_field_matrix() -> tuple[dict[str, object], ...]:
    return _schema.gaussian_field_matrix()


# --- Slang generation -------------------------------------------------------

GENERATED_SLANG_PATH = "shaders/generated/execution_contract.slang"
GENERATED_PRIMITIVE_SLANG_PATH = "shaders/generated/gaussian_types.slang"
GENERATED_DISK_SLANG_PATH = "shaders/generated/disk_types.slang"
GENERATED_SH_SLANG_PATH = "shaders/generated/sh_types.slang"

_SLANG_BANNER = (
    "// DO NOT EDIT. Generated from src/trainer/contracts.py.\n"
    "// Regenerate with:  python -m trainer.contracts\n"
    "// Validated not-stale by tests/test_contracts.py.\n"
)


def generate_execution_slang(ex: ExecutionContract = EXECUTION) -> str:
    """Emit the Slang header of shared execution + layout constants."""
    consts = [
        ("TILE_WIDTH", ex.tile_width),
        ("TILE_HEIGHT", ex.tile_height),
        ("TILE_AREA", ex.tile_area),
        ("CLUSTER_SIZE", ex.cluster_size),
        ("RADIX_PART_SIZE", ex.radix_part_size),
        ("RADIX_DIGITS", ex.radix_digits),
        ("RADIX_KEYS_PER_THREAD", ex.radix_keys_per_thread),
        ("RADIX_PASSES", ex.radix_passes),
        ("RADIX_GLOBAL_HIST_SIZE", ex.radix_global_hist_size),
        ("SCAN_BLOCK_SIZE", ex.scan_block_size),
        ("RASTER_BUCKET_SIZE", ex.raster_bucket_size),
        ("GAUSSIAN_FLOAT_COUNT", GAUSSIAN_FLOAT_COUNT),
    ]
    lines = [_SLANG_BANNER, "#ifndef TRAINER_EXECUTION_CONTRACT",
             "#define TRAINER_EXECUTION_CONTRACT", ""]
    lines += [f"static const int CONTRACT_{name} = {val};" for name, val in consts]
    lines += ["", "#endif // TRAINER_EXECUTION_CONTRACT", ""]
    return "\n".join(lines)


def generate_primitive_slang() -> str:
    """Emit the Gaussian3D storage struct + AdamState moment struct from the
    schema. Field ORDER is the ABI (packed storage + Adam moment channel order);
    each optimized field contributes an ``<moment_name>_m`` / ``_v`` pair to
    AdamState in the same order."""
    lines = [_SLANG_BANNER, "#ifndef TRAINER_GAUSSIAN_TYPES",
             "#define TRAINER_GAUSSIAN_TYPES", "",
             "// Packed per-gaussian storage; uploaded verbatim by",
             "// trainer.buffer_io.write_gaussians (GAUSSIAN_FLOAT_COUNT float32s).",
             "struct Gaussian3D", "{"]
    for f in GAUSSIAN_FIELDS:
        suffix = f"  // {f.doc}" if f.doc else ""
        lines.append(f"    {f.dtype} {f.name};{suffix}")
    lines += ["};", "", "// Flat optimizer/gradient channel offsets.",
              "namespace GaussianChannel", "{"]
    offsets = gaussian_field_offsets()
    for f in GAUSSIAN_FIELDS:
        lines.append(
            f"    static const uint {f.channel_name} = {offsets[f.name]}u;"
            f"  // {f.scalar_count} scalar(s)")
    lines += [f"    static const int Count = {GAUSSIAN_FLOAT_COUNT};", "}", "",
              "// Apply every schema-declared storage invariant after optimization.",
              "void normalizeGaussian3D(inout Gaussian3D primitive)", "{"]
    for f in GAUSSIAN_FIELDS:
        if f.normalization == "quaternion":
            lines.append(f"    primitive.{f.name} = normalize(primitive.{f.name});")
    lines += ["}", "",
              "// Per-gaussian Adam first/second moments (AoS): the adam kernel loads",
              "// all m/v for one gaussian, steps, writes back (no atomics). One",
              "// (m, v) pair per optimized field, in Gaussian3D field order.",
              "struct AdamState", "{"]
    for f in GAUSSIAN_FIELDS:
        if not f.differentiable:
            continue
        lines.append(f"    {f.dtype} {f.moment_name}_m;")
        lines.append(f"    {f.dtype} {f.moment_name}_v;")
    lines += ["};", "", "#endif // TRAINER_GAUSSIAN_TYPES", ""]
    return "\n".join(lines)


def generate_disk_slang() -> str:
    """Emit Disk2D storage + flat channel offsets from the disk schema."""
    lines = [_SLANG_BANNER, "#ifndef TRAINER_DISK_TYPES",
             "#define TRAINER_DISK_TYPES", "", "struct Disk2D", "{"]
    for field in DISK_FIELDS:
        suffix = f"  // {field.doc}" if field.doc else ""
        lines.append(f"    {field.dtype} {field.name};{suffix}")
    lines += ["};", "", "namespace DiskChannel", "{"]
    offsets = disk_field_offsets()
    for field in DISK_FIELDS:
        lines.append(
            f"    static const uint {field.channel_name} = {offsets[field.name]}u;"
            f"  // {field.scalar_count} scalar(s)")
    lines += [f"    static const int Count = {DISK_FLOAT_COUNT};", "}", "",
              "#endif // TRAINER_DISK_TYPES", ""]
    return "\n".join(lines)


def generate_sh_slang() -> str:
    """Emit the optional view-dependent SH appearance layout constants. The DC
    term is ``Gaussian3D.sh0``; these are the higher-band 'rest' coefficients,
    stored in a SEPARATE buffer allocated only when ``sh_degree > 0`` so the
    default SH0 path keeps the 14-float Gaussian3D ABI."""
    lines = [_SLANG_BANNER, "#ifndef TRAINER_SH_TYPES",
             "#define TRAINER_SH_TYPES", "",
             f"static const int SH_MAX_DEGREE = {SH_MAX_DEGREE};",
             f"static const int SH_REST_COEFFS = {SH_REST_COEFFS_MAX};"
             "  // per channel, excludes the DC term",
             f"static const int SH_REST_FLOATS = {SH_REST_FLOATS_MAX};"
             "  // SH_REST_COEFFS * 3 (RGB)",
             "", "#endif // TRAINER_SH_TYPES", ""]
    return "\n".join(lines)


def generated_artifacts() -> dict[str, str]:
    """Repository-relative generated path -> expected file content."""
    return {
        GENERATED_SLANG_PATH: generate_execution_slang(),
        GENERATED_PRIMITIVE_SLANG_PATH: generate_primitive_slang(),
        GENERATED_DISK_SLANG_PATH: generate_disk_slang(),
        GENERATED_SH_SLANG_PATH: generate_sh_slang(),
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def write_generated_slang(repo: Path | None = None) -> list[str]:
    """Write all generated Slang headers; return their absolute paths."""
    repo = repo or _repo_root()
    written = []
    for rel, text in generated_artifacts().items():
        out = repo / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        written.append(str(out))
    return written


def stale_generated_slang(repo: Path | None = None) -> tuple[str, ...]:
    """Return generated paths that are missing or differ from the schema."""
    repo = repo or _repo_root()
    stale = []
    for rel, expected in generated_artifacts().items():
        path = repo / rel
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            stale.append(rel)
    return tuple(stale)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate contract artifacts, or verify them without mutating the tree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="fail if generated artifacts are missing or stale; write nothing",
    )
    args = parser.parse_args(argv)

    if args.check:
        stale = stale_generated_slang()
        if stale:
            print("stale generated contract artifacts:")
            for path in stale:
                print(f"  {path}")
            print("regenerate with: python -m trainer.contracts")
            return 1
        print("generated contract artifacts are current")
        return 0

    for path in write_generated_slang():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
