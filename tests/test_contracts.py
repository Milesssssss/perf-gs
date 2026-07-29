"""Contract tests for the host<->shader ABI (trainer.contracts).

Pins that the executable schema (a) derives the shipped Gaussian layout +
execution constants correctly, (b) is the single source the rest of the host
imports from (buffers.py / splat_stream.py collapsed onto it), (c) keeps the
generated Slang header in sync, and (d) matches the COMPILED Gaussian3D struct
via reflection — the review's "reflection as validation" over the generated
contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trainer.contracts import codegen as C


# --- CPU: derived layout is correct -----------------------------------------

def test_gaussian_layout_derived_values() -> None:
    assert C.GAUSSIAN_FLOAT_COUNT == 14
    assert C.GRAD_SCALAR_COMPONENTS == (
        "mean_x", "mean_y", "mean_z",
        "log_scale_x", "log_scale_y", "log_scale_z",
        "rot_w", "rot_x", "rot_y", "rot_z",
        "opacity",
        "color_r", "color_g", "color_b",
    )
    # One gradient channel per packed scalar — derived, never declared twice.
    assert len(C.GRAD_SCALAR_COMPONENTS) == C.GAUSSIAN_FLOAT_COUNT
    assert C.gaussian_field_offsets() == {
        "mean": 0, "log_scale": 3, "rot_quat": 6, "opacity": 10, "sh0": 11}
    assert C.optimizer_groups() == {
        "position": (0, 3), "scale": (3, 3), "rotation": (6, 4),
        "opacity": (10, 1), "appearance": (11, 3)}


def test_adam_state_float_count() -> None:
    # Two moments (m, v) per optimized scalar (14) = 28 floats.
    assert C.ADAM_STATE_FLOAT_COUNT == 28


def test_gaussian_numpy_dtype_layout() -> None:
    import numpy as np

    dt = C.gaussian_numpy_dtype()
    assert dt.itemsize == C.GAUSSIAN_FLOAT_COUNT * 4          # tightly packed (56 B)
    assert dt.names == ("mean", "log_scale", "rot_quat", "opacity", "sh0")
    # Structured byte offsets == the contract's scalar offsets * 4.
    off = C.gaussian_field_offsets()
    for name in dt.names:
        assert dt.fields[name][1] == off[name] * 4
    # A packed record flat-views back to the 14-float shader layout.
    rec = np.zeros(1, dt)
    rec["opacity"] = 5.0
    assert rec.view(np.float32).reshape(-1)[off["opacity"]] == 5.0


def test_disk_layout_derived_values() -> None:
    assert C.DISK_FLOAT_COUNT == 7
    assert C.DISK_GRAD_SCALAR_COMPONENTS == (
        "center_x", "center_y", "log_radius", "opacity",
        "color_r", "color_g", "color_b",
    )
    assert C.disk_field_offsets() == {
        "center": 0, "log_radius": 2, "opacity": 3, "color_value": 4}
    generated = C.generate_disk_slang()
    assert "struct Disk2D" in generated
    assert "static const int Count = 7;" in generated


def test_gaussian_field_lifecycle_matrix_complete() -> None:
    matrix = C.gaussian_field_matrix()
    assert tuple(row["field"] for row in matrix) == tuple(
        field.name for field in C.GAUSSIAN_FIELDS)
    required_systems = {
        "storage", "allocation", "initialization", "projection", "gradients",
        "optimizer", "compaction", "sorting", "densification",
        "serialization", "export", "normalization",
    }
    for row in matrix:
        assert required_systems <= row.keys()
        assert all(row[name] not in (None, "", ()) for name in required_systems)
    assert {row["field"] for row in matrix if row["normalization"] != "none"} == {
        "rot_quat"}
    assert {row["field"] for row in matrix if row["densification"] == "transform"} == {
        "mean", "log_scale"}


def test_projected_and_eta_channels() -> None:
    assert C.GRAD_2D_SEED_COMPONENTS == (
        "mean2d_x", "mean2d_y", "cov2d_a", "cov2d_b", "cov2d_c", "comp")
    assert C.ETA_ACCUM_COMPONENTS == ("n_high", "n_low", "max_x", "max_y", "max_z")


def test_execution_constants() -> None:
    assert (C.TILE_SIZE, C.CLUSTER_SIZE, C.RADIX_PART_SIZE, C.SCAN_BLOCK_SIZE,
            C.RASTER_BUCKET_SIZE) == (16, 128, 2048, 1024, 32)
    # radix_part_size + global_hist_size are DERIVED from independent inputs
    # (digits, keys_per_thread, passes) — never stored a second time.
    assert C.RADIX_PART_SIZE == C.EXECUTION.radix_digits * C.EXECUTION.radix_keys_per_thread
    assert C.RADIX_GLOBAL_HIST_SIZE == C.EXECUTION.radix_digits * C.EXECUTION.radix_passes
    assert C.RADIX_GLOBAL_HIST_SIZE == 2048


# --- CPU: the collapse kept every host name/value ---------------------------

def test_buffers_reexports_match_contract() -> None:
    from trainer.backend.memory import buffers as B
    assert B.GAUSSIAN_FLOAT_COUNT == C.GAUSSIAN_FLOAT_COUNT
    assert B.GRAD_SCALAR_COMPONENTS == C.GRAD_SCALAR_COMPONENTS
    assert B.GRAD_2D_SEED_COMPONENTS == C.GRAD_2D_SEED_COMPONENTS
    assert B.ETA_ACCUM_COMPONENTS == C.ETA_ACCUM_COMPONENTS
    assert (B.TILE_SIZE, B.CLUSTER_SIZE, B.RADIX_PART_SIZE,
            B.RADIX_GLOBAL_HIST_SIZE, B.SCAN_BLOCK_SIZE, B.RASTER_BUCKET_SIZE) == (
        C.TILE_SIZE, C.CLUSTER_SIZE, C.RADIX_PART_SIZE,
        C.RADIX_GLOBAL_HIST_SIZE, C.SCAN_BLOCK_SIZE, C.RASTER_BUCKET_SIZE)


def test_splat_stream_floats_per_splat_from_contract() -> None:
    from trainer.frontend.viewer import splat_stream
    assert splat_stream.FLOATS_PER_SPLAT == C.GAUSSIAN_FLOAT_COUNT


# --- CPU: the generated Slang header is not stale ---------------------------

def test_generated_slang_not_stale() -> None:
    repo = Path(__file__).resolve().parent.parent
    exec_disk = (repo / C.GENERATED_SLANG_PATH).read_text(encoding="utf-8")
    assert exec_disk == C.generate_execution_slang(), (
        "execution_contract.slang is stale; regenerate with "
        "`python -m trainer.contracts`")
    prim_disk = (repo / C.GENERATED_PRIMITIVE_SLANG_PATH).read_text(encoding="utf-8")
    assert prim_disk == C.generate_primitive_slang(), (
        "gaussian_types.slang is stale; regenerate with "
        "`python -m trainer.contracts`")
    assert C.stale_generated_slang(repo) == ()


def test_contract_check_command() -> None:
    assert C.main(["--check"]) == 0


def test_generated_channels_and_normalization_match_schema() -> None:
    generated = C.generate_primitive_slang()
    offsets = C.gaussian_field_offsets()
    for field in C.GAUSSIAN_FIELDS:
        assert f"static const uint {field.channel_name} = {offsets[field.name]}u;" in generated
        if field.normalization == "quaternion":
            assert f"primitive.{field.name} = normalize(primitive.{field.name});" in generated
    assert f"static const int Count = {C.GAUSSIAN_FLOAT_COUNT};" in generated


def test_runtime_field_handlers_cover_schema() -> None:
    """Mechanical coverage, not a numerical-correctness proof.

    A new schema field must be packed, handed to projection/blending, exposed to
    gradients + Adam, represented by the generic parameter adapter, inherited
    or transformed by densification, and exported. Whole-struct copy assertions
    cover compaction and Morton sorting; behavior tests cover their numerics.
    """
    repo = Path(__file__).resolve().parent.parent

    def source(path: str) -> str:
        return (repo / path).read_text(encoding="utf-8")

    packing = source("src/trainer/backend/memory/buffer_io.py")
    export = source("src/trainer/data/ply.py")
    bin_count = source("shaders/rasterize/tile_binned/bin_count.slang")
    projection_backward = source("shaders/rasterize/tile_binned/projection_backward.slang")
    blend_backward = source("shaders/rasterize/tile_binned/backward_persplat.slang")
    regularization = source("shaders/losses/regularization.slang")
    adam = source("shaders/optimizers/adam.slang")
    parameters = source("shaders/primitives/gaussian/parameters.slang")
    split_simple = source("shaders/densify/split_and_clone.slang")
    split_structure = source("shaders/densify/split_anisotropic.slang")
    compact = source("shaders/densify/scatter_compact.slang")
    morton = source("shaders/densify/morton_sort.slang")

    assert "gaussians_out[dst]  = gaussians_in[i];" in compact
    assert "gaussians_out[i]  = gaussians_in[src];" in morton
    assert "Gaussian3D child = parent;" in split_simple
    assert "Gaussian3D child = parent;" in split_structure
    assert "normalizeGaussian3D(g);" in adam
    assert "normalizeGaussian3D(primitive);" in parameters

    gradient_sources = projection_backward + blend_backward + regularization
    for field in C.GAUSSIAN_FIELDS:
        assert f'block["{field.name}"]' in packing
        # export path (data/ply.py) subscripts each field by name; the local
        # variable name is irrelevant to "is every field serialized?".
        assert f'["{field.name}"]' in export
        assert f"primitive.{field.name}" in parameters
        assert f"g.{field.name}" in adam
        if field.projection_role == "geometry":
            assert f"model.{field.name} = g.{field.name};" in bin_count
        else:
            assert f"bd.{field.name} = g.{field.name};" in bin_count
        for gradient in field.gradients:
            assert f"grad_{gradient}" in gradient_sources
            assert f"grad_{gradient}" in adam
        if field.densification == "transform":
            assert f"child.{field.name}" in split_simple
            assert f"child.{field.name}" in split_structure


# --- GPU: reflection validates the compiled struct matches the contract -----

@pytest.mark.parametrize("backend", ["vulkan"])
def test_gaussian_struct_size_matches_contract(backend) -> None:
    import slangpy as spy
    from trainer.backend.memory import buffers as B
    from trainer.backend.compute.device import create_device
    from trainer.backend.render.pipeline import _module_for_reflection

    device = create_device(backend=backend)
    module = _module_for_reflection(device)
    types = B._reflect_types(module)
    buf = spy.NDBuffer(device=device, dtype=types.gaussian, shape=(1,))
    # Gaussian3D is a tightly-packed 14 float32s — buffer_io.write_gaussians
    # uploads exactly that — so one element must be GAUSSIAN_FLOAT_COUNT floats.
    # Catches a Slang struct edit the schema didn't follow.
    assert buf.storage.size == C.GAUSSIAN_FLOAT_COUNT * 4
    adam = spy.NDBuffer(device=device, dtype=types.adam, shape=(1,))
    assert adam.storage.size == C.ADAM_STATE_FLOAT_COUNT * 4
