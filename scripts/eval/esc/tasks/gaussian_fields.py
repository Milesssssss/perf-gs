# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The anchor task family: adding a Gaussian field.

Per the ESC design, *"median success-adjusted cost of adding a new optimizable
parameter"* is the headline metric because that one edit exposes whether fields,
layouts, optimizer state, Python<->Slang sync, initialization, serialization,
gradients and configuration are centralised or scattered.

Two change tasks probe the ``FieldSpec`` seam from different lifecycle angles:

    add_optimizable_parameter   a differentiable float3 field (full path:
                                gradients, Adam moments, projection, export)
    add_serialized_property     a serialized *non*-differentiable float field
                                (must NOT create optimizer moments)

Both are verified CPU-only against the sandbox's own ``trainer.contract_schema``
so they run without a GPU. Fixed field names keep the hidden tests deterministic.
"""

from __future__ import annotations

from esc.model import AgentTelemetry
from esc.tasks.base import Check, CommandChangeTask, ComprehensionTask, HiddenTest

# CPU-only pins in the real suite that adding a field necessarily invalidates;
# a successful change must update them (a genuine locality cost).
_EXISTING_CONTRACT_PINS = Check(
    args=(
        "-m", "pytest",
        "tests/test_contracts.py::test_gaussian_layout_derived_values",
        "tests/test_contracts.py::test_adam_state_float_count",
        "tests/test_contracts.py::test_gaussian_numpy_dtype_layout",
        "-q", "-p", "no:cacheprovider",
    ),
    kind="existing",
    name="contract-pins",
)

# Generated Slang must be regenerated (`python -m trainer.contracts`) or this
# fails — the ABI/currency gate.
_CONTRACTS_CHECK = Check(
    args=("-m", "trainer.contracts", "--check"),
    kind="abi",
    name="contracts --check",
)


_ADD_OPTIMIZABLE_PROMPT = """\
Add a new optimizable Gaussian field to the 3DGS trainer.

Requirements (use these EXACT names):
- Field name: `reflectance`, dtype `float3`.
- It must be differentiable (own gradient channels + Adam moments) and
  serialized on export.
- Give it optimizer_group `reflectance`.
- Wire it end-to-end: declare it in the FieldSpec source of truth
  (trainer.contract_schema.GAUSSIAN_FIELDS), regenerate the contract Slang
  (`python -m trainer.contracts`), and update any lifecycle handlers,
  optimizer learning-rate API, and pinned contract tests so the suite stays
  green.
- Keep all existing behaviour and numerical outputs unchanged for the other
  fields.
"""

_ADD_SERIALIZED_PROMPT = """\
Add a new *serialized but non-differentiable* Gaussian field.

Requirements (use these EXACT names):
- Field name: `confidence`, dtype `float`.
- It must be serialized on export but NOT differentiable: it must create no
  optimizer moments and no Adam state.
- Declare it in trainer.contract_schema.GAUSSIAN_FIELDS with
  `differentiable=False`, regenerate the contract Slang
  (`python -m trainer.contracts`), and update the pinned contract tests.
- Do not change the Adam state layout for the existing differentiable fields.
"""

_HIDDEN_ADD_OPTIMIZABLE = HiddenTest(
    filename="tests/test_esc_hidden_reflectance.py",
    source='''\
from trainer import contracts as C


def test_reflectance_is_wired_end_to_end():
    offsets = C.gaussian_field_offsets()
    assert "reflectance" in offsets, "field missing from schema"
    assert C.GAUSSIAN_FLOAT_COUNT == 17, "float3 field should add 3 scalars"
    assert len(C.GRAD_SCALAR_COMPONENTS) == C.GAUSSIAN_FLOAT_COUNT
    assert C.ADAM_STATE_FLOAT_COUNT == 34, "differentiable field needs Adam moments"
    assert "reflectance" in C.optimizer_groups(), "optimizer group not registered"
    assert "reflectance" in C.gaussian_numpy_dtype().names
''',
)

_HIDDEN_ADD_SERIALIZED = HiddenTest(
    filename="tests/test_esc_hidden_confidence.py",
    source='''\
from trainer import contracts as C


def test_confidence_is_serialized_but_not_differentiable():
    offsets = C.gaussian_field_offsets()
    assert "confidence" in offsets, "field missing from schema"
    assert C.GAUSSIAN_FLOAT_COUNT == 15, "float field should add 1 scalar"
    # differentiable=False => no additional Adam moments beyond the base 28.
    assert C.ADAM_STATE_FLOAT_COUNT == 28, "non-differentiable field must not add moments"
''',
)


def add_optimizable_parameter_task() -> CommandChangeTask:
    return CommandChangeTask(
        name="add_optimizable_parameter",
        family="gaussian_field",
        prompt=_ADD_OPTIMIZABLE_PROMPT,
        baseline=AgentTelemetry(
            tokens=80_000, wall_time_s=900.0, retries=1, tool_calls=40,
            patch_lines=140, files_inspected=12, failed_test_cycles=1,
        ),
        hidden_tests=(_HIDDEN_ADD_OPTIMIZABLE,),
        checks=(_CONTRACTS_CHECK, _EXISTING_CONTRACT_PINS),
        expected_change_files=(
            "src/trainer/contract_schema.py",
            "src/trainer/optimizer.py",
            "shaders/generated/gaussian_types.slang",
            "shaders/generated/execution_contract.slang",
            "shaders/optimizers/adam.slang",
            "tests/test_contracts.py",
        ),
        doc="Headline metric: cost of adding a new optimizable parameter.",
    )


def add_serialized_property_task() -> CommandChangeTask:
    return CommandChangeTask(
        name="add_serialized_property",
        family="gaussian_field",
        prompt=_ADD_SERIALIZED_PROMPT,
        baseline=AgentTelemetry(
            tokens=55_000, wall_time_s=600.0, retries=1, tool_calls=28,
            patch_lines=90, files_inspected=8, failed_test_cycles=1,
        ),
        hidden_tests=(_HIDDEN_ADD_SERIALIZED,),
        checks=(_CONTRACTS_CHECK, _EXISTING_CONTRACT_PINS),
        expected_change_files=(
            "src/trainer/contract_schema.py",
            "shaders/generated/gaussian_types.slang",
            "tests/test_contracts.py",
        ),
        doc="Serialization lifecycle probe: serialized but non-differentiable field.",
    )


def gaussian_field_comprehension_tasks() -> tuple[ComprehensionTask, ...]:
    return (
        ComprehensionTask(
            name="where_declare_parameter",
            question="Where would a new optimizable Gaussian parameter be declared?",
            expected_files=("src/trainer/contract_schema.py",),
            answer_keywords=("FieldSpec", "GAUSSIAN_FIELDS"),
            doc="Source-of-truth location for the field layout.",
        ),
        ComprehensionTask(
            name="who_owns_optimizer",
            question="Which component owns optimizer creation and the Adam step?",
            expected_files=("src/trainer/optimizer.py",),
            answer_keywords=("PrimitiveOptimizer", "adam"),
        ),
        ComprehensionTask(
            name="cli_to_kernel_path",
            question="What is the path from a CLI argument to a GPU kernel dispatch?",
            expected_files=(
                "scripts/running/v2_train.py",
                "src/trainer/training/cli.py",
                "src/trainer/rasterizer.py",
            ),
            answer_keywords=("build_parser", "dispatch"),
        ),
        ComprehensionTask(
            name="field_layout_source_of_truth",
            question="Which single source of truth determines the Gaussian field layout?",
            expected_files=("src/trainer/contract_schema.py",),
            answer_keywords=("GAUSSIAN_FIELDS", "scalar_count"),
        ),
        ComprehensionTask(
            name="opacity_blast_radius",
            question="Which modules are likely affected by changing opacity representation?",
            expected_files=(
                "src/trainer/contract_schema.py",
                "src/trainer/optimizer.py",
            ),
            answer_keywords=("opacity", "sigmoid"),
        ),
    )
