# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Adjacent change families that stress different extension points.

Only the densification-mode change is a functional CPU-only task; the other families are comprehension probes.
"""

from __future__ import annotations

from esc.model import AgentTelemetry
from esc.tasks.base import Check, CommandChangeTask, ComprehensionTask, HiddenTest

_ADD_DENSIFY_MODE_PROMPT = """\
Add a new densification strategy selectable from the CLI.

Requirements (use these EXACT names):
- Add `DensificationMode.HYBRID = "hybrid"` to trainer.config_types.
- Keep the existing modes (none, simple, structure_aware) working and their
  `from_args` resolution intact.
- Wire the new mode through the densification dispatch so selecting it does not
  crash the training loop's densify tick.
"""

_HIDDEN_DENSIFY_MODE = HiddenTest(
    filename="tests/test_esc_hidden_hybrid_mode.py",
    source='''\
from trainer.config_types import DensificationMode


def test_hybrid_mode_added_without_breaking_existing():
    assert DensificationMode("hybrid") is DensificationMode.HYBRID
    assert DensificationMode("none") is DensificationMode.NONE
    assert DensificationMode("simple") is DensificationMode.SIMPLE
    assert DensificationMode("structure_aware") is DensificationMode.STRUCTURE_AWARE
''',
)


def add_densify_mode_task() -> CommandChangeTask:
    return CommandChangeTask(
        name="add_densification_mode",
        family="densification",
        prompt=_ADD_DENSIFY_MODE_PROMPT,
        baseline=AgentTelemetry(
            tokens=45_000, wall_time_s=480.0, retries=1, tool_calls=22,
            patch_lines=70, files_inspected=7, failed_test_cycles=1,
        ),
        hidden_tests=(_HIDDEN_DENSIFY_MODE,),
        checks=(Check(args=("-c", "import trainer.config_types"),
                      kind="existing", name="config-imports"),),
        expected_change_files=(
            "src/trainer/config_types.py",
            "src/trainer/densify.py",
            "src/trainer/training/densification.py",
        ),
        doc="Extensibility of the densification-strategy dispatch.",
    )


def adjacent_comprehension_tasks() -> tuple[ComprehensionTask, ...]:
    return (
        ComprehensionTask(
            name="add_densification_mode_where",
            question="What must change to add a new densification mode?",
            expected_files=(
                "src/trainer/config_types.py",
                "src/trainer/densify.py",
                "src/trainer/training/densification.py",
            ),
            answer_keywords=("DensificationMode", "densif"),
        ),
        ComprehensionTask(
            name="add_loss_where",
            question="Where do you add a new loss term and hook it into training?",
            expected_files=("src/trainer/losses.py",),
            answer_keywords=("loss",),
        ),
        ComprehensionTask(
            name="replace_optimizer",
            question="How would you replace one optimizer while preserving the others?",
            expected_files=("src/trainer/optimizer.py",),
            answer_keywords=("OptimizerLearningRates", "group"),
        ),
        ComprehensionTask(
            name="new_primitive_field_set",
            question="What determines a new primitive type's field set?",
            expected_files=(
                "src/trainer/contract_schema.py",
                "src/trainer/config_types.py",
            ),
            answer_keywords=("FieldSpec", "PrimitiveKind"),
        ),
    )
