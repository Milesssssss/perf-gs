"""Task suite: change-task and comprehension-task families for the ESC harness."""

from __future__ import annotations

from esc.tasks.base import (
    ChangeTask,
    Check,
    CommandChangeTask,
    CommandRunner,
    ComprehensionTask,
    HiddenTest,
    LiveCommandRunner,
    TaskEnvironment,
    TaskVerification,
)
from esc.tasks.grading import grade_comprehension
from esc.tasks.registry import (
    TaskFamily,
    all_change_tasks,
    all_comprehension_tasks,
    default_families,
    select_families,
)

__all__ = [
    "ChangeTask",
    "Check",
    "CommandChangeTask",
    "CommandRunner",
    "ComprehensionTask",
    "HiddenTest",
    "LiveCommandRunner",
    "TaskEnvironment",
    "TaskVerification",
    "grade_comprehension",
    "TaskFamily",
    "all_change_tasks",
    "all_comprehension_tasks",
    "default_families",
    "select_families",
]
