"""Task registry — assembles the change/comprehension families to evaluate."""

from __future__ import annotations

from dataclasses import dataclass

from esc.tasks.adjacent import add_densify_mode_task, adjacent_comprehension_tasks
from esc.tasks.base import ChangeTask, ComprehensionTask
from esc.tasks.gaussian_fields import (
    add_optimizable_parameter_task,
    add_serialized_property_task,
    gaussian_field_comprehension_tasks,
)


@dataclass(frozen=True)
class TaskFamily:
    """A change-family plus its comprehension probes."""

    name: str
    change_tasks: tuple[ChangeTask, ...]
    comprehension_tasks: tuple[ComprehensionTask, ...]


def default_families() -> tuple[TaskFamily, ...]:
    """The task suite. The anchor `gaussian_field` family carries the headline
    metric; `densification` widens the probe to a second extension point."""
    return (
        TaskFamily(
            name="gaussian_field",
            change_tasks=(add_optimizable_parameter_task(), add_serialized_property_task()),
            comprehension_tasks=gaussian_field_comprehension_tasks(),
        ),
        TaskFamily(
            name="densification",
            change_tasks=(add_densify_mode_task(),),
            comprehension_tasks=adjacent_comprehension_tasks(),
        ),
    )


def all_change_tasks(families: tuple[TaskFamily, ...]) -> list[ChangeTask]:
    return [task for family in families for task in family.change_tasks]


def all_comprehension_tasks(families: tuple[TaskFamily, ...]) -> list[ComprehensionTask]:
    return [task for family in families for task in family.comprehension_tasks]


def select_families(
    families: tuple[TaskFamily, ...], names: tuple[str, ...]
) -> tuple[TaskFamily, ...]:
    """Filter families by name (empty ``names`` keeps all)."""
    if not names:
        return families
    wanted = frozenset(names)
    return tuple(f for f in families if f.name in wanted)
