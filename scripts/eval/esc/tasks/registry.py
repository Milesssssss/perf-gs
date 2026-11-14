# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Task registry — assembles the change/comprehension families to evaluate."""

from __future__ import annotations

from dataclasses import dataclass

from esc.tasks.adjacent import add_densify_mode_task, adjacent_comprehension_tasks
from esc.tasks.base import ChangeTask, ComprehensionTask


@dataclass(frozen=True)
class TaskFamily:
    """A change-family plus its comprehension probes."""

    name: str
    change_tasks: tuple[ChangeTask, ...]
    comprehension_tasks: tuple[ComprehensionTask, ...]


def default_families() -> tuple[TaskFamily, ...]:
    return (
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
