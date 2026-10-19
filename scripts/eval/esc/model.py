# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Immutable value objects that flow through the harness.

These carry *no behaviour* beyond trivial derived properties. All scoring logic
lives in :mod:`esc.scoring`; keeping the data inert makes results easy to
serialise, diff between champion and candidate, and store as a dashboard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# --- Subscores and the headline result --------------------------------------

@dataclass(frozen=True)
class SubScores:
    """The six behavioural dimensions, each clamped to ``[0, 1]``."""

    correctness: float
    edit_efficiency: float
    understandability: float
    locality: float
    testability: float
    static_quality: float

    def as_ordered(self) -> tuple[tuple[str, float], ...]:
        return (
            ("C correctness", self.correctness),
            ("E edit_efficiency", self.edit_efficiency),
            ("U understandability", self.understandability),
            ("L locality", self.locality),
            ("T testability", self.testability),
            ("S static_quality", self.static_quality),
        )


@dataclass(frozen=True)
class EscResult:
    """The headline ESC value plus its full diagnostic dashboard."""

    esc: float
    subscores: SubScores
    #: Free-form per-subscore evidence (raw metrics, per-task rows, ...).
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "esc": self.esc,
            "subscores": asdict(self.subscores),
            "diagnostics": self.diagnostics,
        }


# --- Agent telemetry --------------------------------------------------------

@dataclass(frozen=True)
class AgentTelemetry:
    """What one agent run cost. Fed into edit-efficiency ``E``."""

    tokens: int = 0
    wall_time_s: float = 0.0
    retries: int = 0
    tool_calls: int = 0
    patch_lines: int = 0
    files_inspected: int = 0
    failed_test_cycles: int = 0

    def cost_map(self) -> dict[str, float]:
        """The four normalisable cost axes used by ``E``."""
        return {
            "tokens": float(self.tokens),
            "time": float(self.wall_time_s),
            "retries": float(self.retries),
            "patch": float(self.patch_lines),
        }


# --- Change-task run outcome ------------------------------------------------

@dataclass(frozen=True)
class PatchStats:
    """Structural footprint of an applied change (feeds locality ``L``)."""

    files_changed: int = 0
    modules_changed: int = 0
    lines_changed: int = 0
    unrelated_components_touched: int = 0
    duplicate_declarations_updated: int = 0
    used_designated_extension_point: bool = True
    introduced_parallel_path: bool = False


@dataclass(frozen=True)
class GateReport:
    """Outcome of the hard behavioural gates for one candidate/run."""

    passed: bool
    failures: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return self.passed


@dataclass(frozen=True)
class TaskRunResult:
    """One (task, seed) change-task run."""

    task_name: str
    seed: int
    success: bool
    gate_report: GateReport
    telemetry: AgentTelemetry
    patch_stats: PatchStats
    detail: str = ""


@dataclass(frozen=True)
class ComprehensionRunResult:
    """One (comprehension-task, seed) run graded for understandability."""

    task_name: str
    seed: int
    answer_score: float          # [0,1] correctness of the free-text answer
    file_precision: float        # [0,1]
    file_recall: float           # [0,1]
    context_tokens: int
    confidence: float            # agent's self-reported confidence in [0,1]
    was_correct: bool            # thresholded correctness for calibration


# --- Testability inputs -----------------------------------------------------

@dataclass(frozen=True)
class MutationResult:
    """Regression-safety evidence for testability ``T``."""

    mutants_total: int = 0
    mutants_killed: int = 0
    seeded_faults_total: int = 0
    seeded_faults_detected: int = 0
    deterministic: bool = True
    flaky_tests: int = 0
    suite_runtime_s: float = 0.0
    tests_weakened_by_agent: bool = False

    @property
    def mutation_score(self) -> float:
        if self.mutants_total == 0:
            return 0.0
        return self.mutants_killed / self.mutants_total

    @property
    def fault_detection(self) -> float:
        if self.seeded_faults_total == 0:
            return 0.0
        return self.seeded_faults_detected / self.seeded_faults_total
