# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Hard behavioural gates."""

from __future__ import annotations

from esc.config import GateThresholds
from esc.gates import GateInputs, evaluate_gates


def _ok() -> GateInputs:
    return GateInputs(
        existing_tests_pass=True,
        max_numeric_deviation=0.0,
        perf_ratio=1.0,
        abi_stable=True,
        deterministic=True,
        static_blockers=0,
        mutation_score=0.9,
    )


def test_clean_candidate_passes():
    assert evaluate_gates(_ok()).passed


def test_failing_tests_block():
    report = evaluate_gates(GateInputs(False, 0.0, 1.0, True, True, 0, 0.9))
    assert not report.passed
    assert any("tests" in f for f in report.failures)


def test_perf_regression_blocks():
    report = evaluate_gates(GateInputs(True, 0.0, 1.10, True, True, 0, 0.9))
    assert not report.passed
    assert any("performance" in f for f in report.failures)


def test_abi_break_blocks():
    report = evaluate_gates(GateInputs(True, 0.0, 1.0, False, True, 0, 0.9))
    assert not report.passed
    assert any("ABI" in f for f in report.failures)


def test_mutation_drop_blocks():
    inputs = GateInputs(True, 0.0, 1.0, True, True, 0, 0.80, champion_mutation_score=0.90)
    report = evaluate_gates(inputs, GateThresholds(max_mutation_drop=0.05))
    assert not report.passed
    assert any("mutation" in f for f in report.failures)


def test_numeric_drift_blocks():
    report = evaluate_gates(GateInputs(True, 1e-2, 1.0, True, True, 0, 0.9))
    assert not report.passed
    assert any("numerical" in f for f in report.failures)
