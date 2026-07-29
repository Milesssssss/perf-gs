"""Static-quality primitives: curves, percentiles, complexity, cycles."""

from __future__ import annotations

import ast

from esc.scoring.static_quality import (
    _FunctionAnalyzer,
    count_import_cycles,
    linear_decreasing,
    percentile,
)


def test_linear_decreasing_endpoints_and_midpoint():
    assert linear_decreasing(5.0, 6.0, 26.0) == 1.0     # below good
    assert linear_decreasing(26.0, 6.0, 26.0) == 0.0    # at bad
    assert linear_decreasing(16.0, 6.0, 26.0) == 0.5    # midpoint


def test_percentile_nearest_rank():
    assert percentile([1, 2, 3, 4, 5], 0.0) == 1
    assert percentile([1, 2, 3, 4, 5], 1.0) == 5
    assert percentile([], 0.95) == 0.0


def _analyze(source: str):
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    return _FunctionAnalyzer().analyze(fn)


def test_cyclomatic_counts_branches():
    metrics = _analyze("def f(x):\n    if x:\n        return 1\n    return 0\n")
    assert metrics.cyclomatic == 2  # base 1 + one if


def test_cognitive_complexity_grows_with_nesting():
    flat = _analyze("def f(x):\n    if x:\n        return 1\n    return 0\n")
    nested = _analyze(
        "def g(x, y):\n"
        "    if x:\n"
        "        if y:\n"
        "            return 1\n"
        "    return 0\n"
    )
    assert nested.cognitive > flat.cognitive


def test_import_cycle_detection():
    graph = {"a": {"b"}, "b": {"a"}, "c": {"a"}}
    count, components = count_import_cycles(graph)
    assert count == 1
    assert sorted(components[0]) == ["a", "b"]


def test_no_cycles_when_acyclic():
    graph = {"a": {"b"}, "b": {"c"}, "c": set()}
    count, _ = count_import_cycles(graph)
    assert count == 0
