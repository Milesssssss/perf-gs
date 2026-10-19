# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Static structural quality ``S`` — conventional metrics as supporting evidence.

Weighted only 5%: these are local code metrics (cyclomatic/cognitive complexity,
size, fan-out, duplication, import cycles), not a measure of whether a real
repository change is easy. They corroborate the behavioural dimensions rather
than drive the score. Each raw metric is mapped through a linear decreasing
curve between a "good" and a "bad" anchor, then weighted into ``S``.

Python is analysed via the ``ast`` module; other source files contribute their
size only.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from esc.config import EscConfig, StaticThresholds
from esc.repo import RepoScanner, SourceFile


def linear_decreasing(value: float, good: float, bad: float) -> float:
    """1.0 at/below ``good``, 0.0 at/above ``bad``, linear in between."""
    if bad == good:
        return 1.0 if value <= good else 0.0
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return 1.0 - (value - good) / (bad - good)


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile of ``values`` (empty -> 0.0)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(q * (len(ordered) - 1))))
    return ordered[rank]


# --- Per-function complexity ------------------------------------------------

@dataclass
class FunctionMetrics:
    name: str
    cyclomatic: int
    cognitive: int
    length: int


class _FunctionAnalyzer:
    """Computes cyclomatic and (approximate SonarSource) cognitive complexity
    for a single function body."""

    _BRANCHING = (ast.If, ast.For, ast.AsyncFor, ast.While)

    def __init__(self) -> None:
        self.cyclomatic = 1
        self.cognitive = 0

    def analyze(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionMetrics:
        for child in node.body:
            self._walk(child, nesting=0)
        length = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
        return FunctionMetrics(node.name, self.cyclomatic, self.cognitive, length)

    def _walk(self, node: ast.AST, nesting: int) -> None:
        increments_nesting = False

        if isinstance(node, self._BRANCHING):
            self.cyclomatic += 1
            self.cognitive += 1 + nesting
            increments_nesting = True
            if isinstance(node, ast.If) and node.orelse:
                # `else`/`elif` costs one structural point (no nesting bonus).
                self.cognitive += 1
        elif isinstance(node, ast.ExceptHandler):
            self.cyclomatic += 1
            self.cognitive += 1 + nesting
            increments_nesting = True
        elif isinstance(node, ast.BoolOp):
            self.cyclomatic += len(node.values) - 1
            self.cognitive += 1
        elif isinstance(node, ast.IfExp):
            self.cyclomatic += 1
            self.cognitive += 1 + nesting
        elif isinstance(node, ast.comprehension):
            self.cyclomatic += len(node.ifs)
            self.cognitive += len(node.ifs)
        elif isinstance(node, ast.Assert):
            self.cyclomatic += 1
        elif isinstance(node, getattr(ast, "match_case", ())):  # 3.10+
            self.cyclomatic += 1
            self.cognitive += 1 + nesting

        child_nesting = nesting + 1 if increments_nesting else nesting
        for child in ast.iter_child_nodes(node):
            self._walk(child, child_nesting)


@dataclass
class ModuleMetrics:
    rel: str
    module_len: int
    functions: list[FunctionMetrics]
    internal_imports: set[str]
    imported_module_ids: set[str]


def analyze_python_module(source: SourceFile, module_id: str,
                          internal_top_level: set[str]) -> ModuleMetrics | None:
    """Parse one Python file into its structural metrics (None on syntax error)."""
    try:
        tree = ast.parse(source.text)
    except SyntaxError:
        return None

    functions: list[FunctionMetrics] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(_FunctionAnalyzer().analyze(node))

    internal: set[str] = set()
    imported_ids: set[str] = set()
    for node in ast.walk(tree):
        for name in _import_targets(node):
            top = name.split(".")[0]
            if top in internal_top_level and top != module_id.split(".")[0]:
                internal.add(top)
            imported_ids.add(name)

    return ModuleMetrics(
        rel=source.rel,
        module_len=source.line_count,
        functions=functions,
        internal_imports=internal,
        imported_module_ids=imported_ids,
    )


def _import_targets(node: ast.AST) -> Iterable[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module]
    return []


# --- Duplication ------------------------------------------------------------

def _significant_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if len(stripped) < 12 or stripped.startswith(("#", "//", "*", "\"\"\"", "'''")):
            continue
        lines.append(stripped)
    return lines


def duplication_ratio(files: Iterable[SourceFile], window: int = 5) -> float:
    """Fraction of ``window``-line blocks that recur verbatim somewhere in the
    corpus. A cheap, honest proxy for copy-paste."""
    seen: dict[int, int] = {}
    total = 0
    for source in files:
        lines = _significant_lines(source.text)
        for i in range(len(lines) - window + 1):
            block = "\n".join(lines[i : i + window])
            key = hash(block)
            seen[key] = seen.get(key, 0) + 1
            total += 1
    if total == 0:
        return 0.0
    duplicated = sum(count for count in seen.values() if count > 1)
    return duplicated / total


# --- Import-cycle detection (Tarjan SCC) ------------------------------------

def count_import_cycles(graph: dict[str, set[str]]) -> tuple[int, list[list[str]]]:
    """Return (number of non-trivial strongly-connected components, the SCCs)."""
    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    result: list[list[str]] = []

    def strongconnect(node: str) -> None:
        indices[node] = lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        for neighbour in graph.get(node, ()):  # type: ignore[arg-type]
            if neighbour not in indices:
                strongconnect(neighbour)
                lowlink[node] = min(lowlink[node], lowlink[neighbour])
            elif neighbour in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbour])
        if lowlink[node] == indices[node]:
            component: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                component.append(w)
                if w == node:
                    break
            if len(component) > 1:
                result.append(component)

    for node in list(graph):
        if node not in indices:
            strongconnect(node)
    return len(result), result


# --- Orchestration ----------------------------------------------------------

def _module_id(rel: str) -> str:
    dotted = rel.rsplit(".", 1)[0].replace("/", ".")
    if dotted.startswith("src."):
        dotted = dotted[len("src.") :]
    return dotted


def _combine(subscores: dict[str, float], thresholds: StaticThresholds) -> float:
    total_weight = sum(weight for _name, weight in thresholds.metric_weights)
    return sum(
        weight * subscores.get(name, 1.0)
        for name, weight in thresholds.metric_weights
    ) / total_weight


def score_static_quality(root: Path, config: EscConfig) -> tuple[float, dict]:
    """Return ``(S, diagnostics)`` for the codebase rooted at ``root``."""
    scanner = RepoScanner(root, config.scan_excludes, config.source_suffixes)
    all_files = scanner.all_files()
    py_files = [f for f in all_files if f.suffix == ".py"]

    module_ids = {f.rel: _module_id(f.rel) for f in py_files}
    internal_top_level = {mid.split(".")[0] for mid in module_ids.values()}
    internal_top_level |= {mid.rsplit(".", 1)[-1] for mid in module_ids.values()}
    id_lookup = {mid: mid for mid in module_ids.values()}
    stem_lookup: dict[str, str] = {}
    for mid in module_ids.values():
        stem_lookup.setdefault(mid.rsplit(".", 1)[-1], mid)

    modules: list[ModuleMetrics] = []
    for source in py_files:
        metrics = analyze_python_module(source, module_ids[source.rel], internal_top_level)
        if metrics is not None:
            modules.append(metrics)

    cyclos = [fn.cyclomatic for m in modules for fn in m.functions]
    cognitives = [fn.cognitive for m in modules for fn in m.functions]
    func_lens = [fn.length for m in modules for fn in m.functions]
    module_lens = [f.line_count for f in all_files]
    fan_outs = [len(m.internal_imports) for m in modules]

    # Import graph for cycle detection.
    graph: dict[str, set[str]] = {mid: set() for mid in module_ids.values()}
    for source, metrics in zip(py_files, modules):
        src_id = module_ids[source.rel]
        for target in metrics.imported_module_ids:
            resolved = id_lookup.get(target) or stem_lookup.get(target.split(".")[-1])
            if resolved and resolved != src_id:
                graph[src_id].add(resolved)
    cycle_count, cycles = count_import_cycles(graph)

    dup = duplication_ratio(all_files)

    t = config.static
    raw = {
        "cyclomatic": percentile(list(map(float, cyclos)), t.complexity_percentile),
        "cognitive": percentile(list(map(float, cognitives)), t.complexity_percentile),
        "function_len": percentile(list(map(float, func_lens)), t.complexity_percentile),
        "module_len": percentile(list(map(float, module_lens)), t.complexity_percentile),
        "fan_out": percentile(list(map(float, fan_outs)), t.complexity_percentile),
        "duplication": dup,
        "cycles": float(cycle_count),
    }
    sub = {
        "cyclomatic": linear_decreasing(raw["cyclomatic"], t.cyclomatic_good, t.cyclomatic_bad),
        "cognitive": linear_decreasing(raw["cognitive"], t.cognitive_good, t.cognitive_bad),
        "function_len": linear_decreasing(raw["function_len"], t.function_len_good, t.function_len_bad),
        "module_len": linear_decreasing(raw["module_len"], t.module_len_good, t.module_len_bad),
        "fan_out": linear_decreasing(raw["fan_out"], t.fan_out_good, t.fan_out_bad),
        "duplication": linear_decreasing(raw["duplication"], t.duplication_good, t.duplication_bad),
        "cycles": linear_decreasing(raw["cycles"], 0.0, 5.0),
    }
    s = _combine(sub, t)

    diagnostics = {
        "python_modules": len(modules),
        "functions": len(cyclos),
        "raw_p95": raw,
        "subscores": sub,
        "import_cycles": cycles[:10],
        "worst_functions": _worst_functions(modules),
    }
    return s, diagnostics


def _worst_functions(modules: list[ModuleMetrics], top: int = 10) -> list[dict]:
    ranked = sorted(
        ((m.rel, fn) for m in modules for fn in m.functions),
        key=lambda pair: pair[1].cognitive,
        reverse=True,
    )
    return [
        {"module": rel, "function": fn.name, "cognitive": fn.cognitive,
         "cyclomatic": fn.cyclomatic, "length": fn.length}
        for rel, fn in ranked[:top]
    ]
