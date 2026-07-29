"""Diagnostic dashboard rendering.

ESC is one scalar, but the design insists the subscore dashboard is never
discarded. These renderers turn an :class:`EscResult` (or a static-only analysis,
or a loop history) into plain-text tables for the terminal and JSON for storage.
"""

from __future__ import annotations

import json
from typing import Any

from esc.config import EscConfig
from esc.loop import LoopHistory
from esc.model import EscResult

_WEIGHT_LABELS = (
    ("C correctness", "correctness"),
    ("E edit_efficiency", "edit_efficiency"),
    ("U understandability", "understandability"),
    ("L locality", "locality"),
    ("T testability", "testability"),
    ("S static_quality", "static_quality"),
)


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "#" * filled + "-" * (width - filled)


def render_dashboard(result: EscResult, config: EscConfig) -> str:
    weights = dict(zip(
        [k for _l, k in _WEIGHT_LABELS], config.weights.as_tuple()
    ))
    measured = set(result.diagnostics.get("measured_dimensions", []))
    lines = [
        "=" * 64,
        f"  Expected Safe Changeability (ESC):  {result.esc:6.2f} / 100",
        "=" * 64,
        f"  {'dimension':22} {'weight':>7} {'score':>7}  {'':20} measured",
        "  " + "-" * 60,
    ]
    subs = dict(result.subscores.as_ordered())
    for label, key in _WEIGHT_LABELS:
        value = subs[label]
        flag = "yes" if key in measured else "NO"
        lines.append(
            f"  {label:22} {weights[key]:>7.2f} {value:>7.3f}  {_bar(value)} {flag}"
        )
    lines.append("  " + "-" * 60)
    lines.extend(_render_locality(result.diagnostics.get("locality", {})))
    lines.extend(_render_correctness(result.diagnostics.get("correctness", {})))
    lines.extend(_render_static(result.diagnostics.get("static_quality", {})))
    unmeasured = result.diagnostics.get("unmeasured_dimensions", [])
    if unmeasured:
        lines.append("")
        lines.append(f"  note: unmeasured dimensions dropped & renormalised: {', '.join(unmeasured)}")
    return "\n".join(lines)


def _render_locality(diag: dict) -> list[str]:
    concepts = diag.get("concepts")
    if not concepts:
        return []
    out = ["", "  Concept entropy (locality) — lower H_c = more localised:",
           f"    {'concept':26} {'sites':>6} {'modules':>8} {'H_c':>6} {'1-H_c':>6}"]
    for row in concepts:
        out.append(
            f"    {row['concept']:26} {row['sites']:>6} {row['modules']:>8} "
            f"{row['entropy']:>6.3f} {row['localization']:>6.3f}"
        )
    return out


def _render_correctness(diag: dict) -> list[str]:
    per_task = diag.get("per_task")
    if not per_task:
        return []
    out = ["", "  Change-task correctness:"]
    for name, row in per_task.items():
        out.append(f"    {name:32} {row['success']}/{row['total']}  ({row['rate']:.0%})")
    return out


def _render_static(diag: dict) -> list[str]:
    worst = diag.get("worst_functions")
    if not worst:
        return []
    out = ["", "  Highest cognitive-complexity functions:"]
    for row in worst[:5]:
        out.append(
            f"    {row['module']}::{row['function']}  "
            f"cog={row['cognitive']} cyc={row['cyclomatic']} len={row['length']}"
        )
    return out


def render_static_analysis(analysis: dict[str, Any]) -> str:
    lines = [
        "=" * 64,
        "  ESC static analysis (agent-free: S + static L)",
        "=" * 64,
        f"  S static_quality   : {analysis['static_quality']:.3f}",
        f"  L locality (static): {analysis['locality_static']:.3f}",
    ]
    concepts = analysis.get("locality_diagnostics", {}).get("concepts", [])
    if concepts:
        lines.append("")
        lines.append(f"    {'concept':26} {'sites':>6} {'modules':>8} {'H_c':>6} {'1-H_c':>6}")
        for row in concepts:
            lines.append(
                f"    {row['concept']:26} {row['sites']:>6} {row['modules']:>8} "
                f"{row['entropy']:>6.3f} {row['localization']:>6.3f}"
            )
    worst = analysis.get("static_quality_diagnostics", {}).get("worst_functions", [])
    if worst:
        lines.append("")
        lines.append("  Highest cognitive-complexity functions:")
        for row in worst[:8]:
            lines.append(
                f"    {row['module']}::{row['function']}  "
                f"cog={row['cognitive']} cyc={row['cyclomatic']} len={row['length']}"
            )
    return "\n".join(lines)


def render_loop(history: LoopHistory) -> str:
    lines = [
        "=" * 64,
        "  Champion/candidate loop",
        "=" * 64,
        f"  iterations : {len(history.iterations)}",
        f"  final ESC  : {history.final_esc:.2f}",
        "",
        f"  {'#':>3} {'hotspot':22} {'champ':>7} {'cand':>7} {'result':8} reason",
        "  " + "-" * 60,
    ]
    for it in history.iterations:
        flag = "ACCEPT" if it.accepted else "reject"
        lines.append(
            f"  {it.index:>3} {it.hotspot:22} {it.champion_esc:>7.2f} "
            f"{it.candidate_esc:>7.2f} {flag:8} {it.decision.reason}"
        )
    return "\n".join(lines)


def result_to_json(result: EscResult) -> str:
    return json.dumps(result.to_dict(), indent=2, default=str)
