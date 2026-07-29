"""Change locality ``L`` — does touching A force edits in B, C, D, and E?

``L`` blends two views of the same property:

    static   1 - H_c averaged over the codebase's cross-cutting concepts
             (are responsibilities localised?)
    dynamic  the measured spread of the agents' patches (files/modules changed,
             unrelated components touched, duplicate declarations updated,
             parallel implementations introduced, extension points used)

The static term alone is gameable (a god file scores perfectly), so it is capped
at half the weight and the dynamic term reflects what real edits actually cost.
When no change runs are available (static-only ``analyze``), ``L`` falls back to
the concept-entropy term.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from esc.config import EscConfig, LocalityThresholds
from esc.model import PatchStats, TaskRunResult
from esc.repo import RepoScanner
from esc.scoring.concept_entropy import concept_entropy_report
from esc.scoring.static_quality import linear_decreasing


def patch_locality(stats: PatchStats, t: LocalityThresholds) -> float:
    """Locality of a single applied patch in ``[0, 1]``."""
    file_term = linear_decreasing(stats.files_changed, t.files_good, t.files_bad)
    module_term = linear_decreasing(stats.modules_changed, t.modules_good, t.modules_bad)
    spread = t.file_vs_module_split * file_term + (1.0 - t.file_vs_module_split) * module_term

    penalty = (
        t.penalty_unrelated * stats.unrelated_components_touched
        + t.penalty_duplicate * stats.duplicate_declarations_updated
        + (t.penalty_parallel_path if stats.introduced_parallel_path else 0.0)
        + (0.0 if stats.used_designated_extension_point else t.penalty_no_extension_point)
    )
    return max(0.0, min(1.0, spread - penalty))


def _static_localization(root: Path, config: EscConfig) -> tuple[float, list[dict]]:
    scanner = RepoScanner(root, config.scan_excludes, config.source_suffixes)
    files = scanner.all_files()
    report = concept_entropy_report(files, config.concepts, config.locality.granularity)
    present = [c for c in report if c.present]
    localization = (
        sum(c.localization for c in present) / len(present) if present else 1.0
    )
    rows = [
        {
            "concept": c.name,
            "sites": c.sites,
            "modules": c.modules,
            "entropy": round(c.entropy, 4),
            "localization": round(c.localization, 4),
            "top_modules": list(c.top_modules),
            "present": c.present,
        }
        for c in report
    ]
    return localization, rows


def score_locality(
    root: Path,
    config: EscConfig,
    change_runs: Sequence[TaskRunResult] = (),
) -> tuple[float, dict]:
    """Return ``(L, diagnostics)`` for the codebase and (optional) change runs."""
    static_term, concept_rows = _static_localization(root, config)

    successful = [r for r in change_runs if r.success]
    if not successful:
        diagnostics = {
            "mode": "static_only",
            "static_localization": static_term,
            "concepts": concept_rows,
        }
        return static_term, diagnostics

    t = config.locality
    per_run = [patch_locality(r.patch_stats, t) for r in successful]
    dynamic_term = sum(per_run) / len(per_run)
    l = t.static_weight * static_term + (1.0 - t.static_weight) * dynamic_term

    diagnostics = {
        "mode": "blended",
        "static_localization": static_term,
        "dynamic_patch_locality": dynamic_term,
        "static_weight": t.static_weight,
        "concepts": concept_rows,
        "per_run": [
            {"task": r.task_name, "seed": r.seed, "locality": loc}
            for r, loc in zip(successful, per_run)
        ],
    }
    return l, diagnostics
