# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Experiment D — where the time actually goes, per stage and per training phase.

Built from the dedicated stage-timing cells (four representative scenes, one per
dataset family, all four regimes). They are separate runs rather than
instrumentation on the main grid for a measured reason: reading GPU timestamps
requires a host drain the hot path never performs, and that drain costs a few
percent of ACTIVE training time. A few percent is small, but it is not nothing on
a run whose timing is itself a reported result — so the instrumented runs are
kept out of every speed table, and this experiment quotes them only for the
BREAKDOWN (proportions between stages), never for absolute speed.

The measured overhead is reported per cell so the reader can see the size of what
was avoided, rather than taking "negligible" on trust.

Two honest limits of the breakdown, both surfaced in the outputs:

* **There is no GPU mask-generation stage.** The rank table is built and uploaded
  once at startup; per-iteration mask work is a few microseconds of CPU
  arithmetic to pick a slice index. It is reported as a host cost, and it is
  ~0.01 ms/iteration — which is the finding, not a measurement gap.
* **Densification is a host-timed cost**, because it runs in its own submit
  outside ``train_step``'s marker tree and reads buffers back (so it is largely
  synchronous anyway). It is reported separately from the GPU stage sums, never
  folded in.

Marker-to-stage grouping lives in ``trainer.frontend.profiling.stage_timing`` and
the RAW per-marker sums are persisted per sample, so relabelling the bars for the
paper never requires re-running anything.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as cfg
from . import figures as figs
from . import stats
from . import tables
from .results import ResultSet

# Presentation order and labels for the stacked bars: roughly the order work
# happens in an iteration, so the stack reads top-to-bottom as a pipeline.
STAGE_ORDER: tuple[tuple[str, str], ...] = (
    ("preprocess_projection", "preprocess / projection"),
    ("tile_binning_sort", "tile binning / sort"),
    ("forward_raster", "forward raster"),
    ("loss", "loss"),
    ("backward_raster", "backward raster"),
    ("backward_projection_sh", "backward projection / SH"),
    ("optimizer_update", "optimizer update"),
    ("densify_accum", "densify accumulators"),
)
STAGE_COLORS = ("#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                "#937860", "#DA8BC3", "#8C8C8C")

BREAKDOWN_COLUMNS = (
    "scene", "dataset_family", "regime", "pixel_fraction", "window",
    "iter_from", "iter_to", "samples",
    *[f"gpu_{k}_ms" for k, _ in STAGE_ORDER],
    "gpu_stage_total_ms", "gpu_forward_parent_ms", "gpu_backward_parent_ms",
    "host_mask_sampling_ms", "host_densify_prune_ms_per_event",
    "unmapped_markers_ms", "sample_overhead_s", "train_active_s", "cell_key",
)

AGGREGATE_COLUMNS = (
    "regime", "window", "n_scenes",
    *[f"gpu_{k}_ms" for k, _ in STAGE_ORDER],
    "gpu_stage_total_ms", "host_mask_sampling_ms",
)

OVERHEAD_COLUMNS = (
    "scene", "regime", "sampled_iters", "total_iters", "sample_coverage",
    "sample_overhead_s", "train_active_s", "overhead_fraction_of_active",
    "mask_sampling_total_ms", "densify_prune_total_ms", "densify_events",
)


def _window_rows(results: ResultSet, fraction: float) -> list[dict]:
    """One row per (cell, window), from each timing cell's stage_timing.json."""
    rows: list[dict] = []
    for regime in cfg.REGIMES:
        use_fraction = 1.0 if regime == "full" else fraction
        for result in results.select(regime=regime, fraction=use_fraction,
                                     seed=cfg.PRIMARY_SEED, stage_timing=True):
            summary = result.stage_timing
            if not summary:
                continue
            windows = list(summary.get("windows") or [])
            whole = summary.get("whole_run")
            if whole:
                windows = windows + [whole]
            host_totals = summary.get("host_totals_ms") or {}
            host_counts = summary.get("host_counts") or {}
            densify_events = int(host_counts.get("densify_prune", 0) or 0)
            for window in windows:
                stage_ms = window.get("stages_ms") or {}
                parents = window.get("parent_markers_ms") or {}
                host = window.get("host_ms") or {}
                row: dict = {
                    "scene": result.cell.scene,
                    "dataset_family": result.cell.family,
                    "regime": regime,
                    "pixel_fraction": use_fraction,
                    "window": window.get("window"),
                    "iter_from": window.get("iter_from"),
                    "iter_to": window.get("iter_to"),
                    "samples": window.get("samples"),
                    "gpu_stage_total_ms": window.get("stage_total_ms"),
                    "gpu_forward_parent_ms": parents.get("forward"),
                    "gpu_backward_parent_ms": parents.get("backward"),
                    "host_mask_sampling_ms": host.get("mask_sampling"),
                    # Per EVENT, not per iteration: densify fires ~19 times in a
                    # 3k run, so a per-iteration average would misrepresent both
                    # its cost and its cadence.
                    "host_densify_prune_ms_per_event": (
                        (host_totals.get("densify_prune") / densify_events)
                        if densify_events else None),
                    "unmapped_markers_ms": round(sum(
                        (window.get("unmapped_markers_ms") or {}).values()), 6),
                    "sample_overhead_s": summary.get("sample_overhead_s"),
                    "train_active_s": result.metric("train_active_s"),
                    "cell_key": result.cell.key,
                }
                for key, _label in STAGE_ORDER:
                    row[f"gpu_{key}_ms"] = stage_ms.get(key)
                rows.append(row)
    return rows


def _aggregate(window_rows: list[dict]) -> list[dict]:
    """Mean stage profile per (regime, window) across the timing scenes."""
    out: list[dict] = []
    windows = []
    for row in window_rows:
        if row["window"] not in windows:
            windows.append(row["window"])
    for regime in cfg.REGIMES:
        for window in windows:
            picked = [r for r in window_rows
                      if r["regime"] == regime and r["window"] == window]
            if not picked:
                continue
            row: dict = {"regime": regime, "window": window,
                         "n_scenes": len({r["scene"] for r in picked})}
            for key, _label in STAGE_ORDER:
                row[f"gpu_{key}_ms"] = stats.mean_std(
                    [r.get(f"gpu_{key}_ms") for r in picked])["mean"]
            row["gpu_stage_total_ms"] = stats.mean_std(
                [r.get("gpu_stage_total_ms") for r in picked])["mean"]
            row["host_mask_sampling_ms"] = stats.mean_std(
                [r.get("host_mask_sampling_ms") for r in picked])["mean"]
            out.append(row)
    return out


def _overhead_rows(results: ResultSet, fraction: float) -> list[dict]:
    """The cost of the instrumentation itself, per cell."""
    rows: list[dict] = []
    for regime in cfg.REGIMES:
        use_fraction = 1.0 if regime == "full" else fraction
        for result in results.select(regime=regime, fraction=use_fraction,
                                     seed=cfg.PRIMARY_SEED, stage_timing=True):
            summary = result.stage_timing
            if not summary:
                continue
            active = result.metric("train_active_s")
            overhead = summary.get("sample_overhead_s")
            host_totals = summary.get("host_totals_ms") or {}
            host_counts = summary.get("host_counts") or {}
            rows.append({
                "scene": result.cell.scene,
                "regime": regime,
                "sampled_iters": summary.get("sampled_iters"),
                "total_iters": summary.get("total_iters"),
                "sample_coverage": summary.get("sample_coverage"),
                "sample_overhead_s": overhead,
                "train_active_s": active,
                "overhead_fraction_of_active": (
                    (overhead / active) if (overhead and active) else None),
                "mask_sampling_total_ms": host_totals.get("mask_sampling"),
                "densify_prune_total_ms": host_totals.get("densify_prune"),
                "densify_events": host_counts.get("densify_prune"),
            })
    return rows


def _figure_stacked(aggregate: list[dict], figures_dir: Path) -> None:
    """The paper's stacked runtime-breakdown figure: regimes x windows."""
    plt = figs.pyplot()
    if plt is None:
        return
    windows = [w for w in ("1-1000", "1001-2000", "2001-3000", "all")
               if any(r["window"] == w for r in aggregate)]
    if not windows:
        return
    fig, axes = plt.subplots(1, len(windows),
                             figsize=(figs.FIG_WIDTH_DOUBLE,
                                      2.6), squeeze=False, sharey=True)
    for column, window in enumerate(windows):
        ax = axes[0][column]
        regimes = [r for r in cfg.REGIMES
                   if any(x["regime"] == r and x["window"] == window
                          for x in aggregate)]
        positions = np.arange(len(regimes))
        bottoms = np.zeros(len(regimes))
        for (key, label), color in zip(STAGE_ORDER, STAGE_COLORS):
            values = []
            for regime in regimes:
                row = next(x for x in aggregate if x["regime"] == regime
                           and x["window"] == window)
                values.append(row.get(f"gpu_{key}_ms") or 0.0)
            ax.bar(positions, values, 0.62, bottom=bottoms, color=color,
                   label=(label if column == 0 else None), linewidth=0)
            bottoms += np.asarray(values, dtype=float)
        ax.set_xticks(positions)
        ax.set_xticklabels([figs.style_for(r)["label"] for r in regimes],
                           rotation=30, ha="right")
        ax.set_title(f"iters {window}", fontsize=8)
        if column == 0:
            ax.set_ylabel("GPU ms per iteration")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, -0.13), fontsize=6.5)
    fig.suptitle("Experiment D: per-stage GPU time per iteration "
                 f"(mean over {len(cfg.TIMING_SCENES)} scenes; sampled 1 in "
                 f"{cfg.STAGE_TIMING_EVERY} iterations)", fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    figs.save(fig, figures_dir, "expD_runtime_breakdown")


def _latex_breakdown(aggregate: list[dict], latex_dir: Path) -> None:
    whole = [r for r in aggregate if r["window"] == "all"]
    rows: list[list[str]] = []
    for regime in cfg.REGIMES:
        row = next((r for r in whole if r["regime"] == regime), None)
        if row is None:
            continue
        cells = [cfg.REGIME_LABELS[regime]]
        for key, _label in STAGE_ORDER:
            cells.append(tables.num(row.get(f"gpu_{key}_ms"), 3))
        cells.append(tables.num(row.get("gpu_stage_total_ms"), 3))
        rows.append(cells)
    header = ["Supervision"] + [
        label.replace("/", "/\\-").replace(" ", "~") for _k, label in STAGE_ORDER
    ] + ["Total"]
    body = tables.latex_table(
        caption=(rf"Per-stage GPU time per training iteration (ms), whole-run "
                 rf"mean over {len(cfg.TIMING_SCENES)} representative scenes "
                 rf"(\textsc{{bicycle}}, \textsc{{kitchen}}, \textsc{{train}}, "
                 rf"\textsc{{drjohnson}}), sampled on one iteration in "
                 rf"{cfg.STAGE_TIMING_EVERY}. Sparse supervision only shortens "
                 rf"the backward pixel loop, so the forward stages are expected "
                 rf"to be unchanged; that they are is a check on the "
                 rf"instrumentation as much as a result."),
        label="tab:stbn_runtime",
        column_spec="l" + "r" * (len(STAGE_ORDER) + 1),
        header=header, rows=rows, small=True,
        notes=("Mask selection is not a stage: the rank table is uploaded once "
               "at startup and indexed inline by the kernels, so per-iteration "
               "mask cost is host-side and below 0.01\\,ms. Densification runs "
               "in its own submit and is reported separately, per event, in "
               "\\texttt{tables/runtime\\_breakdown.csv}."),
        source="tables/runtime_breakdown_aggregate.csv")
    tables.write_latex(latex_dir / "table_runtime_breakdown.tex", body)


def run(results: ResultSet, out: dict, *,
        fraction: float = cfg.PRIMARY_FRACTION) -> dict:
    tables_dir, figures_dir = out["tables"], out["figures"]

    window_rows = _window_rows(results, fraction)
    if not window_rows:
        return {"status": "no stage-timing cells completed",
                "scenes": list(cfg.TIMING_SCENES)}

    tables.write_csv(tables_dir / "runtime_breakdown.csv", window_rows,
                     BREAKDOWN_COLUMNS)
    aggregate = _aggregate(window_rows)
    tables.write_csv(tables_dir / "runtime_breakdown_aggregate.csv", aggregate,
                     AGGREGATE_COLUMNS)
    overhead = _overhead_rows(results, fraction)
    tables.write_csv(tables_dir / "runtime_timing_overhead.csv", overhead,
                     OVERHEAD_COLUMNS)

    _figure_stacked(aggregate, figures_dir)
    _latex_breakdown(aggregate, latex_dir=out["latex"])

    whole = {r["regime"]: r for r in aggregate if r["window"] == "all"}
    fractions_of_total: dict[str, dict] = {}
    for regime, row in whole.items():
        total = row.get("gpu_stage_total_ms") or 0.0
        if total <= 0:
            continue
        fractions_of_total[regime] = {
            key: (row.get(f"gpu_{key}_ms") or 0.0) / total
            for key, _label in STAGE_ORDER}

    overhead_summary = stats.mean_std(
        [r["overhead_fraction_of_active"] for r in overhead])
    overhead_pct = (overhead_summary["mean"] or 0.0) * 100.0
    # Describe the measured cost rather than asserting it is small: at one sample
    # in ten the drain is a substantial fraction of active time, which is exactly
    # why these are dedicated cells. The proportions between stages remain valid
    # either way — a host drain adds host idle, not GPU execution time, and the
    # stage numbers come from GPU timestamps.
    verdict = (
        f"Detailed timing was restricted to {len(cfg.TIMING_SCENES)} "
        f"representative scenes as DEDICATED runs. The timestamp drain costs a "
        f"measured {overhead_pct:.1f}% of active training time at one sample in "
        f"{cfg.STAGE_TIMING_EVERY}"
        + (", which is far too much to place on a run whose wall-clock is a "
           "reported result." if overhead_pct >= 5.0 else
           ", small enough to be harmless, but kept off the main grid anyway so "
           "no reported timing depends on it.")
        + " The stage PROPORTIONS are unaffected by the drain: they come from GPU "
          "timestamps, and a host wait adds host idle rather than GPU execution "
          "time. These cells are therefore used for the breakdown only, and never "
          "for absolute speed.")
    return {
        "scenes": sorted({r["scene"] for r in window_rows}),
        "windows": sorted({r["window"] for r in window_rows}),
        "whole_run_by_regime": whole,
        "stage_share_of_total": fractions_of_total,
        "instrumentation_overhead_fraction_of_active": overhead_summary,
        "decision": verdict,
        "artifacts": {
            "breakdown_csv": "tables/runtime_breakdown.csv",
            "aggregate_csv": "tables/runtime_breakdown_aggregate.csv",
            "overhead_csv": "tables/runtime_timing_overhead.csv",
            "figures": ["figures/expD_runtime_breakdown.pdf"],
            "latex": ["latex/table_runtime_breakdown.tex"],
        },
    }
