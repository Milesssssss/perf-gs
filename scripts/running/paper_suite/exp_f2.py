# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Experiment F2 — orchestrate the gradient-fidelity study and aggregate it.

The measurement lives in ``scripts/running/paper_gradient_study.py``, one process
per scene, so it can neither share state with nor be mistaken for a training cell.
This module launches those processes, then turns their JSON into the raw and
summary CSVs and the figures.

Priority is on RAW DATA, per the brief: every individual observation is written
before any aggregate, so figures can be rebuilt later without touching the GPU.
Aggregates keep scene and checkpoint separate rather than collapsing everything
into one number — the whole question is whether the effect depends on training
stage, and averaging over stages would answer it by construction.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import config as cfg
from . import figures as figs
from . import tables

STUDY_SCRIPT = cfg.REPO / "scripts" / "running" / "paper_gradient_study.py"

# Scenes for the first pass, per the brief: not the full benchmark yet.
STUDY_SCENES: tuple[str, ...] = ("bicycle", "bonsai", "counter")
STUDY_STATES: tuple[int, ...] = (500, 1500, 3000)
STUDY_K: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

SINGLE_COLUMNS = (
    "scene", "checkpoint", "camera_id", "sampling_method", "pixel_ratio",
    "sample", "slice_index", "parameter_group",
    "relative_gradient_error", "cosine_similarity", "gradient_norm_ratio",
    "dense_norm", "sparse_norm", "n_gaussians", "int64_overflow",
)

ACCUM_COLUMNS = (
    "scene", "checkpoint", "camera_id", "sampling_method", "pixel_ratio",
    "sequence", "K", "parameter_group",
    "cumulative_relative_gradient_error", "cumulative_cosine_similarity",
    "gradient_norm_ratio", "n_gaussians", "int64_overflow",
)

SUMMARY_COLUMNS = (
    "scene", "checkpoint", "sampling_method", "parameter_group", "metric",
    "n", "mean", "std", "median", "min", "max",
)

ACCUM_SUMMARY_COLUMNS = (
    "scene", "checkpoint", "sampling_method", "parameter_group", "K", "metric",
    "n", "mean", "std", "median", "min", "max",
)

SANITY_COLUMNS = ("scene", "checkpoint", "camera_id", "check", "expect",
                  "relative_gradient_error", "cosine_similarity",
                  "gradient_norm_ratio", "rho", "passed", "note")

# Method presentation order and styling: the 2x2 of (spatial, temporal) plus IID.
METHOD_ORDER = ("iid_white", "spatial_blue", "temporal_only", "stbn")
METHOD_STYLE = {
    "iid_white": {"color": "#E69F00", "marker": "s", "linestyle": "--",
                  "label": "IID white"},
    "spatial_blue": {"color": "#56B4E9", "marker": "^", "linestyle": "-.",
                     "label": "Spatial-only blue"},
    "temporal_only": {"color": "#CC79A7", "marker": "v", "linestyle": ":",
                      "label": "Temporal-only"},
    "stbn": {"color": "#009E73", "marker": "D", "linestyle": "-",
             "label": "STBN (both)"},
}
GROUP_ORDER = ("global_scalar", "position", "scale", "rotation", "opacity",
               "color_sh0", "color_sh_rest")


def _style(method: str) -> dict:
    return METHOD_STYLE.get(method, {"color": "#777777", "marker": ".",
                                     "linestyle": "-", "label": method})


def run_studies(out: dict, *, python: str, scenes=STUDY_SCENES,
                states=STUDY_STATES, k_values=STUDY_K, cameras: int = 8,
                single_masks: int = 4, sequences: int = 2,
                fraction: float = cfg.PRIMARY_FRACTION, force: bool = False,
                timeout_s: float = 21600.0) -> dict:
    """One study process per scene, reusing any already-written result."""
    directory: Path = out["gradient_analysis"]
    directory.mkdir(parents=True, exist_ok=True)
    logs: Path = out["logs"]
    status: dict[str, dict] = {}

    for scene in scenes:
        target = directory / f"study_{scene}.json"
        if target.is_file() and not force:
            try:
                json.loads(target.read_text())
                print(f"[grad-study] cached {scene}", flush=True)
                status[scene] = {"status": "cached"}
                continue
            except ValueError:
                pass
        directory_path = cfg.scene_dir(scene)
        if directory_path is None:
            status[scene] = {"status": "missing_scene"}
            continue

        command = [
            python, str(STUDY_SCRIPT),
            "--scene", str(directory_path),
            "--state-iters", *[str(s) for s in states],
            "--k-values", *[str(k) for k in k_values],
            "--cameras", str(cameras),
            "--single-masks", str(single_masks),
            "--sequences", str(sequences),
            "--probe-fraction", f"{fraction:.4f}",
            # STBN compiled in, NVIDIA table available, GLOBAL thresholding.
            # The study clears the density schedule so the advance is dense.
            "--stbn-pixel-fraction", f"{fraction:.4f}",
            "--stbn-source", "nvidia",
            "--stbn-select-mode", cfg.SELECT_GLOBAL,
            "--iters", str(max(states)),
            "--seed", str(cfg.PRIMARY_SEED),
            "--sh-degree", "3",
            "--densify-mode", "structure_aware",
            "--mip-splatting",
            "--downsample", "4",
            "--no-viewer",
            "--out", str(target),
        ]
        stdout_path = logs / f"study_{scene}.stdout.log"
        stderr_path = logs / f"study_{scene}.stderr.log"
        print(f"[grad-study] RUN {scene}", flush=True)
        started = time.perf_counter()
        try:
            with stdout_path.open("w", encoding="utf-8", errors="replace") as o, \
                    stderr_path.open("w", encoding="utf-8", errors="replace") as e:
                proc = subprocess.run(command, cwd=str(cfg.REPO), stdout=o,
                                      stderr=e, timeout=timeout_s)
            ok = proc.returncode == 0 and target.is_file()
            status[scene] = {"status": "ok" if ok else "failed",
                            "return_code": proc.returncode,
                            "seconds": time.perf_counter() - started,
                            "stderr_path": str(stderr_path.relative_to(cfg.REPO))}
            if not ok:
                print(f"[grad-study] FAILED {scene} (rc={proc.returncode}); "
                      f"see {stderr_path}", flush=True)
        except (OSError, subprocess.SubprocessError) as error:
            status[scene] = {"status": "failed",
                            "error": f"{type(error).__name__}: {error}"}
            print(f"[grad-study] FAILED {scene}: {error}", flush=True)
    return status


# int64 is what the fixed-point gradient accumulators are, and on some views a
# single backward saturates them. A wrapped accumulator converts to a float64 of
# magnitude ~9.2e18, which turns every ratio built from it into garbage (norm
# ratios of 1e9 were what exposed this). Such an observation is a FAILED
# MEASUREMENT, not a bad method, so it is flagged, excluded from every statistic,
# and counted in the output rather than quietly averaged in.
INT64_MAX = 9.223372036854775807e18
OVERFLOW_NORM_LIMIT = 0.5 * INT64_MAX
# A relative error this large cannot be a real sparse estimate at rho=0.25; it is
# the accumulation-path symptom of the same saturation.
OVERFLOW_RATIO_LIMIT = 50.0


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _is_overflowed(row: dict) -> bool:
    """Whether this observation's gradient buffers saturated int64."""
    for key in ("dense_norm", "sparse_norm"):
        value = _finite(row.get(key))
        if value is not None and abs(value) > OVERFLOW_NORM_LIMIT:
            return True
    for key in ("relative_gradient_error", "gradient_norm_ratio",
                "cumulative_relative_gradient_error"):
        value = _finite(row.get(key))
        if value is not None and abs(value) > OVERFLOW_RATIO_LIMIT:
            return True
    return False


def _flag_overflow(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into (clean, overflowed), stamping every row with the flag."""
    clean, bad = [], []
    for row in rows:
        row["int64_overflow"] = _is_overflowed(row)
        (bad if row["int64_overflow"] else clean).append(row)
    return clean, bad


def _load(out: dict, scenes) -> list[dict]:
    payloads = []
    for scene in scenes:
        path = out["gradient_analysis"] / f"study_{scene}.json"
        if not path.is_file():
            continue
        try:
            payloads.append(json.loads(path.read_text()))
        except ValueError:
            continue
    return payloads


def _summarize(rows: list[dict], keys: tuple[str, ...],
               metrics: tuple[str, ...]) -> list[dict]:
    """mean / std / median / min / max / n, grouped by ``keys``.

    ``n`` is reported for every row because a mean of 4 observations and a mean of
    96 are not interchangeable, and the group sizes here differ by method.
    """
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    out: list[dict] = []
    for key, group in sorted(buckets.items(), key=lambda kv: [str(x) for x in kv[0]]):
        for metric in metrics:
            values = np.asarray([r[metric] for r in group
                                 if r.get(metric) is not None], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            out.append({
                **dict(zip(keys, key)),
                "metric": metric,
                "n": int(values.size),
                "mean": float(values.mean()),
                "std": (float(values.std(ddof=1)) if values.size > 1 else 0.0),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
            })
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _fig_by_method(single: list[dict], directory: Path, checkpoints) -> None:
    """Figures 1 and 2: error and cosine by method, one panel per checkpoint."""
    plt = figs.pyplot()
    if plt is None:
        return
    for metric, stem, ylabel in (
            ("relative_gradient_error", "expF2_error_by_method",
             "relative $L_2$ error vs dense"),
            ("cosine_similarity", "expF2_cosine_by_method",
             "cosine similarity to dense")):
        fig, axes = plt.subplots(1, len(checkpoints),
                                 figsize=(figs.FIG_WIDTH_DOUBLE, 2.4),
                                 squeeze=False, sharey=True)
        for column, checkpoint in enumerate(checkpoints):
            ax = axes[0][column]
            drew = False
            for index, method in enumerate(METHOD_ORDER):
                values = [r[metric] for r in single
                          if r["sampling_method"] == method
                          and r["checkpoint"] == checkpoint
                          and r["parameter_group"] == "global_scalar"
                          and r.get(metric) is not None]
                if not values:
                    continue
                drew = True
                style = _style(method)
                ax.bar([index], [float(np.mean(values))], 0.65,
                       yerr=[float(np.std(values, ddof=1)) if len(values) > 1 else 0.0],
                       capsize=2, color=style["color"], label=style["label"],
                       error_kw={"linewidth": 0.7})
                # Every observation as a dot: a mean without its spread would
                # hide that the methods overlap.
                jitter = np.random.default_rng(index).normal(0, 0.06, len(values))
                ax.scatter(np.full(len(values), index) + jitter, values, s=3,
                           color="#333333", alpha=0.35, zorder=3,
                           edgecolors="none")
            ax.set_xticks(range(len(METHOD_ORDER)))
            ax.set_xticklabels([_style(m)["label"] for m in METHOD_ORDER],
                               rotation=30, ha="right")
            ax.set_title(f"iter {checkpoint}", fontsize=8)
            if column == 0:
                ax.set_ylabel(ylabel)
            if not drew:
                figs.annotate_empty(ax)
        fig.suptitle(f"Experiment F2: single-mask gradient fidelity at "
                     f"$\\rho$=0.25 (all scenes, all cameras)", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        figs.save(fig, directory, stem)


def _fig_by_group(single: list[dict], directory: Path) -> None:
    """Figure 3: relative error per parameter group.

    Per-group is the point: one large group can dominate the global norm, so a
    single number would hide which parameters actually suffer.
    """
    plt = figs.pyplot()
    if plt is None:
        return
    groups = [g for g in GROUP_ORDER
              if any(r["parameter_group"] == g for r in single)]
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(figs.FIG_WIDTH_DOUBLE, 2.6))
    width = 0.8 / max(len(METHOD_ORDER), 1)
    x = np.arange(len(groups))
    for index, method in enumerate(METHOD_ORDER):
        means, errs = [], []
        for group in groups:
            values = [r["relative_gradient_error"] for r in single
                      if r["sampling_method"] == method
                      and r["parameter_group"] == group
                      and r.get("relative_gradient_error") is not None]
            means.append(float(np.mean(values)) if values else np.nan)
            errs.append(float(np.std(values, ddof=1))
                        if len(values) > 1 else 0.0)
        offset = (index - (len(METHOD_ORDER) - 1) / 2) * width
        style = _style(method)
        ax.bar(x + offset, means, width, yerr=errs, capsize=1.5,
               color=style["color"], label=style["label"],
               error_kw={"linewidth": 0.6})
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_ylabel("relative $L_2$ error vs dense")
    ax.legend(ncol=4, fontsize=6.5)
    ax.set_title("Single-mask gradient error by parameter group "
                 "(mean $\\pm$ s.d. over scenes, checkpoints, cameras, masks)",
                 fontsize=8, loc="left")
    fig.tight_layout()
    figs.save(fig, directory, "expF2_error_by_parameter_group")


def _fig_accumulation(accum: list[dict], directory: Path, checkpoints,
                      k_values) -> None:
    """Figures 4 and 5: cumulative error and cosine against K.

    The central hypothesis lives here — at equal total sample budget, does the
    accumulated STBN gradient approach dense faster than IID?
    """
    plt = figs.pyplot()
    if plt is None:
        return
    for metric, stem, ylabel, logy in (
            ("cumulative_relative_gradient_error", "expF2_cumulative_error_vs_K",
             "relative $L_2$ error of $\\bar g(K)$", True),
            ("cumulative_cosine_similarity", "expF2_cumulative_cosine_vs_K",
             "cosine similarity of $\\bar g(K)$", False)):
        fig, axes = plt.subplots(1, len(checkpoints),
                                 figsize=(figs.FIG_WIDTH_DOUBLE, 2.4),
                                 squeeze=False, sharey=True)
        for column, checkpoint in enumerate(checkpoints):
            ax = axes[0][column]
            drew = False
            for method in METHOD_ORDER:
                xs, ys, es = [], [], []
                for k in k_values:
                    values = [r[metric] for r in accum
                              if r["sampling_method"] == method
                              and r["checkpoint"] == checkpoint
                              and r["K"] == k
                              and r["parameter_group"] == "global_scalar"
                              and r.get(metric) is not None]
                    if not values:
                        continue
                    xs.append(k)
                    ys.append(float(np.mean(values)))
                    es.append(float(np.std(values, ddof=1))
                              if len(values) > 1 else 0.0)
                if not xs:
                    continue
                drew = True
                style = _style(method)
                ax.errorbar(xs, ys, yerr=es, color=style["color"],
                            marker=style["marker"], linestyle=style["linestyle"],
                            linewidth=1.1, markersize=3.5, capsize=1.5,
                            label=style["label"], elinewidth=0.6)
            # 1/sqrt(K) reference: what independent averaging alone would give.
            if drew and logy:
                base = [r[metric] for r in accum
                        if r["sampling_method"] == "iid_white"
                        and r["checkpoint"] == checkpoint and r["K"] == 1
                        and r["parameter_group"] == "global_scalar"
                        and r.get(metric) is not None]
                if base:
                    ref = float(np.mean(base))
                    ks = np.asarray(k_values, dtype=float)
                    ax.plot(ks, ref / np.sqrt(ks), color="#999999",
                            linestyle=":", linewidth=1.0,
                            label="$1/\\sqrt{K}$ (IID reference)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(k_values))
            ax.set_xticklabels([str(k) for k in k_values])
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("revisits $K$")
            ax.set_title(f"iter {checkpoint}", fontsize=8)
            if column == 0:
                ax.set_ylabel(ylabel)
                ax.legend(fontsize=6)
            if not drew:
                figs.annotate_empty(ax)
        fig.suptitle("Experiment F2: accumulated sparse gradient vs the dense "
                     "gradient, frozen state", fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        figs.save(fig, directory, stem)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(out: dict, *, python: str, scenes=STUDY_SCENES, states=STUDY_STATES,
        k_values=STUDY_K, cameras: int = 8, single_masks: int = 4,
        sequences: int = 2, fraction: float = cfg.PRIMARY_FRACTION,
        force: bool = False, skip_runs: bool = False) -> dict:
    directory: Path = out["gradient_analysis"]
    status = ({} if skip_runs else
              run_studies(out, python=python, scenes=scenes, states=states,
                          k_values=k_values, cameras=cameras,
                          single_masks=single_masks, sequences=sequences,
                          fraction=fraction, force=force))

    payloads = _load(out, scenes)
    if not payloads:
        return {"status": "no gradient studies completed", "run_status": status}

    single, accum, sanity, fingerprints = [], [], [], []
    for payload in payloads:
        single += payload.get("single_mask") or []
        accum += payload.get("accumulation") or []
        sanity += payload.get("sanity_checks") or []
        for fingerprint in payload.get("state_fingerprints") or []:
            fingerprints.append({"scene": payload.get("scene"), **fingerprint})

    # Raw CSVs keep EVERY observation, overflowed ones included and flagged, so
    # the exclusion is auditable rather than invisible.
    single_clean, single_bad = _flag_overflow(single)
    accum_clean, accum_bad = _flag_overflow(accum)
    tables.write_csv(directory / "study_single_mask_raw.csv", single,
                     SINGLE_COLUMNS)
    tables.write_csv(directory / "study_accumulation_raw.csv", accum,
                     ACCUM_COLUMNS)
    overflow = {
        "single_total": len(single), "single_excluded": len(single_bad),
        "accumulation_total": len(accum), "accumulation_excluded": len(accum_bad),
        "single_excluded_fraction": (len(single_bad) / len(single)
                                     if single else None),
        "accumulation_excluded_fraction": (len(accum_bad) / len(accum)
                                           if accum else None),
        "affected_cells": sorted({
            f"{r.get('scene')}@{r.get('checkpoint')}/cam{r.get('camera_id')}"
            for r in single_bad + accum_bad}),
        "cause": ("the int64 fixed-point gradient accumulator saturates on some "
                  "views; a wrapped accumulator reads back as ~9.2e18 and makes "
                  "every ratio built from it meaningless. These are failed "
                  "MEASUREMENTS, not bad methods, and are excluded from all "
                  "statistics."),
    }
    single, accum = single_clean, accum_clean
    tables.write_csv(directory / "study_sanity_checks.csv", sanity,
                     SANITY_COLUMNS)

    # Summaries keep scene and checkpoint separate on purpose.
    tables.write_csv(
        directory / "study_single_mask_summary.csv",
        _summarize(single, ("scene", "checkpoint", "sampling_method",
                           "parameter_group"),
                   ("relative_gradient_error", "cosine_similarity",
                    "gradient_norm_ratio")),
        SUMMARY_COLUMNS)
    tables.write_csv(
        directory / "study_accumulation_summary.csv",
        _summarize(accum, ("scene", "checkpoint", "sampling_method",
                           "parameter_group", "K"),
                   ("cumulative_relative_gradient_error",
                    "cumulative_cosine_similarity", "gradient_norm_ratio")),
        ACCUM_SUMMARY_COLUMNS)
    # Pooled over scenes, for the headline numbers only.
    pooled = _summarize(single, ("sampling_method", "parameter_group"),
                        ("relative_gradient_error", "cosine_similarity",
                         "gradient_norm_ratio"))
    tables.write_csv(directory / "study_pooled_summary.csv",
                     [{"scene": "ALL", "checkpoint": "ALL", **r} for r in pooled],
                     SUMMARY_COLUMNS)

    checkpoints = sorted({r["checkpoint"] for r in single})
    ks = sorted({r["K"] for r in accum})
    _fig_by_method(single, directory, checkpoints)
    _fig_by_group(single, directory)
    _fig_accumulation(accum, directory, checkpoints, ks)

    def pooled_metric(method: str, metric: str, group="global_scalar"):
        values = [v for v in (_finite(r.get(metric)) for r in single
                              if r["sampling_method"] == method
                              and r["parameter_group"] == group)
                  if v is not None]
        return (float(np.median(values)) if values else None,
                float(np.std(values, ddof=1)) if len(values) > 1 else None,
                len(values))

    def accum_metric(method: str, k: int, metric: str):
        # MEDIAN, not mean: even after excluding saturated observations the
        # distribution over cameras is heavy-tailed, and a mean would be steered
        # by whichever camera happened to have the noisiest single view.
        values = [v for v in (_finite(r.get(metric)) for r in accum
                              if r["sampling_method"] == method
                              and r["K"] == k
                              and r["parameter_group"] == "global_scalar")
                  if v is not None]
        return float(np.median(values)) if values else None

    headline = {}
    for method in METHOD_ORDER:
        err = pooled_metric(method, "relative_gradient_error")
        cos = pooled_metric(method, "cosine_similarity")
        ratio = pooled_metric(method, "gradient_norm_ratio")
        headline[method] = {
            "relative_error_median": err[0], "relative_error_std": err[1],
            "cosine_median": cos[0], "cosine_std": cos[1],
            "norm_ratio_median": ratio[0], "n_observations": err[2],
            "cumulative_error_by_K": {
                str(k): accum_metric(method, k,
                                     "cumulative_relative_gradient_error")
                for k in ks},
            "cumulative_cosine_by_K": {
                str(k): accum_metric(method, k, "cumulative_cosine_similarity")
                for k in ks},
        }

    all_frozen = all(f.get("state_unchanged") for f in fingerprints) if fingerprints else None
    sanity_ok = all(c.get("passed") for c in sanity if c.get("passed") is not None)
    return {
        "status": "ok",
        "run_status": status,
        "scenes": sorted({r["scene"] for r in single}),
        "checkpoints": checkpoints,
        "k_values": ks,
        "n_single_observations": len(single),
        "n_accumulation_observations": len(accum),
        "int64_overflow": overflow,
        "headline": headline,
        "sanity_checks": sanity,
        "sanity_all_passed": bool(sanity_ok),
        "state_unchanged_everywhere": all_frozen,
        "select_mode": payloads[0].get("select_mode"),
        "normalization": payloads[0].get("normalization"),
        "temporal_only_definition": payloads[0].get("temporal_only_definition"),
        "sh_rest_limitation": payloads[0].get("sh_rest_limitation"),
        "artifacts": {
            "single_raw_csv": "gradient_analysis/study_single_mask_raw.csv",
            "accumulation_raw_csv": "gradient_analysis/study_accumulation_raw.csv",
            "sanity_csv": "gradient_analysis/study_sanity_checks.csv",
            "single_summary_csv":
                "gradient_analysis/study_single_mask_summary.csv",
            "accumulation_summary_csv":
                "gradient_analysis/study_accumulation_summary.csv",
            "pooled_summary_csv": "gradient_analysis/study_pooled_summary.csv",
            "figures": [
                "gradient_analysis/expF2_error_by_method.pdf",
                "gradient_analysis/expF2_cosine_by_method.pdf",
                "gradient_analysis/expF2_error_by_parameter_group.pdf",
                "gradient_analysis/expF2_cumulative_error_vs_K.pdf",
                "gradient_analysis/expF2_cumulative_cosine_vs_K.pdf",
            ],
        },
    }
