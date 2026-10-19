# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Section 11: drive the frozen-state gradient study and aggregate it.

One subprocess per scene (a clean device, and a scene that crashes the driver costs
one scene), then aggregation into the tables and figures the report cites.

Three questions, and each maps onto one artifact:

  1. Does adaptive sampling improve K=1 gradient quality over uniform sampling?
     -> ``gradient_single_mask.csv``, the K=1 relative error and cosine per method.
  2. Does adaptive STBN still accumulate toward the dense gradient faster than
     adaptive IID and adaptive independent blue?
     -> ``gradient_accumulation.csv`` and ``gradient_vs_K`` figures.
  3. Does the Monte-Carlo average of the inverse-probability-weighted estimator
     converge to the dense gradient?
     -> the ``gradient_norm_ratio`` column: an unbiased estimator's accumulated
     mean has norm ratio -> 1 and relative error -> 0 as K grows. A BIASED one
     plateaus at a nonzero error with a norm ratio away from 1, which is exactly
     what would show up here if the 1/p_i weighting were wrong.

Two exclusions, both stated rather than applied silently:

  * Observations whose int64 fixed-point gradient accumulator SATURATED are
    flagged and dropped from every statistic. A wrapped accumulator reads back as
    ~9.2e18 and makes any ratio built from it meaningless — a failed MEASUREMENT,
    not a bad method. The count is reported.
  * ``scale`` and ``color_sh_rest`` are kept in the raw CSVs but excluded from the
    headline: the existing study's diagnostics on them are the ones most affected
    by that saturation, and SH-rest additionally cannot ride the accumulation path
    at all (its buffer is overwritten per backward).
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from paper_suite import figures as figs
from paper_suite import tables

from . import config as cfg

# Same thresholds the existing study uses, so an exclusion here means the same
# thing it means there.
INT64_MAX = 9.223372036854775807e18
OVERFLOW_NORM_LIMIT = 0.5 * INT64_MAX
OVERFLOW_RATIO_LIMIT = 50.0

METHOD_ORDER = ("uniform_iid", "uniform_stbn", "adaptive_iid",
                "adaptive_indep_blue", "adaptive_stbn")
METHOD_LABELS = {
    "uniform_iid": "Uniform IID",
    "uniform_stbn": "Uniform STBN",
    "adaptive_iid": "Adaptive IID",
    "adaptive_indep_blue": "Adaptive indep. blue",
    "adaptive_stbn": "Adaptive STBN",
}
METHOD_STYLE = {
    "uniform_iid": {"color": "#E69F00", "marker": "s", "linestyle": "--"},
    "uniform_stbn": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "adaptive_iid": {"color": "#D55E00", "marker": "v", "linestyle": ":"},
    "adaptive_indep_blue": {"color": "#56B4E9", "marker": "^", "linestyle": "-."},
    "adaptive_stbn": {"color": "#009E73", "marker": "D", "linestyle": "-"},
}

SINGLE_COLUMNS = (
    "scene", "checkpoint", "camera_id", "sampling_method", "parameter_group",
    "pixel_ratio", "sample", "slice_index", "relative_gradient_error",
    "cosine_similarity", "gradient_norm_ratio", "dense_norm", "sparse_norm",
    "n_gaussians", "int64_overflow",
)
ACCUM_COLUMNS = (
    "scene", "checkpoint", "camera_id", "sampling_method", "parameter_group",
    "pixel_ratio", "sequence", "K", "cumulative_relative_gradient_error",
    "cumulative_cosine_similarity", "gradient_norm_ratio", "n_gaussians",
    "int64_overflow",
)
SUMMARY_COLUMNS = (
    "sampling_method", "parameter_group", "metric", "n", "mean", "std", "median",
    "min", "max",
)
K_COLUMNS = (
    "sampling_method", "parameter_group", "K", "n",
    "cumulative_relative_gradient_error_mean",
    "cumulative_relative_gradient_error_std",
    "cumulative_cosine_similarity_mean", "gradient_norm_ratio_mean",
)
PROBABILITY_COLUMNS = (
    "scene", "checkpoint", "camera_id", "n_gaussians", "p_mean", "p_min", "p_max",
    "p_p99", "saturated_fraction", "dynamic_range", "revisit_max_abs_change",
    "measured_p_mean", "measured_selected_fraction", "measured_budget_error_max",
)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _study_command(python: str, scene: str, out_path: Path, *,
                   cameras: int) -> list[str]:
    directory = cfg.scene_dir(scene)
    if directory is None:
        raise FileNotFoundError(f"scene {scene!r} not found")
    return [
        python, str(cfg.REPO / "scripts" / "running" / "adaptive_stbn"
                    / "gradient_study.py"),
        "--scene", str(directory),
        "--downsample", str(cfg.scene_downsample(scene)),
        "--seed", str(cfg.PRIMARY_SEED),
        "--iters", str(max(cfg.GRADIENT_STATES)),
        "--sh-degree", "3",
        "--densify-mode", "structure_aware",
        "--mip-splatting",
        "--eval", "1",
        "--eval-holdout-stride", "8",
        "--no-viewer",
        # A sparse fraction is required so STBN_ENABLED is injected and the
        # compacted backward path is the one under measurement. The density itself
        # is applied per mask, and the state is advanced DENSELY (the study clears
        # the loop's density schedule).
        "--stbn-pixel-fraction", f"{cfg.PRIMARY_FRACTION:.4f}",
        "--stbn-select-mode", "global",
        "--stbn-source", "nvidia",
        "--state-iters", *[str(s) for s in cfg.GRADIENT_STATES],
        "--k-values", *[str(k) for k in cfg.GRADIENT_K_VALUES],
        "--cameras", str(cameras),
        "--sequences", str(cfg.GRADIENT_SEQUENCES),
        "--single-masks", str(cfg.GRADIENT_SINGLE_MASKS),
        *(("--measure-sh-rest",) if cfg.GRADIENT_MEASURE_SH_REST else ()),
        "--probe-fraction", f"{cfg.PRIMARY_FRACTION:.4f}",
        "--probe-alpha", f"{cfg.PRIMARY_ALPHA:.4f}",
        "--probe-sigma", f"{cfg.PRIMARY_SIGMA:.4f}",
        "--probe-importance-downsample", str(cfg.PRIMARY_DOWNSAMPLE),
        "--out", str(out_path),
        "--run-name", f"adaptive_grad_probe_{scene}",
    ]


def run(paths: dict, *, python: str, scenes: tuple[str, ...], cameras: int,
        logs_dir: Path, force: bool = False,
        timeout_s: float = 14400.0) -> dict:
    """Run the study per scene (skipping any already on disk), then aggregate."""
    out_dir = paths["gradient_validation"]
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    status: dict[str, dict] = {}
    for scene in scenes:
        target = out_dir / f"study_{scene}.json"
        if target.is_file() and not force:
            print(f"[grad] cached {scene}: {target.name}", flush=True)
            status[scene] = {"status": "cached", "path": target.name}
            continue
        if cfg.scene_dir(scene) is None:
            print(f"[grad] SKIP {scene}: dataset not present", flush=True)
            status[scene] = {"status": "missing_scene"}
            continue
        argv = _study_command(python, scene, target, cameras=cameras)
        stdout = logs_dir / f"grad_{scene}.stdout.log"
        stderr = logs_dir / f"grad_{scene}.stderr.log"
        started = time.perf_counter()
        print(f"[grad] RUN {scene}", flush=True)
        try:
            with stdout.open("w", encoding="utf-8", errors="replace") as out, \
                    stderr.open("w", encoding="utf-8", errors="replace") as err:
                proc = subprocess.run(argv, cwd=str(cfg.REPO), stdout=out,
                                      stderr=err, timeout=timeout_s)
            code = proc.returncode
        except (subprocess.TimeoutExpired, OSError,
                subprocess.SubprocessError) as error:
            code = None
            print(f"[grad] FAILED {scene}: {type(error).__name__}: {error}",
                  flush=True)
        wall = time.perf_counter() - started
        ok = code == 0 and target.is_file()
        status[scene] = {"status": "ok" if ok else "failed",
                         "return_code": code, "wall_s": round(wall, 1),
                         "stderr": str(stderr.relative_to(cfg.REPO))}
        print(f"[grad] {scene}: {'ok' if ok else 'FAILED'} in {wall / 60:.1f} min",
              flush=True)

    return {**aggregate(paths, scenes), "runs": status}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _is_overflowed(row: dict) -> bool:
    """Whether this observation's int64 gradient accumulator saturated."""
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


def _cell(row: dict) -> tuple:
    return (row.get("scene"), row.get("checkpoint"), row.get("camera_id"),
            row.get("parameter_group"))


def _saturated_cells(single: list[dict]) -> set[tuple]:
    """The ``(scene, checkpoint, camera, group)`` cells whose DENSE reference
    saturated int64.

    This is why the flagging cannot be per row. Only the single-mask rows carry
    ``dense_norm``; the accumulation rows carry only ratios built from it. When the
    dense reference has wrapped, those ratios come out looking *plausible* — a norm
    ratio of 0.54 and an error of 0.7 rather than anything above the 50x tripwire —
    so a per-row filter passes them through and a failed measurement becomes a
    "result". Measured on bicycle: the accumulation curve appeared to get WORSE
    with K, which is impossible for an unbiased estimator and was entirely this.

    The verdict is per (cell, GROUP), not per cell, because the saturation is
    group-local: at bicycle iteration 3000 the ``scale`` reference reads 1.6e19
    while position, rotation, opacity and SH0 sit at 1e8-1e10 and are perfectly
    usable. Excluding the whole checkpoint would throw away good measurements.
    """
    return {_cell(row) for row in single if _is_overflowed(row)}


def _flag(rows: list[dict], saturated: set[tuple]) -> tuple[list[dict],
                                                            list[dict]]:
    """Split into (clean, excluded), stamping every row with the flag."""
    clean, bad = [], []
    for row in rows:
        row["int64_overflow"] = (_is_overflowed(row) or _cell(row) in saturated)
        (bad if row["int64_overflow"] else clean).append(row)
    return clean, bad


# ---------------------------------------------------------------------------
# Pooling over the reported groups
# ---------------------------------------------------------------------------
#
# ``global_scalar`` (emitted by the shared study machinery) pools ALL FOURTEEN
# scalar gradients, which means it pools ``scale`` — and ``scale`` is the group
# whose accumulator saturates. One wrapped group is enough to destroy the pooled
# number while every other group is healthy, so the headline is pooled over the
# REPORTED groups instead.
#
# Reconstructed rather than re-measured. The three metrics per group are built from
# three scalars — dot(a,b), |a|^2, |b|^2 — and for a known accumulation scale s
# those are recoverable from the reported metrics:
#
#     |b|      = dense_norm                       (reported, or taken from the
#                                                  single-mask row of the same cell:
#                                                  the dense reference is identical)
#     |a|      = norm_ratio * s * |b|
#     dot      = cosine * |a| * |b|
#
# and dot/|a|^2/|b|^2 are ADDITIVE over a partition of the parameters, so summing
# the four groups and recomputing gives exactly what a second pass over the buffers
# would have. ``_verify_pooling`` checks that by reconstructing ``global_scalar``
# from all five groups and comparing against the value the study actually reported.

POOLED_GROUP = "reported_scalar"


def _dense_norms(single: list[dict]) -> dict[tuple, float]:
    """``(scene, checkpoint, camera, group) -> |dense|``, from the single-mask rows.

    The dense reference is computed once per (state, camera) and reused by every
    method and every K, so one lookup serves the accumulation rows too.
    """
    out: dict[tuple, float] = {}
    for row in single:
        value = _finite(row.get("dense_norm"))
        if value is not None:
            out.setdefault(_cell(row), value)
    return out


def _scalars(row: dict, dense_norm: float, scale: float,
             ratio_key: str, cosine_key: str) -> tuple[float, float, float] | None:
    ratio = _finite(row.get(ratio_key))
    cosine = _finite(row.get(cosine_key))
    if ratio is None or cosine is None or dense_norm <= 0.0:
        return None
    norm_a = ratio * scale * dense_norm
    return (cosine * norm_a * dense_norm, norm_a * norm_a, dense_norm * dense_norm)


def _pool(rows: list[dict], groups: tuple[str, ...], *, key_fields: tuple[str, ...],
          scale_field: str | None, ratio_key: str, cosine_key: str,
          error_key: str, dense: dict[tuple, float],
          label: str) -> list[dict]:
    """One pooled row per observation, over exactly ``groups``.

    An observation contributes only when EVERY requested group is present and
    clean for it — a partial pool would silently change what the metric is
    averaging over from one row to the next.
    """
    buckets: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        if row["parameter_group"] not in groups:
            continue
        identity = tuple(row.get(f) for f in key_fields)
        buckets.setdefault(identity, {})[row["parameter_group"]] = row
    out: list[dict] = []
    for identity, found in buckets.items():
        if len(found) != len(groups):
            continue
        total = [0.0, 0.0, 0.0]
        template = next(iter(found.values()))
        scale = 1.0 if scale_field is None else float(template.get(scale_field) or 1)
        complete = True
        for group in groups:
            row = found[group]
            norm_b = dense.get((row.get("scene"), row.get("checkpoint"),
                                row.get("camera_id"), group))
            parts = (None if norm_b is None
                     else _scalars(row, norm_b, scale, ratio_key, cosine_key))
            if parts is None:
                complete = False
                break
            total = [t + p for t, p in zip(total, parts)]
        if not complete:
            continue
        dot, aa, bb = total
        if bb <= 0.0 or aa <= 0.0:
            continue
        norm_b = np.sqrt(bb)
        diff_sq = max(aa / (scale * scale) - 2.0 * dot / scale + bb, 0.0)
        pooled = {**dict(zip(key_fields, identity)),
                  "parameter_group": label,
                  "int64_overflow": False,
                  error_key: float(np.sqrt(diff_sq) / norm_b),
                  cosine_key: float(dot / (np.sqrt(aa) * norm_b)),
                  ratio_key: float(np.sqrt(aa) / scale / norm_b)}
        if scale_field is None:
            pooled["dense_norm"] = float(norm_b)
            pooled["sparse_norm"] = float(np.sqrt(aa) / scale)
        out.append(pooled)
    return out


SINGLE_KEYS = ("scene", "checkpoint", "camera_id", "sampling_method",
               "pixel_ratio", "sample", "slice_index")
ACCUM_KEYS = ("scene", "checkpoint", "camera_id", "sampling_method",
              "pixel_ratio", "sequence", "K", "n_gaussians")


def _verify_pooling(single: list[dict], dense: dict[tuple, float]) -> dict:
    """Reconstruct ``global_scalar`` from all five groups and compare.

    The pooled headline is derived arithmetic, not a measurement, so it needs a
    check that the derivation is right. ``global_scalar`` is the one pooled value
    the study reports directly, so reproducing it from the per-group rows validates
    the whole reconstruction — including the dense-norm lookup the accumulation
    rows depend on.
    """
    from paper_gradient_study import SCALAR_GROUPS

    reconstructed = _pool(single, tuple(SCALAR_GROUPS), key_fields=SINGLE_KEYS,
                          scale_field=None, ratio_key="gradient_norm_ratio",
                          cosine_key="cosine_similarity",
                          error_key="relative_gradient_error", dense=dense,
                          label="_check")
    reported = {tuple(r.get(f) for f in SINGLE_KEYS): r for r in single
                if r["parameter_group"] == "global_scalar"}
    errors = []
    for row in reconstructed:
        target = reported.get(tuple(row.get(f) for f in SINGLE_KEYS))
        if target is None:
            continue
        for metric in ("relative_gradient_error", "cosine_similarity",
                       "gradient_norm_ratio"):
            a, b = _finite(row.get(metric)), _finite(target.get(metric))
            if a is None or b is None or abs(b) > OVERFLOW_RATIO_LIMIT:
                continue
            errors.append(abs(a - b) / max(abs(b), 1e-12))
    return {
        "n_compared": len(errors),
        "max_relative_disagreement": (float(max(errors)) if errors else None),
        "passed": bool(errors) and max(errors) < 1e-4,
        "what": ("global_scalar reconstructed from the five per-group rows vs the "
                 "value the study reported directly; validates the pooling "
                 "arithmetic used for " + POOLED_GROUP),
    }


def _summarize(rows: list[dict], keys: tuple[str, ...],
               metrics: tuple[str, ...]) -> list[dict]:
    """mean / std / median / min / max / n, grouped by ``keys``.

    ``n`` is reported on every row because a mean of 4 observations and a mean of
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
            out.append({**dict(zip(keys, key)), "metric": metric,
                        "n": int(values.size), "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                        "median": float(np.median(values)),
                        "min": float(values.min()), "max": float(values.max())})
    return out


def aggregate(paths: dict, scenes: tuple[str, ...]) -> dict:
    out_dir = paths["gradient_validation"]
    payloads = []
    for scene in scenes:
        path = out_dir / f"study_{scene}.json"
        if not path.is_file():
            continue
        try:
            payloads.append(json.loads(path.read_text()))
        except ValueError:
            print(f"[grad] {path.name} is unreadable; skipped", flush=True)
    if not payloads:
        return {"status": "missing", "reason": "no per-scene study payloads found"}

    single = [r for p in payloads for r in p.get("single_mask", [])]
    accum = [r for p in payloads for r in p.get("accumulation", [])]
    probability = [r for p in payloads for r in p.get("probability_maps", [])]
    sanity = [r for p in payloads for r in p.get("sanity_checks", [])]

    saturated = _saturated_cells(single)
    dense = _dense_norms([r for r in single if not _is_overflowed(r)])
    verification = _verify_pooling(single, dense)
    print(f"[grad] pooling check: {verification}", flush=True)

    single_clean, single_bad = _flag(single, saturated)
    accum_clean, accum_bad = _flag(accum, saturated)

    # The headline group: pooled over the four REPORTED groups only, so one
    # saturated group cannot destroy it the way it destroys global_scalar.
    single_clean += _pool(single_clean, cfg.GRADIENT_GROUPS,
                          key_fields=SINGLE_KEYS, scale_field=None,
                          ratio_key="gradient_norm_ratio",
                          cosine_key="cosine_similarity",
                          error_key="relative_gradient_error", dense=dense,
                          label=POOLED_GROUP)
    accum_clean += _pool(accum_clean, cfg.GRADIENT_GROUPS, key_fields=ACCUM_KEYS,
                         scale_field="K", ratio_key="gradient_norm_ratio",
                         cosine_key="cumulative_cosine_similarity",
                         error_key="cumulative_relative_gradient_error",
                         dense=dense, label=POOLED_GROUP)

    # Raw CSVs keep EVERY measured observation, flagged, so an exclusion is
    # auditable; the pooled rows are derived and go only into the summaries.
    tables.write_csv(out_dir / "gradient_single_mask.csv", single, SINGLE_COLUMNS)
    tables.write_csv(out_dir / "gradient_accumulation.csv", accum, ACCUM_COLUMNS)
    tables.write_csv(out_dir / "gradient_probability_maps.csv", probability,
                     PROBABILITY_COLUMNS)

    reported = tuple(cfg.GRADIENT_GROUPS) + (POOLED_GROUP,)
    summary = _summarize([r for r in single_clean
                          if r["parameter_group"] in reported],
                         ("sampling_method", "parameter_group"),
                         ("relative_gradient_error", "cosine_similarity",
                          "gradient_norm_ratio"))
    tables.write_csv(out_dir / "gradient_single_summary.csv", summary,
                     SUMMARY_COLUMNS)

    k_rows = _k_table([r for r in accum_clean
                       if r["parameter_group"] in reported])
    tables.write_csv(out_dir / "gradient_by_K.csv", k_rows, K_COLUMNS)

    written = _figures(paths, single_clean, k_rows)

    measured_bad = [r for r in single_bad + accum_bad]
    overflow = {
        "single_total": len(single), "single_excluded": len(single_bad),
        "accumulation_total": len(accum), "accumulation_excluded": len(accum_bad),
        "single_excluded_fraction": (len(single_bad) / len(single)
                                     if single else 0.0),
        "accumulation_excluded_fraction": (len(accum_bad) / len(accum)
                                           if accum else 0.0),
        "saturated_groups": sorted({c[3] for c in saturated}),
        "affected_cells": sorted({f"{r['scene']}@{r['checkpoint']}/cam"
                                  f"{r['camera_id']}/{r['parameter_group']}"
                                  for r in measured_bad}),
        "cause": ("the int64 fixed-point gradient accumulator saturates on some "
                  "views; a wrapped accumulator reads back as ~9.2e18 and makes "
                  "every ratio built from it meaningless. Failed MEASUREMENTS, "
                  "not bad methods, and excluded from all statistics."),
        "propagation": (
            "flagged per (scene, checkpoint, camera, GROUP). Per-group because the "
            "saturation is group-local — at bicycle iteration 3000 the `scale` "
            "reference reads 1.6e19 while position/rotation/opacity/SH0 sit at "
            "1e8-1e10 and are usable. Propagated from the single-mask rows to the "
            "accumulation rows of the same cell because only the former carry "
            "`dense_norm`: with a wrapped dense reference the accumulation ratios "
            "come out PLAUSIBLE rather than absurd, so a per-row tripwire passes "
            "them through."),
        "pooling_verification": verification,
    }

    record = {
        "status": "ok",
        "scenes": [p["scene"] for p in payloads],
        "checkpoints": sorted({r["checkpoint"] for r in single}),
        "k_values": sorted({int(r["K"]) for r in accum}) if accum else [],
        "rho_enforced": payloads[0].get("probe_fraction_enforced"),
        "alpha": payloads[0].get("alpha"),
        "sigma": payloads[0].get("sigma"),
        "groups_reported": list(cfg.GRADIENT_GROUPS),
        "headline_group": POOLED_GROUP,
        "headline_group_definition": (
            f"pooled over {list(cfg.GRADIENT_GROUPS)} only. NOT global_scalar, "
            f"which pools all fourteen scalar gradients and therefore pools "
            f"`scale` — the one group whose accumulator saturates. Derived by "
            f"summing the per-group dot/|a|^2/|b|^2 (additive over a partition) "
            f"and recomputing; the derivation is checked by reconstructing "
            f"global_scalar itself, see int64_overflow.pooling_verification."),
        "groups_excluded_from_headline": ["scale", "color_sh_rest"],
        "groups_excluded_reason": (
            "the int64 accumulator saturation documented in overflow.cause hits "
            "these hardest, and color_sh_rest additionally cannot ride the "
            "K-accumulation path (its buffer is overwritten per backward). Kept "
            "in the raw CSVs, excluded from every headline number."),
        "n_single_observations": len(single_clean),
        "n_accumulation_observations": len(accum_clean),
        "int64_overflow": overflow,
        "sanity_checks": sanity,
        "headline": _headline(single_clean, k_rows),
        "probability_maps": _probability_headline(probability),
        "outputs": {
            "single_csv": "gradient_validation/gradient_single_mask.csv",
            "accumulation_csv": "gradient_validation/gradient_accumulation.csv",
            "single_summary_csv": "gradient_validation/gradient_single_summary.csv",
            "by_K_csv": "gradient_validation/gradient_by_K.csv",
            "probability_csv": "gradient_validation/gradient_probability_maps.csv",
            "figures": [str(p.relative_to(paths["root"])) for p in written],
        },
    }
    (out_dir / "gradient_validation.json").write_text(json.dumps(record, indent=2))
    return record


def _k_table(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        buckets.setdefault((row["sampling_method"], row["parameter_group"],
                            int(row["K"])), []).append(row)
    out: list[dict] = []
    for (method, group, k), items in sorted(
            buckets.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        def column(name: str) -> np.ndarray:
            values = np.asarray([r[name] for r in items
                                 if r.get(name) is not None], dtype=float)
            return values[np.isfinite(values)]
        error = column("cumulative_relative_gradient_error")
        cosine = column("cumulative_cosine_similarity")
        ratio = column("gradient_norm_ratio")
        if error.size == 0:
            continue
        out.append({
            "sampling_method": method, "parameter_group": group, "K": k,
            "n": int(error.size),
            "cumulative_relative_gradient_error_mean": float(error.mean()),
            "cumulative_relative_gradient_error_std": (
                float(error.std(ddof=1)) if error.size > 1 else 0.0),
            "cumulative_cosine_similarity_mean": (float(cosine.mean())
                                                  if cosine.size else None),
            "gradient_norm_ratio_mean": (float(ratio.mean())
                                         if ratio.size else None),
        })
    return out


def _headline(single: list[dict], k_rows: list[dict]) -> dict:
    """Per method: K=1 quality, and the K-curve on the pooled scalar gradient."""
    out: dict[str, dict] = {}
    for method in METHOD_ORDER:
        rows = [r for r in single if r["sampling_method"] == method
                and r["parameter_group"] == POOLED_GROUP]
        errors = np.asarray([r["relative_gradient_error"] for r in rows
                             if r["relative_gradient_error"] is not None])
        cosines = np.asarray([r["cosine_similarity"] for r in rows
                              if r["cosine_similarity"] is not None])
        ratios = np.asarray([r["gradient_norm_ratio"] for r in rows
                             if r["gradient_norm_ratio"] is not None])
        curve = {int(r["K"]): r for r in k_rows
                 if r["sampling_method"] == method
                 and r["parameter_group"] == POOLED_GROUP}
        out[method] = {
            "n_observations": int(errors.size),
            "relative_error_median": (float(np.median(errors))
                                      if errors.size else None),
            "relative_error_mean": float(errors.mean()) if errors.size else None,
            "relative_error_std": (float(errors.std(ddof=1))
                                   if errors.size > 1 else None),
            "cosine_median": float(np.median(cosines)) if cosines.size else None,
            "norm_ratio_median": float(np.median(ratios)) if ratios.size else None,
            "cumulative_error_by_K": {
                str(k): row["cumulative_relative_gradient_error_mean"]
                for k, row in sorted(curve.items())},
            "cumulative_cosine_by_K": {
                str(k): row["cumulative_cosine_similarity_mean"]
                for k, row in sorted(curve.items())},
            "norm_ratio_by_K": {str(k): row["gradient_norm_ratio_mean"]
                                for k, row in sorted(curve.items())},
        }
    # Per-group K=1 comparison, which is where an allocation effect should show up
    # unevenly (position and opacity are driven by different pixels than SH0).
    per_group: dict[str, dict] = {}
    for group in cfg.GRADIENT_GROUPS:
        per_group[group] = {}
        for method in METHOD_ORDER:
            values = np.asarray([r["relative_gradient_error"] for r in single
                                 if r["sampling_method"] == method
                                 and r["parameter_group"] == group
                                 and r["relative_gradient_error"] is not None])
            per_group[group][method] = (float(np.median(values))
                                        if values.size else None)
    out["_per_group_k1_relative_error_median"] = per_group
    return out


def _probability_headline(rows: list[dict]) -> dict:
    if not rows:
        return {}
    def column(name: str) -> np.ndarray:
        return np.asarray([r[name] for r in rows if r.get(name) is not None],
                          dtype=float)
    drift = column("revisit_max_abs_change")
    return {
        "n_cameras": len(rows),
        "p_mean_mean": float(column("p_mean").mean()),
        "p_min_min": float(column("p_min").min()),
        "p_max_max": float(column("p_max").max()),
        "saturated_fraction_mean": float(column("saturated_fraction").mean()),
        "dynamic_range_median": float(np.median(column("dynamic_range"))),
        "budget_error_max": float(column("measured_budget_error_max").max())
        if column("measured_budget_error_max").size else None,
        "revisit_max_abs_change_max": float(drift.max()) if drift.size else None,
        "revisit_note": (
            "measured at a FROZEN state, so the map cannot legitimately change "
            "between repeat visits of one camera; a nonzero value would mean it "
            "depends on something it should not. This is the floor that design "
            "section 15's optional per-camera EMA would be justified against — "
            "the EMA stays disabled and unimplemented unless a run at a MOVING "
            "state shows material drift."),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _figures(paths: dict, single: list[dict], k_rows: list[dict]) -> list[Path]:
    plt = figs.pyplot()
    if plt is None:
        return []
    out = paths["figures"]
    written: list[Path] = []

    # 1. Cumulative relative error and cosine against K, pooled scalar gradient.
    #    Log-log on the error: an unbiased estimator averaged over K independent
    #    draws falls as K^-1/2, so a straight line of that slope is the signature
    #    of unbiasedness and a PLATEAU is the signature of bias.
    fig, axes = plt.subplots(1, 2, figsize=(figs.FIG_WIDTH_DOUBLE, 2.6))
    for method in METHOD_ORDER:
        curve = sorted([r for r in k_rows if r["sampling_method"] == method
                        and r["parameter_group"] == POOLED_GROUP],
                       key=lambda r: r["K"])
        if not curve:
            continue
        ks = [r["K"] for r in curve]
        style = METHOD_STYLE[method]
        axes[0].plot(ks, [r["cumulative_relative_gradient_error_mean"]
                          for r in curve], label=METHOD_LABELS[method],
                     linewidth=1.0, markersize=3, **style)
        axes[1].plot(ks, [r["cumulative_cosine_similarity_mean"] for r in curve],
                     label=METHOD_LABELS[method], linewidth=1.0, markersize=3,
                     **style)
    reference_ks = sorted({r["K"] for r in k_rows}) or [1]
    first = [r for r in k_rows if r["parameter_group"] == POOLED_GROUP
             and r["K"] == min(reference_ks)]
    if first:
        base = float(np.mean([r["cumulative_relative_gradient_error_mean"]
                              for r in first]))
        axes[0].plot(reference_ks,
                     [base * (min(reference_ks) / k) ** 0.5 for k in reference_ks],
                     color="#999999", linewidth=0.8, linestyle=":",
                     label=r"$K^{-1/2}$")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("accumulated masks $K$")
    axes[0].set_ylabel("relative gradient error")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("accumulated masks $K$")
    axes[1].set_ylabel("cosine similarity")
    figs.legend_below(fig, axes[0], ncol=3)
    written += figs.save(fig, out, "gradient_vs_K")

    # 2. K=1 relative error by parameter group — where allocation helps unevenly.
    groups = list(cfg.GRADIENT_GROUPS)
    fig, ax = plt.subplots(figsize=(figs.FIG_WIDTH_DOUBLE, 2.4))
    width = 0.8 / max(len(METHOD_ORDER), 1)
    for index, method in enumerate(METHOD_ORDER):
        values = []
        for group in groups:
            picked = np.asarray([r["relative_gradient_error"] for r in single
                                 if r["sampling_method"] == method
                                 and r["parameter_group"] == group
                                 and r["relative_gradient_error"] is not None])
            values.append(float(np.median(picked)) if picked.size else np.nan)
        offsets = np.arange(len(groups)) + (index - len(METHOD_ORDER) / 2) * width
        ax.bar(offsets, values, width=width * 0.92,
               color=METHOD_STYLE[method]["color"], label=METHOD_LABELS[method])
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([g.replace("_", " ") for g in groups])
    ax.set_ylabel("median relative error at $K=1$")
    figs.legend_below(fig, ax, ncol=3)
    written += figs.save(fig, out, "gradient_k1_by_group")

    # 3. Norm ratio against K. Unbiasedness in one panel: the accumulated mean's
    #    norm must approach the dense norm. A ratio pinned away from 1 is bias.
    fig, ax = plt.subplots(figsize=(figs.FIG_WIDTH_SINGLE * 1.5, 2.4))
    for method in METHOD_ORDER:
        curve = sorted([r for r in k_rows if r["sampling_method"] == method
                        and r["parameter_group"] == POOLED_GROUP],
                       key=lambda r: r["K"])
        if not curve:
            continue
        ax.plot([r["K"] for r in curve],
                [r["gradient_norm_ratio_mean"] for r in curve],
                label=METHOD_LABELS[method], linewidth=1.0, markersize=3,
                **METHOD_STYLE[method])
    ax.axhline(1.0, color="#000000", linewidth=0.7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("accumulated masks $K$")
    ax.set_ylabel(r"$\|$sparse mean$\| / \|$dense$\|$")
    figs.legend_below(fig, ax, ncol=3)
    written += figs.save(fig, out, "gradient_norm_ratio_vs_K")
    return written
