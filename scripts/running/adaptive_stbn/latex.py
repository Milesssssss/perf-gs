# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""LaTeX tables and the raw index — the ``tables/`` and ``raw/`` halves of the
required output layout.

``tables/`` holds paper-ready booktabs tables, one per headline claim. Each carries
a comment naming the CSV it was generated from, so a number in a paper can be
traced back to the run that produced it.

``raw/`` holds the unaggregated index: every manifest cell with every field, and
the stage records verbatim. The point is that any table above can be rebuilt from
``raw/`` alone, without re-reading the suite's code — which is what makes the
aggregation auditable rather than merely reproducible.

Two conventions inherited from :mod:`paper_suite.tables` and kept deliberately:
missing renders as ``--`` and never as ``0``, and an aggregate over fewer cells
than planned is daggered rather than silently thinned.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from paper_suite import tables

from . import config as cfg
from . import gradients as grad
from .manifest import Manifest

SOURCE_NOTE = "scripts/running/adaptive_stbn"


def _num(value, digits: int = 3, *, plus: bool = False) -> str:
    return tables.num(value, digits, plus=plus)


def write_all(paths: dict, manifest: Manifest, training: dict,
              gradient: dict | None, sampling: dict | None) -> dict:
    """Write every LaTeX table and the raw index. Returns a list of paths."""
    written: list[Path] = []
    written += _raw(paths, manifest)
    written += _pilot_table(paths, training)
    written += _contrast_table(paths, training)
    written += _fraction_table(paths, training)
    if gradient:
        written += _gradient_table(paths, gradient)
    if sampling:
        written += _sampling_table(paths, sampling)
    return {"files": [str(p.relative_to(paths["root"])) for p in written]}


# ---------------------------------------------------------------------------
# raw/
# ---------------------------------------------------------------------------

def _raw(paths: dict, manifest: Manifest) -> list[Path]:
    out = paths["raw"]
    out.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for _, record in sorted(manifest.records.items())]
    columns = sorted({key for row in rows for key in row}) if rows else []
    # Lists (argv, consumers, dirty files) are JSON-encoded rather than dropped:
    # the argv is the single most useful field for reproducing a cell by hand.
    flat = [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
             for k, v in row.items()} for row in rows]
    written = [tables.write_csv(out / "all_cells.csv", flat, columns)]
    (out / "stage_records.json").write_text(
        json.dumps(manifest.stages, indent=2, default=str))
    written.append(out / "stage_records.json")
    (out / "suite_plan.json").write_text(
        json.dumps(manifest.suite, indent=2, default=str))
    written.append(out / "suite_plan.json")
    return written


# ---------------------------------------------------------------------------
# tables/
# ---------------------------------------------------------------------------

def _pilot_table(paths: dict, training: dict) -> list[Path]:
    summary = training.get("pilot_summary") or []
    if not summary:
        return []
    rows = []
    for row in summary:
        marker = "" if row.get("complete") else r"$^{\dagger}$"
        rows.append([
            tables.latex_escape(cfg.REGIME_LABELS.get(row["regime"],
                                                      row["regime"])),
            f"{row['n_scenes']}{marker}",
            _num(row.get("psnr_mean"), 3), _num(row.get("ssim_mean"), 4),
            _num(row.get("lpips_mean"), 4),
            _num(row.get("backward_raster_ms_mean"), 3),
            _num(row.get("loss_ms_mean"), 3),
            _num(row.get("adaptive_sampling_ms_mean"), 3),
            _num(row.get("backward_speedup_vs_dense"), 2),
            _num(row.get("n_gaussians_mean"), 0),
        ])
    body = tables.latex_table(
        caption=(f"Five-scene pilot at $\\rho={cfg.PRIMARY_FRACTION}$, "
                 f"$\\alpha={cfg.PRIMARY_ALPHA}$, $\\sigma={cfg.PRIMARY_SIGMA}$, "
                 f"seed {cfg.PRIMARY_SEED}, {cfg.ITERS} iterations. Means over "
                 f"scenes; the backward-raster column is the primary performance "
                 f"metric and the adaptive column is the sampler's own share of "
                 f"the (dense) loss pass, already contained in it."),
        label="tab:adaptive-pilot",
        column_spec="lrrrrrrrrr",
        header=["Regime", "$n$", "PSNR", "SSIM", "LPIPS", "bwd ms", "loss ms",
                "adapt ms", "bwd speedup", "Gaussians"],
        rows=rows, source="training/pilot_summary.csv",
        notes=tables.INCOMPLETE_NOTE)
    return [tables.write_latex(paths["tables"] / "pilot_summary.tex", body)]


def _contrast_table(paths: dict, training: dict) -> list[Path]:
    contrasts = training.get("contrasts") or []
    if not contrasts:
        return []
    rows = [[
        tables.latex_escape(f"{r['treatment']} - {r['control']}"),
        tables.latex_escape(r["axis"]), f"{r['n_scenes']}",
        _num(r.get("psnr_delta_mean"), 3, plus=True),
        _num(r.get("psnr_delta_std"), 3),
        _num(r.get("ssim_delta_mean"), 4, plus=True),
        _num(r.get("lpips_delta_mean"), 4, plus=True),
        f"{r.get('wins')}/{r.get('losses')}",
    ] for r in contrasts]
    body = tables.latex_table(
        caption=("Controlled contrasts, computed as per-scene paired deltas and "
                 "then averaged. LPIPS is a distance, so negative is better there "
                 "and positive is better in the other two. The win/loss column is "
                 "the count of scenes, which five scenes at one seed cannot "
                 "replace with a mean."),
        label="tab:adaptive-contrasts",
        column_spec="llrrrrrr",
        header=["Contrast", "Axis", "$n$", r"$\Delta$PSNR", "sd",
                r"$\Delta$SSIM", r"$\Delta$LPIPS", "win/loss"],
        rows=rows, source="training/pilot_contrasts.csv")
    return [tables.write_latex(paths["tables"] / "contrasts.tex", body)]


def _fraction_table(paths: dict, training: dict) -> list[Path]:
    sweep = training.get("fraction_sweep") or []
    if not sweep:
        return []
    rows = [[
        tables.latex_escape(cfg.REGIME_LABELS.get(r["regime"], r["regime"])),
        _num(r.get("rho_enforced"), 5), f"{r['n_scenes']}",
        _num(r.get("psnr_mean"), 3), _num(r.get("psnr_std"), 3),
        _num(r.get("ssim_mean"), 4), _num(r.get("lpips_mean"), 4),
        _num(r.get("backward_raster_ms_mean"), 3),
        _num(r.get("backward_speedup_vs_dense"), 2),
    ] for r in sweep]
    body = tables.latex_table(
        caption=("Pixel-budget sweep, uniform STBN against adaptive STBN. "
                 "$\\rho$ is the ENFORCED fraction "
                 "$\\mathrm{pixels\\_per\\_tile}(\\rho)/64$, which is the density "
                 "the uniform table realizes; both arms are matched on it."),
        label="tab:adaptive-fractions",
        column_spec="lrrrrrrrr",
        header=["Regime", r"$\rho$", "$n$", "PSNR", "sd", "SSIM", "LPIPS",
                "bwd ms", "bwd speedup"],
        rows=rows, source="training/fraction_sweep.csv")
    return [tables.write_latex(paths["tables"] / "fraction_sweep.tex", body)]


def _gradient_table(paths: dict, gradient: dict) -> list[Path]:
    headline = gradient.get("headline") or {}
    ks = ("1", "2", "4", "8", "16", "32", "64")
    rows = []
    for method in grad.METHOD_ORDER:
        stats = headline.get(method)
        if not isinstance(stats, dict):
            continue
        errors = stats.get("cumulative_error_by_K") or {}
        rows.append([tables.latex_escape(grad.METHOD_LABELS[method])]
                    + [_num(errors.get(k), 4) for k in ks]
                    + [_num((stats.get("norm_ratio_by_K") or {}).get("64"), 4)])
    if not rows:
        return []
    body = tables.latex_table(
        caption=(f"Frozen-state gradient error against the DENSE gradient, "
                 f"accumulated over $K$ masks at a matched expected "
                 f"$\\rho={gradient.get('rho_enforced')}$. Pooled over "
                 f"{', '.join(gradient.get('groups_reported', []))}; "
                 f"\\texttt{{scale}} is excluded because its int64 accumulator "
                 f"saturates. The last column is the norm ratio at $K=64$, which "
                 f"must approach 1 for an unbiased estimator."),
        label="tab:adaptive-gradient",
        column_spec="l" + "r" * (len(ks) + 1),
        header=["Method"] + [f"$K{{=}}{k}$" for k in ks] + ["ratio $K{=}64$"],
        rows=rows, source="gradient_validation/gradient_by_K.csv")
    return [tables.write_latex(paths["tables"] / "gradient_by_K.tex", body)]


def _sampling_table(paths: dict, sampling: dict) -> list[Path]:
    headline = sampling.get("headline") or {}
    rows = []
    for sampler, stats in headline.items():
        rows.append([
            tables.latex_escape(sampler),
            _num(stats.get("calibration_rmse_max"), 5),
            _num(stats.get("rmse_over_floor_max"), 2),
            _num(stats.get("calibration_max_abs"), 4),
            _num(stats.get("low_freq_ratio_vs_white_mean"), 3),
            _num(stats.get("temporal_lag1_mean"), 4, plus=True),
        ])
    if not rows:
        return []
    body = tables.latex_table(
        caption=("Sampling validation on four static importance fields. "
                 "Calibration RMSE is reported against the floor a perfectly "
                 "calibrated independent sampler would score at the same number "
                 "of visits. The last two columns are on the variance-normalized "
                 "residual $Z_i=(M_i-p_i)/\\sqrt{p_i(1-p_i)}$, not on the mask: a "
                 "nonuniform density necessarily puts the mean field's "
                 "low-frequency content into $M$."),
        label="tab:adaptive-sampling",
        column_spec="lrrrrr",
        header=["Sampler", "max RMSE", "/floor", r"max $|\hat p - p|$",
                "low-freq / white", "lag-1 autocorr."],
        rows=rows, source="sampling_validation/calibration.csv")
    return [tables.write_latex(paths["tables"] / "sampling_validation.tex", body)]
