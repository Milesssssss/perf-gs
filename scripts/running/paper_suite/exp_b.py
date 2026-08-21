"""Experiment B — the pixel-budget sweep.

Seed 42, rho in {0.10, 0.25, 0.50}, the three sparse regimes, all 13 scenes. The
rho=0.25 column is the SAME cells Experiment A trained (not a re-run), and the
dense baseline is shared across all three columns because a pixel fraction does
not apply to dense supervision — there is one dense run per scene and it is the
reference for every budget.

Everything here is expressed as a GAP AGAINST DENSE on the same scene, rather than
as an absolute score, because absolute PSNR varies by ~5 dB across these scenes
and would swamp a budget effect of a few tenths. The dense run is the natural
per-scene control, and it is paired exactly.

One comparability caveat is carried explicitly through this experiment. The
rho=0.25 cells are the instrumented ones: their loop contains the convergence
ladder's eight held-out sweeps, whose cost ``train_active_s`` subtracts. The 0.10
and 0.50 cells have no mid-loop sweeps and need no correction. So the
time-versus-fraction curve mixes one corrected point with two uncorrected ones.
Experiment A's timing validation measures whether that correction is sound; the
figure and table both annotate the corrected point, and the report states the
measured discrepancy rather than leaving the reader to assume there is none.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as cfg
from . import figures as figs
from . import stats
from . import tables
from .results import RAW_COLUMNS, ResultSet

SWEEP_COLUMNS = (
    "grouping", "group", "regime", "pixel_fraction", "n_scenes", "n_cells",
    "scenes_expected", "complete", "timing_corrected",
    "psnr_srgb_mean", "psnr_srgb_std",
    "ssim_srgb_mean", "lpips_mean",
    "psnr_gap_vs_dense_mean", "psnr_gap_vs_dense_std",
    "ssim_gap_vs_dense_mean",
    "lpips_gap_vs_dense_mean", "lpips_gap_vs_dense_std",
    "train_active_s_mean", "train_active_s_std",
    "speedup_vs_dense_mean", "n_gaussians_mean",
)

GAP_BY_SCENE_COLUMNS = (
    "scene", "dataset_family", "regime", "pixel_fraction", "seed",
    "psnr_srgb", "dense_psnr_srgb", "psnr_gap_vs_dense",
    "ssim_srgb", "dense_ssim_srgb", "ssim_gap_vs_dense",
    "lpips", "dense_lpips", "lpips_gap_vs_dense",
    "train_active_s", "dense_train_active_s", "speedup_vs_dense",
    "n_gaussians", "timing_corrected",
)

CONTRAST_COLUMNS = (
    "metric", "contrast", "treatment", "baseline", "pixel_fraction",
    "n_scenes", "n_cells", "mean_delta", "std_delta_scenes",
    "min_delta_scene", "max_delta_scene", "ci_low", "ci_high",
    "ci_excludes_zero", "sign_test_p", "wins", "losses", "ties", "win_rate",
    "higher_is_better", "unpaired_dropped",
)

GAP_METRICS = (("psnr_srgb", "psnr_gap_vs_dense"),
               ("ssim_srgb", "ssim_gap_vs_dense"),
               ("lpips", "lpips_gap_vs_dense"))


def _dense_index(results: ResultSet, seed: int) -> dict[str, dict]:
    """``{scene: dense metrics}`` — the per-scene control for every budget."""
    out: dict[str, dict] = {}
    for result in results.select(regime="full", fraction=1.0, seed=seed):
        out[result.cell.scene] = {
            "psnr_srgb": result.metric("psnr_srgb"),
            "ssim_srgb": result.metric("ssim_srgb"),
            "lpips": result.metric("lpips"),
            "train_active_s": result.metric("train_active_s"),
        }
    return out


def _gap_rows(results: ResultSet, scenes, fractions, seed: int) -> list[dict]:
    """Per-cell gap against the same scene's dense run."""
    dense = _dense_index(results, seed)
    rows: list[dict] = []
    for fraction in fractions:
        for regime in cfg.SPARSE_REGIMES:
            for result in results.select(regime=regime, fraction=fraction,
                                         seed=seed):
                scene = result.cell.scene
                if scene not in set(scenes):
                    continue
                reference = dense.get(scene)
                if reference is None:
                    # No dense control for this scene: a gap cannot be formed,
                    # and inventing one from another scene's dense run would be
                    # meaningless. Skipped, and visible as a missing scene in the
                    # aggregate's completeness flag.
                    continue

                def gap(metric: str):
                    value = result.metric(metric)
                    base = reference.get(metric)
                    if value is None or base is None:
                        return None
                    # Signed so NEGATIVE always means "worse than dense",
                    # including for LPIPS, where the raw values are lower-better
                    # and the naive difference would carry the opposite sign.
                    if stats.HIGHER_IS_BETTER.get(metric):
                        return value - base
                    return base - value

                active = result.metric("train_active_s")
                dense_active = reference.get("train_active_s")
                rows.append({
                    "scene": scene,
                    "dataset_family": cfg.scene_family(scene),
                    "regime": regime,
                    "pixel_fraction": fraction,
                    "seed": seed,
                    "psnr_srgb": result.metric("psnr_srgb"),
                    "dense_psnr_srgb": reference.get("psnr_srgb"),
                    "psnr_gap_vs_dense": gap("psnr_srgb"),
                    "ssim_srgb": result.metric("ssim_srgb"),
                    "dense_ssim_srgb": reference.get("ssim_srgb"),
                    "ssim_gap_vs_dense": gap("ssim_srgb"),
                    "lpips": result.metric("lpips"),
                    "dense_lpips": reference.get("lpips"),
                    "lpips_gap_vs_dense": gap("lpips"),
                    "train_active_s": active,
                    "dense_train_active_s": dense_active,
                    "speedup_vs_dense": ((dense_active / active)
                                         if active and dense_active else None),
                    "n_gaussians": result.metric("n_gaussians"),
                    # True when this cell's active time required subtracting
                    # mid-loop observation cost.
                    "timing_corrected": bool(result.cell.convergence),
                })
    return rows


def _aggregate(gap_rows: list[dict], scenes, fractions) -> list[dict]:
    """Family and all-scene aggregates, scene-weighted."""
    groupings: list[tuple[str, dict[str, tuple[str, ...]]]] = [
        ("family", dict(cfg.DATASET_FAMILIES)),
        ("family_group", {
            group: tuple(s for fam in fams for s in cfg.DATASET_FAMILIES[fam])
            for group, fams in cfg.FAMILY_GROUPS.items()}),
        ("all", {"all_scenes": tuple(scenes)}),
    ]
    out: list[dict] = []
    for grouping, groups in groupings:
        for group, group_scenes in groups.items():
            members = [s for s in group_scenes if s in set(scenes)]
            if not members:
                continue
            for fraction in fractions:
                for regime in cfg.SPARSE_REGIMES:
                    picked = [r for r in gap_rows
                              if r["regime"] == regime
                              and abs(r["pixel_fraction"] - fraction) < 1e-9
                              and r["scene"] in set(members)]
                    if not picked:
                        continue
                    present = {r["scene"] for r in picked}
                    row: dict = {
                        "grouping": grouping, "group": group, "regime": regime,
                        "pixel_fraction": fraction,
                        "n_scenes": len(present), "n_cells": len(picked),
                        "scenes_expected": len(members),
                        "complete": len(present) == len(members),
                        "timing_corrected": any(r["timing_corrected"]
                                                for r in picked),
                    }
                    for key in ("psnr_srgb", "ssim_srgb", "lpips",
                                "psnr_gap_vs_dense", "ssim_gap_vs_dense",
                                "lpips_gap_vs_dense", "train_active_s",
                                "speedup_vs_dense", "n_gaussians"):
                        summary = stats.mean_std([r.get(key) for r in picked])
                        row[f"{key}_mean"] = summary["mean"]
                        row[f"{key}_std"] = summary["std"]
                    out.append(row)
    return out


def _contrasts(results: ResultSet, scenes, fractions, seed: int) -> list[dict]:
    """STBN over IID and STBN over independent blue noise, at each fraction."""
    pairs = (("stbn", "iid_white", "stbn_vs_iid"),
             ("stbn", "indep_blue", "stbn_vs_indep_blue"),
             ("indep_blue", "iid_white", "indep_blue_vs_iid"))
    rows: list[dict] = []
    for fraction in fractions:
        for treatment, baseline, name in pairs:
            for metric in ("psnr_srgb", "ssim_srgb", "lpips",
                           "train_active_s"):
                treat = results.metric_map(metric, regime=treatment,
                                           fraction=fraction, seeds=(seed,),
                                           scenes=scenes)
                base = results.metric_map(metric, regime=baseline,
                                          fraction=fraction, seeds=(seed,),
                                          scenes=scenes)
                if not treat or not base:
                    continue
                result = stats.paired_summary(
                    treatment, baseline, metric,
                    treatment_values=treat, baseline_values=base,
                    resamples=cfg.BOOTSTRAP_RESAMPLES, seed=cfg.BOOTSTRAP_SEED)
                row = result.as_row()
                row.update({"contrast": name, "pixel_fraction": fraction})
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _figure_gaps(aggregate: list[dict], figures_dir: Path, fractions) -> None:
    plt = figs.pyplot()
    if plt is None:
        return
    allrows = [r for r in aggregate
               if r["grouping"] == "all" and r["group"] == "all_scenes"]
    if not allrows:
        return

    panels = [("psnr_gap_vs_dense", "PSNR gap vs dense (dB)", 3),
              ("lpips_gap_vs_dense", "LPIPS gap vs dense", 4),
              ("train_active_s", "active train time (s)", 1),
              ("speedup_vs_dense", "speedup vs dense ($\\times$)", 2)]
    # Four panels each need their own y-label and tick labels; the extra width
    # keeps tight_layout from having to give up (it warns and leaves overlaps).
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(figs.FIG_WIDTH_DOUBLE * 1.25, 2.4))
    for ax, (key, ylabel, _digits) in zip(axes, panels):
        drew = False
        for regime in cfg.SPARSE_REGIMES:
            style = figs.style_for(regime)
            xs, ys, es = [], [], []
            for fraction in fractions:
                row = next((r for r in allrows if r["regime"] == regime
                            and abs(r["pixel_fraction"] - fraction) < 1e-9), None)
                if row is None or row.get(f"{key}_mean") is None:
                    continue
                xs.append(fraction)
                ys.append(row[f"{key}_mean"])
                es.append(row.get(f"{key}_std") or 0.0)
            if xs:
                drew = True
                ax.errorbar(xs, ys, yerr=es, capsize=2, linewidth=1.2,
                            markersize=4, color=style["color"],
                            marker=style["marker"], linestyle=style["linestyle"],
                            label=style["label"])
        if key.endswith("gap_vs_dense"):
            ax.axhline(0.0, color="#000000", linewidth=0.7, linestyle=":")
        if key == "speedup_vs_dense":
            ax.axhline(1.0, color="#000000", linewidth=0.7, linestyle=":")
        ax.set_xlabel("pixel fraction $\\rho$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(fractions))
        if not drew:
            figs.annotate_empty(ax)
    axes[0].set_title("Experiment B: budget sweep, all 13 scenes "
                      "(mean $\\pm$ s.d. over scenes, seed 42)", loc="left")
    axes[0].legend()
    fig.tight_layout()
    figs.save(fig, figures_dir, "expB_budget_gaps")


def _figure_contrasts(contrast_rows: list[dict], figures_dir: Path,
                      fractions) -> None:
    """STBN's advantage over each baseline as a function of budget."""
    plt = figs.pyplot()
    if plt is None:
        return
    panels = [("psnr_srgb", "$\\Delta$ PSNR (dB)"),
              ("lpips", "$\\Delta$ LPIPS (negative = better)")]
    contrasts = [("stbn_vs_iid", "STBN $-$ IID white", "#009E73", "D"),
                 ("stbn_vs_indep_blue", "STBN $-$ indep. blue", "#56B4E9", "^"),
                 ("indep_blue_vs_iid", "indep. blue $-$ IID", "#E69F00", "s")]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(figs.FIG_WIDTH_DOUBLE * 0.72, 2.4))
    for ax, (metric, ylabel) in zip(axes, panels):
        drew = False
        for name, label, color, marker in contrasts:
            xs, ys, los, his = [], [], [], []
            for fraction in fractions:
                row = next((r for r in contrast_rows
                            if r["contrast"] == name and r["metric"] == metric
                            and abs(r["pixel_fraction"] - fraction) < 1e-9), None)
                if row is None or row.get("mean_delta") is None:
                    continue
                xs.append(fraction)
                ys.append(row["mean_delta"])
                los.append(row.get("ci_low"))
                his.append(row.get("ci_high"))
            if not xs:
                continue
            drew = True
            ax.plot(xs, ys, color=color, marker=marker, linewidth=1.2,
                    markersize=4, label=label)
            if all(v is not None for v in los + his):
                ax.fill_between(xs, los, his, color=color, alpha=0.15,
                                linewidth=0)
        ax.axhline(0.0, color="#000000", linewidth=0.7, linestyle=":")
        ax.set_xlabel("pixel fraction $\\rho$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(list(fractions))
        if not drew:
            figs.annotate_empty(ax)
    axes[0].set_title("Paired improvement vs budget (band = 95% bootstrap CI "
                      "over per-scene differences)", loc="left")
    axes[0].legend()
    fig.tight_layout()
    figs.save(fig, figures_dir, "expB_contrasts_vs_fraction")


def _latex_sweep(aggregate: list[dict], latex_dir: Path, fractions) -> None:
    allrows = [r for r in aggregate
               if r["grouping"] == "all" and r["group"] == "all_scenes"]
    rows: list[list[str]] = []
    breaks: list[int] = []
    for fraction in fractions:
        if rows:
            breaks.append(len(rows))
        for regime in cfg.SPARSE_REGIMES:
            row = next((r for r in allrows if r["regime"] == regime
                        and abs(r["pixel_fraction"] - fraction) < 1e-9), None)
            if row is None:
                continue
            corrected = r"$^{\ddagger}$" if row.get("timing_corrected") else ""
            rows.append([
                (f"{fraction:.2f}" if regime == cfg.SPARSE_REGIMES[0] else ""),
                cfg.REGIME_LABELS[regime],
                tables.num(row.get("psnr_srgb_mean"), 2),
                tables.num(row.get("psnr_gap_vs_dense_mean"), 3, plus=True),
                tables.num(row.get("lpips_mean"), 4),
                tables.num(row.get("lpips_gap_vs_dense_mean"), 4, plus=True),
                tables.num(row.get("train_active_s_mean"), 1) + corrected,
                tables.num(row.get("speedup_vs_dense_mean"), 2),
            ])
    body = tables.latex_table(
        caption=(r"Pixel-budget sweep, all 13 scenes at seed 42. Gaps are paired "
                 r"against the SAME scene's dense run and signed so that negative "
                 r"is worse than dense for both metrics. Speedup is dense active "
                 r"time divided by sparse active time."),
        label="tab:stbn_budget",
        column_spec="llrrrrrr",
        header=[r"$\rho$", "Supervision", "PSNR", r"$\Delta$ vs dense",
                "LPIPS", r"$\Delta$ vs dense", "Active (s)", "Speedup"],
        rows=rows, midrule_before=breaks,
        notes=(r"$^{\ddagger}$ these cells also carry the convergence ladder, so "
               r"their active time is the wall-clock minus the measured cost of "
               r"the mid-loop evaluation sweeps; the other budgets need no such "
               r"correction. See the timing validation in the report."),
        source="tables/budget_sweep_summary.csv")
    tables.write_latex(latex_dir / "table_pixel_fraction_sweep.tex", body)


def run(results: ResultSet, out: dict, *, scenes=cfg.SCENES,
        fractions=cfg.SWEEP_FRACTIONS, seed: int = cfg.PRIMARY_SEED) -> dict:
    tables_dir, figures_dir = out["tables"], out["figures"]
    latex_dir = out["latex"]

    cells = [r for r in results.select(seed=seed)
             if r.cell.scene in set(scenes)
             and (r.cell.regime == "full"
                  or any(abs(r.cell.fraction - f) < 1e-9 for f in fractions))]
    tables.write_csv(out["raw"] / "budget_sweep_raw.csv", results.rows(cells),
                     RAW_COLUMNS)

    gap_rows = _gap_rows(results, scenes, fractions, seed)
    tables.write_csv(tables_dir / "budget_sweep_by_scene.csv", gap_rows,
                     GAP_BY_SCENE_COLUMNS)

    aggregate = _aggregate(gap_rows, scenes, fractions)
    tables.write_csv(tables_dir / "budget_sweep_summary.csv", aggregate,
                     SWEEP_COLUMNS)

    contrast_rows = _contrasts(results, scenes, fractions, seed)
    tables.write_csv(tables_dir / "budget_sweep_contrasts.csv", contrast_rows,
                     CONTRAST_COLUMNS)

    _figure_gaps(aggregate, figures_dir, fractions)
    _figure_contrasts(contrast_rows, figures_dir, fractions)
    _latex_sweep(aggregate, latex_dir, fractions)

    trends = {}
    for regime in cfg.SPARSE_REGIMES:
        trends[regime] = {
            f"{r['pixel_fraction']:.2f}": {
                "psnr_gap_vs_dense": r.get("psnr_gap_vs_dense_mean"),
                "lpips_gap_vs_dense": r.get("lpips_gap_vs_dense_mean"),
                "active_s": r.get("train_active_s_mean"),
                "speedup_vs_dense": r.get("speedup_vs_dense_mean"),
                "n_scenes": r.get("n_scenes"),
                "complete": r.get("complete"),
            }
            for r in aggregate
            if r["grouping"] == "all" and r["group"] == "all_scenes"
            and r["regime"] == regime
        }
    return {
        "seed": seed,
        "fractions": list(fractions),
        "n_cells": len(cells),
        "trends_all_scenes": trends,
        "contrasts": {f"{r['contrast']}::{r['metric']}::rho{r['pixel_fraction']}":
                      r for r in contrast_rows},
        "artifacts": {
            "raw_csv": "raw/budget_sweep_raw.csv",
            "by_scene_csv": "tables/budget_sweep_by_scene.csv",
            "summary_csv": "tables/budget_sweep_summary.csv",
            "contrasts_csv": "tables/budget_sweep_contrasts.csv",
            "figures": ["figures/expB_budget_gaps.pdf",
                        "figures/expB_contrasts_vs_fraction.pdf"],
            "latex": ["latex/table_pixel_fraction_sweep.tex"],
        },
    }
