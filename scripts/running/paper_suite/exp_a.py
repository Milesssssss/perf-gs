# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Experiment A — the primary 25% ablation.

13 scenes x 4 regimes x 3 seeds at rho = 0.25, 3000 iterations. Produces the
paper's main quantitative table, the dataset-family aggregate, and the
spatial-vs-temporal decomposition.

The decomposition is the point of the whole design, so it is worth restating what
each contrast isolates and why the arithmetic is valid:

    indep_blue - iid_white      SPATIAL: same independent per-step draw, same
                                per-tile count, same reweighting; only the noise
                                the slices were ranked from differs.
    stbn       - indep_blue     TEMPORAL: the same 64 blue-noise slices; only the
                                slice ORDER differs (independent draw vs each
                                camera's own visit counter).
    stbn       - iid_white      the two effects combined, and it should be close
                                to their sum — if it is not, they interact, which
                                is itself worth reporting.
    stbn       - full           the cost of sparsity: what 25% of the pixels gives
                                up against dense supervision.

Timing needs one extra care. The seed-42 cells carry the convergence ladder, so
their wall-clock contains eight held-out sweeps that the other seeds' does not.
``train_active_s`` subtracts that cost, but a subtraction is a correction, not a
measurement, so the headline speed numbers are taken from the two UNOBSERVED seeds
and seed 42 is reported separately as a cross-check
(:func:`timing_validation`). If the corrected and uncorrected figures agree, the
correction is sound and that is stated; if they do not, the discrepancy is
reported rather than averaged away.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as cfg
from . import figures as figs
from . import stats
from . import tables
from .results import (PER_VIEW_COLUMNS, RAW_COLUMNS, ResultSet, per_view_rows)

# The four contrasts, as (treatment, baseline, short name, what it isolates).
CONTRASTS: tuple[tuple[str, str, str, str], ...] = (
    ("indep_blue", "iid_white", "spatial",
     "blue-noise arrangement, independent draws both sides"),
    ("stbn", "indep_blue", "temporal",
     "temporal ordering, identical blue-noise slice pool"),
    ("stbn", "iid_white", "spatial+temporal",
     "combined effect of arrangement and ordering"),
    ("stbn", "full", "sparsity_cost",
     "25% supervision against dense supervision"),
)

METRICS: tuple[str, ...] = ("psnr_srgb", "ssim_srgb", "lpips",
                            "train_active_s", "n_gaussians", "render_fps")
QUALITY_METRICS: tuple[str, ...] = ("psnr_srgb", "ssim_srgb", "lpips")

SCENE_SUMMARY_COLUMNS = (
    "scene", "dataset_family", "dataset_family_group", "regime",
    "pixel_fraction", "n_seeds", "seeds_expected", "complete",
    "psnr_srgb_mean", "psnr_srgb_std", "psnr_srgb_min", "psnr_srgb_max",
    "ssim_srgb_mean", "ssim_srgb_std",
    "lpips_mean", "lpips_std",
    "train_active_s_mean", "train_active_s_std",
    "train_wall_s_mean", "process_wall_s_mean",
    "n_gaussians_mean", "n_gaussians_std",
    "render_fps_mean", "gpu_device_peak_mb_mean",
    "width", "height", "n_train", "n_test",
)

DATASET_SUMMARY_COLUMNS = (
    "grouping", "group", "regime", "pixel_fraction", "n_scenes", "n_cells",
    "scenes_expected", "complete",
    "psnr_srgb_mean", "psnr_srgb_std",
    "ssim_srgb_mean", "ssim_srgb_std",
    "lpips_mean", "lpips_std",
    "train_active_s_mean", "train_active_s_std",
    "n_gaussians_mean", "render_fps_mean",
)

DELTA_COLUMNS = (
    "metric", "contrast", "isolates", "treatment", "baseline",
    "n_scenes", "n_cells", "mean_delta", "std_delta_scenes",
    "min_delta_scene", "max_delta_scene", "ci_low", "ci_high",
    "ci_excludes_zero", "sign_test_p", "wins", "losses", "ties", "win_rate",
    "higher_is_better", "unpaired_dropped",
)

DELTA_BY_SCENE_COLUMNS = (
    "metric", "contrast", "treatment", "baseline", "scene", "dataset_family",
    "seed", "delta", "treatment_value", "baseline_value",
)


def _scene_rows(results: ResultSet, scenes, seeds, fraction) -> list[dict]:
    """Per (scene, regime) mean/std across seeds."""
    rows: list[dict] = []
    for scene in scenes:
        for regime in cfg.REGIMES:
            use_fraction = 1.0 if regime == "full" else fraction
            cells = [r for r in results.select(regime=regime,
                                               fraction=use_fraction,
                                               scene=scene)
                     if r.cell.seed in set(seeds)]
            if not cells:
                continue
            row: dict = {
                "scene": scene,
                "dataset_family": cfg.scene_family(scene),
                "dataset_family_group": cfg.family_group(scene),
                "regime": regime,
                "pixel_fraction": use_fraction,
                "n_seeds": len(cells),
                "seeds_expected": len(seeds),
                "complete": len(cells) == len(seeds),
            }
            for metric in ("psnr_srgb", "ssim_srgb", "lpips",
                           "train_active_s", "n_gaussians"):
                summary = stats.mean_std([c.metric(metric) for c in cells])
                row[f"{metric}_mean"] = summary["mean"]
                row[f"{metric}_std"] = summary["std"]
                if metric == "psnr_srgb":
                    row["psnr_srgb_min"] = summary["min"]
                    row["psnr_srgb_max"] = summary["max"]
            for metric in ("train_wall_s", "process_wall_s", "render_fps",
                           "gpu_device_peak_mb"):
                row[f"{metric}_mean"] = stats.mean_std(
                    [c.metric(metric) for c in cells])["mean"]
            first = cells[0].record
            row.update({"width": first.width, "height": first.height,
                        "n_train": first.n_train, "n_test": first.n_test})
            rows.append(row)
    return rows


def _dataset_rows(results: ResultSet, scenes, seeds, fraction) -> list[dict]:
    """Aggregates over dataset families, and over all scenes.

    Two groupings are emitted: the four fine families (Mip-NeRF 360 split into
    outdoor/indoor, T&T, Deep Blending) and the three coarse ones the literature
    reports. Both carry ``complete``, so an aggregate over a partially-failed
    family is never mistaken for a full one.
    """
    rows: list[dict] = []
    groupings: list[tuple[str, dict[str, tuple[str, ...]]]] = [
        ("family", {k: v for k, v in cfg.DATASET_FAMILIES.items()}),
        ("family_group", {
            group: tuple(s for fam in fams for s in cfg.DATASET_FAMILIES[fam])
            for group, fams in cfg.FAMILY_GROUPS.items()}),
        ("all", {"all_scenes": tuple(scenes)}),
    ]
    for grouping, groups in groupings:
        for group, group_scenes in groups.items():
            members = [s for s in group_scenes if s in set(scenes)]
            if not members:
                continue
            for regime in cfg.REGIMES:
                use_fraction = 1.0 if regime == "full" else fraction
                cells = [r for r in results.select(regime=regime,
                                                   fraction=use_fraction)
                         if r.cell.scene in set(members)
                         and r.cell.seed in set(seeds)]
                if not cells:
                    continue
                present_scenes = {r.cell.scene for r in cells}
                row: dict = {
                    "grouping": grouping,
                    "group": group,
                    "regime": regime,
                    "pixel_fraction": use_fraction,
                    "n_scenes": len(present_scenes),
                    "n_cells": len(cells),
                    "scenes_expected": len(members),
                    "complete": (len(present_scenes) == len(members)
                                 and len(cells) == len(members) * len(seeds)),
                }
                # Scene-then-seed pooling would weight a scene by how many of its
                # seeds survived. Average within a scene first so every scene
                # counts once, which is also the unit the CIs use.
                for metric in ("psnr_srgb", "ssim_srgb", "lpips",
                               "train_active_s", "n_gaussians", "render_fps"):
                    per_scene = []
                    for scene in sorted(present_scenes):
                        values = [c.metric(metric) for c in cells
                                  if c.cell.scene == scene]
                        summary = stats.mean_std(values)
                        if summary["mean"] is not None:
                            per_scene.append(summary["mean"])
                    summary = stats.mean_std(per_scene)
                    row[f"{metric}_mean"] = summary["mean"]
                    row[f"{metric}_std"] = summary["std"]
                rows.append(row)
    return rows


def _paired(results: ResultSet, scenes, seeds, fraction
            ) -> tuple[list[dict], list[dict], dict]:
    """All contrasts x metrics: summary rows, per-scene rows, and the objects."""
    summary_rows: list[dict] = []
    scene_rows: list[dict] = []
    objects: dict[tuple[str, str], stats.PairedResult] = {}

    for treatment, baseline, name, isolates in CONTRASTS:
        for metric in METRICS:
            treat_map = results.metric_map(
                metric, regime=treatment,
                fraction=(1.0 if treatment == "full" else fraction),
                seeds=seeds, scenes=scenes)
            base_map = results.metric_map(
                metric, regime=baseline,
                fraction=(1.0 if baseline == "full" else fraction),
                seeds=seeds, scenes=scenes)
            if not treat_map or not base_map:
                continue
            result = stats.paired_summary(
                treatment, baseline, metric,
                treatment_values=treat_map, baseline_values=base_map,
                resamples=cfg.BOOTSTRAP_RESAMPLES, seed=cfg.BOOTSTRAP_SEED)
            objects[(name, metric)] = result
            row = result.as_row()
            row.update({"contrast": name, "isolates": isolates})
            summary_rows.append(row)

            for (scene, seed), delta in sorted(result.cell_deltas.items()):
                scene_rows.append({
                    "metric": metric, "contrast": name,
                    "treatment": treatment, "baseline": baseline,
                    "scene": scene, "dataset_family": cfg.scene_family(scene),
                    "seed": seed, "delta": delta,
                    "treatment_value": treat_map.get((scene, seed)),
                    "baseline_value": base_map.get((scene, seed)),
                })
    return summary_rows, scene_rows, objects


def timing_validation(results: ResultSet, scenes, fraction) -> dict:
    """Does subtracting observation time reconcile seed 42 with the clean seeds?

    Seed 42 carries eight held-out sweeps inside its loop; seeds 1337/2026 carry
    none. If ``train_active_s`` is doing its job, seed 42's corrected active time
    should sit inside the seed-to-seed spread of the unobserved runs. This
    computes that comparison per regime so the report can state it as a checked
    fact instead of an assumption.
    """
    clean_seeds = tuple(s for s in cfg.SEEDS if s != cfg.PRIMARY_SEED)
    out: dict = {"clean_seeds": list(clean_seeds),
                 "observed_seed": cfg.PRIMARY_SEED, "per_regime": {}}
    for regime in cfg.REGIMES:
        use_fraction = 1.0 if regime == "full" else fraction
        observed, clean = [], []
        per_scene_gap = []
        for scene in scenes:
            cells = results.select(regime=regime, fraction=use_fraction,
                                   scene=scene)
            obs = [c for c in cells if c.cell.seed == cfg.PRIMARY_SEED]
            cln = [c for c in cells if c.cell.seed in clean_seeds]
            if not obs or not cln:
                continue
            obs_active = stats.mean_std([c.metric("train_active_s") for c in obs])
            cln_active = stats.mean_std([c.metric("train_active_s") for c in cln])
            if obs_active["mean"] is None or cln_active["mean"] is None:
                continue
            observed.append(obs_active["mean"])
            clean.append(cln_active["mean"])
            per_scene_gap.append(obs_active["mean"] - cln_active["mean"])
        if not per_scene_gap:
            continue
        gap = stats.mean_std(per_scene_gap)
        relative = [
            (o - c) / c for o, c in zip(observed, clean) if c not in (0, None)]
        out["per_regime"][regime] = {
            "n_scenes": len(per_scene_gap),
            "observed_active_s_mean": stats.mean_std(observed)["mean"],
            "clean_active_s_mean": stats.mean_std(clean)["mean"],
            "gap_s_mean": gap["mean"],
            "gap_s_std": gap["std"],
            "gap_relative_mean": stats.mean_std(relative)["mean"],
            "gap_relative_max_abs": (max(abs(r) for r in relative)
                                     if relative else None),
        }
    worst = [v.get("gap_relative_max_abs") for v in out["per_regime"].values()
             if v.get("gap_relative_max_abs") is not None]
    out["worst_relative_gap"] = max(worst) if worst else None
    out["interpretation"] = (
        "Relative gap between the observation-corrected active time of the "
        "instrumented seed and the uncorrected active time of the unobserved "
        "seeds. Small values (a few percent, comparable to seed-to-seed spread) "
        "mean the correction is sound; large values mean the mid-loop sweeps "
        "perturbed the loop beyond their own duration and only the unobserved "
        "seeds should be quoted for speed.")
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _figure_per_scene_bars(results: ResultSet, scenes, seeds, fraction,
                           figures_dir: Path) -> None:
    plt = figs.pyplot()
    if plt is None:
        return
    present = [s for s in scenes
               if results.select(scene=s, regime="stbn", fraction=fraction)]
    if not present:
        return

    panels = [("psnr_srgb", "held-out PSNR (dB)"),
              ("ssim_srgb", "held-out SSIM"),
              ("lpips", "held-out LPIPS (lower better)"),
              ("train_active_s", "active train time (s)")]
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(max(figs.FIG_WIDTH_DOUBLE,
                                          0.55 * len(present) + 2.2),
                                      2.1 * len(panels)), sharex=True)
    x = np.arange(len(present))
    width = 0.8 / len(cfg.REGIMES)

    for ax, (metric, ylabel) in zip(axes, panels):
        finite: list[float] = []
        for index, regime in enumerate(cfg.REGIMES):
            use_fraction = 1.0 if regime == "full" else fraction
            style = figs.style_for(regime)
            means, errs = [], []
            for scene in present:
                cells = [c for c in results.select(regime=regime,
                                                   fraction=use_fraction,
                                                   scene=scene)
                         if c.cell.seed in set(seeds)]
                summary = stats.mean_std([c.metric(metric) for c in cells])
                means.append(summary["mean"] if summary["mean"] is not None
                             else np.nan)
                errs.append(summary["std"] or 0.0)
                if summary["mean"] is not None:
                    finite.append(summary["mean"])
            offset = (index - (len(cfg.REGIMES) - 1) / 2) * width
            ax.bar(x + offset, means, width, yerr=errs, capsize=1.5,
                   color=style["color"], label=style["label"],
                   error_kw={"linewidth": 0.6})
        ax.set_ylabel(ylabel)
        if metric in QUALITY_METRICS:
            # The regimes differ by tenths of a dB; a zero-based axis would
            # flatten the entire result into one band.
            figs.zoom_to_data(ax, finite,
                              min_pad=(0.05 if metric == "psnr_srgb" else 0.002))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(present, rotation=35, ha="right")
    axes[0].set_title(f"Experiment A: 4 regimes at $\\rho$={fraction}, "
                      f"mean $\\pm$ s.d. over seeds {sorted(seeds)}")
    axes[0].legend(ncol=len(cfg.REGIMES), loc="upper right")
    fig.tight_layout()
    figs.save(fig, figures_dir, "expA_per_scene_regimes")


def _figure_paired_deltas(objects, figures_dir: Path) -> None:
    """Per-scene paired deltas with the bootstrap interval on the mean.

    Shows every scene as a point, so a mean that depends on one outlier is
    visible as one, and negative cases are as prominent as positive ones.
    """
    plt = figs.pyplot()
    if plt is None:
        return
    panels = [("psnr_srgb", "$\\Delta$ PSNR (dB)"),
              ("ssim_srgb", "$\\Delta$ SSIM"),
              ("lpips", "$\\Delta$ LPIPS")]
    names = [c[2] for c in CONTRASTS]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(figs.FIG_WIDTH_DOUBLE, 2.6))
    for ax, (metric, label) in zip(axes, panels):
        any_data = False
        for index, name in enumerate(names):
            result = objects.get((name, metric))
            if result is None or not result.scene_deltas:
                continue
            any_data = True
            values = list(result.scene_deltas.values())
            jitter = np.random.default_rng(index).normal(0, 0.045, len(values))
            ax.scatter(np.full(len(values), index) + jitter, values, s=9,
                       alpha=0.6, color="#666666", zorder=3,
                       edgecolors="none")
            mean = result.scene_summary.get("mean")
            low, high = result.ci.get("ci_low"), result.ci.get("ci_high")
            if mean is not None:
                ax.plot([index], [mean], marker="_", markersize=18,
                        color="#0072B2", zorder=4)
            if low is not None and high is not None:
                ax.plot([index, index], [low, high], color="#0072B2",
                        linewidth=2.0, zorder=4, solid_capstyle="butt")
        ax.axhline(0.0, color="#000000", linewidth=0.7, linestyle=":")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel(label)
        if not any_data:
            figs.annotate_empty(ax)
    axes[0].set_title("Paired per-scene differences (dot = scene, "
                      "bar = mean with 95% bootstrap CI)", loc="left")
    fig.tight_layout()
    figs.save(fig, figures_dir, "expA_paired_deltas")


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def _latex_main(dataset_rows: list[dict], latex_dir: Path, fraction: float,
                seeds) -> None:
    """Main quantitative table: family aggregates x regimes."""
    by_group: dict[str, dict[str, dict]] = {}
    for row in dataset_rows:
        if row["grouping"] not in ("family_group", "all"):
            continue
        by_group.setdefault(row["group"], {})[row["regime"]] = row

    order = [g for g in (*cfg.FAMILY_GROUPS, "all_scenes") if g in by_group]
    pretty = {"mipnerf360": "Mip-NeRF 360", "tanks_and_temples": "Tanks\\&Temples",
              "deep_blending": "Deep Blending", "all_scenes": "All 13 scenes"}

    rows: list[list[str]] = []
    breaks: list[int] = []
    for group in order:
        if group == "all_scenes" and rows:
            breaks.append(len(rows))
        for regime in cfg.REGIMES:
            row = by_group[group].get(regime)
            if row is None:
                continue
            incomplete = r"$^{\dagger}$" if not row.get("complete") else ""
            rows.append([
                (pretty.get(group, group) if regime == cfg.REGIMES[0] else ""),
                cfg.REGIME_LABELS[regime] + incomplete,
                tables.num(row.get("psnr_srgb_mean"), 2),
                tables.num(row.get("psnr_srgb_std"), 2),
                tables.num(row.get("ssim_srgb_mean"), 4),
                tables.num(row.get("lpips_mean"), 4),
                tables.num(row.get("train_active_s_mean"), 1),
                tables.num((row.get("n_gaussians_mean") or 0) / 1000.0, 0)
                if row.get("n_gaussians_mean") else tables.LATEX_NA,
            ])
    body = tables.latex_table(
        caption=(rf"Held-out quality and active training time at "
                 rf"$\rho={fraction}$, {cfg.ITERS} iterations, "
                 rf"mean over {len(seeds)} seeds and over the scenes of each "
                 rf"dataset. PSNR/SSIM higher is better, LPIPS lower is better. "
                 rf"The s.d.\ column is the seed-to-seed spread of the "
                 rf"scene-averaged PSNR."),
        label="tab:stbn_main",
        column_spec="llrrrrrr",
        header=["Dataset", "Supervision", "PSNR", "s.d.", "SSIM", "LPIPS",
                "Active (s)", "Gauss.\\ (k)"],
        rows=rows, midrule_before=breaks,
        notes=("Dense supervises every pixel; the three sparse regimes supervise "
               f"{fraction:.0%} of pixels per backward pass with identical "
               r"per-tile counts and $1/\rho$ reweighting, differing only in the "
               r"mask's spatial arrangement and temporal ordering. "
               + tables.INCOMPLETE_NOTE),
        source="tables/main_25pct_dataset_summary.csv")
    tables.write_latex(latex_dir / "table_main_quantitative.tex", body)


def _latex_ablation(delta_rows: list[dict], latex_dir: Path,
                    fraction: float, n_scenes: int) -> None:
    """Spatial-vs-temporal decomposition, with intervals and win counts."""
    wanted = ("spatial", "temporal", "spatial+temporal", "sparsity_cost")
    pretty = {
        "spatial": r"IID $\rightarrow$ indep.\ blue \emph{(spatial)}",
        "temporal": r"Indep.\ blue $\rightarrow$ STBN \emph{(temporal)}",
        "spatial+temporal": r"IID $\rightarrow$ STBN \emph{(both)}",
        "sparsity_cost": r"Dense $\rightarrow$ STBN \emph{(cost of sparsity)}",
    }
    index = {(r["contrast"], r["metric"]): r for r in delta_rows}
    rows: list[list[str]] = []
    for contrast in wanted:
        for metric, digits in (("psnr_srgb", 3), ("ssim_srgb", 4),
                               ("lpips", 4)):
            row = index.get((contrast, metric))
            if row is None:
                continue
            rows.append([
                (pretty[contrast] if metric == "psnr_srgb" else ""),
                {"psnr_srgb": "PSNR (dB)", "ssim_srgb": "SSIM",
                 "lpips": "LPIPS"}[metric],
                tables.num(row.get("mean_delta"), digits, plus=True),
                tables.num(row.get("std_delta_scenes"), digits),
                tables.ci_cell(row, digits),
                f"{row.get('wins', 0)}/{row.get('n_cells', 0)}",
                tables.num(row.get("sign_test_p"), 3),
            ])
    body = tables.latex_table(
        caption=(rf"Spatial-versus-temporal decomposition at $\rho={fraction}$. "
                 rf"Differences are paired within (scene, seed); the s.d., the "
                 rf"95\% bootstrap CI and the sign test are computed over the "
                 rf"{n_scenes} per-scene differences (seeds averaged within a "
                 rf"scene), and the win count is over all (scene, seed) cells. "
                 rf"Positive is better for PSNR/SSIM, negative for LPIPS."),
        label="tab:stbn_ablation",
        column_spec="llrrlrr",
        header=["Contrast", "Metric", r"$\Delta$", "s.d.", "95\\% CI",
                "Wins", "Sign $p$"],
        rows=rows,
        midrule_before=[i for i in range(3, len(rows), 3)],
        notes=("An interval containing zero means the data do not resolve the "
               "sign of the effect at this sample size; it is reported as such "
               "rather than as an absence of effect."),
        source="tables/main_25pct_paired_deltas.csv")
    tables.write_latex(latex_dir / "table_spatial_vs_temporal.tex", body)


def _latex_family(dataset_rows: list[dict], latex_dir: Path,
                  fraction: float) -> None:
    by_family: dict[str, dict[str, dict]] = {}
    for row in dataset_rows:
        if row["grouping"] != "family":
            continue
        by_family.setdefault(row["group"], {})[row["regime"]] = row
    pretty = {"mipnerf360_outdoor": "360 outdoor", "mipnerf360_indoor": "360 indoor",
              "tanks_and_temples": "Tanks\\&Temples", "deep_blending": "Deep Blending"}
    rows: list[list[str]] = []
    for family in cfg.DATASET_FAMILIES:
        if family not in by_family:
            continue
        for regime in cfg.REGIMES:
            row = by_family[family].get(regime)
            if row is None:
                continue
            rows.append([
                (pretty.get(family, family) if regime == cfg.REGIMES[0] else ""),
                cfg.REGIME_LABELS[regime],
                str(row.get("n_scenes", "")),
                tables.num(row.get("psnr_srgb_mean"), 2),
                tables.num(row.get("ssim_srgb_mean"), 4),
                tables.num(row.get("lpips_mean"), 4),
                tables.num(row.get("train_active_s_mean"), 1),
            ])
    body = tables.latex_table(
        caption=(rf"Per-dataset-family behaviour at $\rho={fraction}$, averaged "
                 rf"over scenes (each scene averaged over seeds first, so every "
                 rf"scene is weighted equally)."),
        label="tab:stbn_family",
        column_spec="llrrrrr",
        header=["Family", "Supervision", "$n$", "PSNR", "SSIM", "LPIPS",
                "Active (s)"],
        rows=rows,
        midrule_before=[i for i in range(4, len(rows), 4)],
        source="tables/main_25pct_dataset_summary.csv")
    tables.write_latex(latex_dir / "table_dataset_family.tex", body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(results: ResultSet, out: dict, *, scenes=cfg.SCENES, seeds=cfg.SEEDS,
        fraction: float = cfg.PRIMARY_FRACTION) -> dict:
    """Emit every Experiment A artifact and return its findings."""
    tables_dir, figures_dir = out["tables"], out["figures"]
    latex_dir, raw_dir = out["latex"], out["raw"]

    cells = [r for r in results.select(fraction=None)
             if r.cell.seed in set(seeds) and r.cell.scene in set(scenes)
             and (abs(r.cell.fraction - fraction) < 1e-9
                  or r.cell.regime == "full")]
    raw_rows = results.rows(cells)
    tables.write_csv(tables_dir / "main_25pct_raw.csv", raw_rows, RAW_COLUMNS)

    view_rows = per_view_rows(cells)
    tables.write_csv(raw_dir / "main_25pct_per_test_view.csv", view_rows,
                     PER_VIEW_COLUMNS)

    scene_rows = _scene_rows(results, scenes, seeds, fraction)
    tables.write_csv(tables_dir / "main_25pct_scene_summary.csv", scene_rows,
                     SCENE_SUMMARY_COLUMNS)

    dataset_rows = _dataset_rows(results, scenes, seeds, fraction)
    tables.write_csv(tables_dir / "main_25pct_dataset_summary.csv", dataset_rows,
                     DATASET_SUMMARY_COLUMNS)

    delta_rows, delta_scene_rows, objects = _paired(results, scenes, seeds,
                                                    fraction)
    tables.write_csv(tables_dir / "main_25pct_paired_deltas.csv", delta_rows,
                     DELTA_COLUMNS)
    tables.write_csv(tables_dir / "main_25pct_paired_deltas_by_scene.csv",
                     delta_scene_rows, DELTA_BY_SCENE_COLUMNS)

    _figure_per_scene_bars(results, scenes, seeds, fraction, figures_dir)
    _figure_paired_deltas(objects, figures_dir)

    scenes_measured = len({r["scene"] for r in scene_rows}) or len(scenes)
    _latex_main(dataset_rows, latex_dir, fraction, seeds)
    _latex_ablation(delta_rows, latex_dir, fraction, scenes_measured)
    _latex_family(dataset_rows, latex_dir, fraction)

    validation = timing_validation(results, scenes, fraction)

    all_scenes_rows = {r["regime"]: r for r in dataset_rows
                       if r["grouping"] == "all" and r["group"] == "all_scenes"}
    return {
        "n_cells": len(cells),
        "n_test_views": len(view_rows),
        "fraction": fraction,
        "seeds": list(seeds),
        "aggregate_all_scenes": all_scenes_rows,
        "paired": {f"{r['contrast']}::{r['metric']}": r for r in delta_rows},
        "timing_validation": validation,
        "artifacts": {
            "raw_csv": "tables/main_25pct_raw.csv",
            "scene_summary_csv": "tables/main_25pct_scene_summary.csv",
            "dataset_summary_csv": "tables/main_25pct_dataset_summary.csv",
            "paired_deltas_csv": "tables/main_25pct_paired_deltas.csv",
            "paired_deltas_by_scene_csv":
                "tables/main_25pct_paired_deltas_by_scene.csv",
            "per_view_csv": "raw/main_25pct_per_test_view.csv",
            "figures": ["figures/expA_per_scene_regimes.pdf",
                        "figures/expA_paired_deltas.pdf"],
            "latex": ["latex/table_main_quantitative.tex",
                      "latex/table_spatial_vs_temporal.tex",
                      "latex/table_dataset_family.tex"],
        },
    }
