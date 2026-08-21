"""Experiment S — does forcing exactly k pixels per 8x8 execution tile matter?

The sparse mask can be built two ways, and they are not the same sampling
process:

    global      threshold the WHOLE 128x128 slice, keep its lowest rho fraction.
                One threshold, so the surviving points are the blue-noise point
                set NVIDIA's spatial energy function was optimized for. Per-tile
                counts are then free to vary (measured 16 +/- 1.4 at rho=0.25).

    tile_rank   rank within each 8x8 tile and keep exactly k per tile. Uniform
                per-tile work, but the effective threshold now varies from tile to
                tile, which lets pixels on either side of a tile boundary both
                survive and clump — the clustering the global ranking avoids.

The execution tiling is an implementation detail of how GPU threads process
pixels; making the sampling DISTRIBUTION respect those boundaries imposes a
stratification constraint that was not in the asset. Measured on the mask alone
(Experiment E), per-tile ranking costs ~30% of the blue mask's low-frequency
suppression AND makes the white-noise control ~24% bluer than true white, so it
compresses the blue-vs-white contrast from both ends.

This experiment asks whether that spectral damage shows up in trained quality.
It is a paired comparison on identical (scene, seed, rho) cells where the ONLY
difference is the selection mode, so the answer is attributable. The dense
baseline is shared — a selection mode does not apply to dense supervision.

Reported for all three sparse regimes, because the prediction is not uniform: the
white control should IMPROVE under tile_rank (it is being stratified for free),
while the two blue conditions should DEGRADE. If that pattern holds, it is direct
evidence that the per-tile constraint was flattering the control.
"""
from __future__ import annotations

from pathlib import Path

from . import config as cfg
from . import figures as figs
from . import stats
from . import tables
from .results import RAW_COLUMNS, ResultSet

METRICS = ("psnr_srgb", "ssim_srgb", "lpips", "train_active_s",
           "n_gaussians")

DELTA_COLUMNS = (
    "regime", "metric", "treatment", "baseline", "n_scenes", "n_cells",
    "mean_delta", "std_delta_scenes", "min_delta_scene", "max_delta_scene",
    "ci_low", "ci_high", "ci_excludes_zero", "sign_test_p",
    "wins", "losses", "ties", "win_rate", "higher_is_better",
    "unpaired_dropped",
)

BY_SCENE_COLUMNS = (
    "regime", "metric", "scene", "dataset_family", "seed", "delta",
    "global_value", "tile_rank_value",
)

SUMMARY_COLUMNS = (
    "regime", "select_mode", "n_scenes", "psnr_srgb_mean", "psnr_srgb_std",
    "ssim_srgb_mean", "lpips_mean", "train_active_s_mean",
    "n_gaussians_mean",
)


def _summary_rows(results: ResultSet, scenes, fraction, seed) -> list[dict]:
    rows: list[dict] = []
    for regime in cfg.SPARSE_REGIMES:
        for mode in (cfg.SELECT_GLOBAL, cfg.SELECT_TILE_RANK):
            cells = [r for r in results.select(regime=regime, fraction=fraction,
                                               seed=seed, select_mode=mode)
                     if r.cell.scene in set(scenes)]
            if not cells:
                continue
            row: dict = {"regime": regime, "select_mode": mode,
                         "n_scenes": len({c.cell.scene for c in cells})}
            for metric in METRICS:
                summary = stats.mean_std([c.metric(metric) for c in cells])
                row[f"{metric}_mean"] = summary["mean"]
                row[f"{metric}_std"] = summary["std"]
            rows.append(row)
    return rows


def _paired(results: ResultSet, scenes, fraction, seed
            ) -> tuple[list[dict], list[dict]]:
    """global minus tile_rank, per regime and metric."""
    summary_rows: list[dict] = []
    scene_rows: list[dict] = []
    for regime in cfg.SPARSE_REGIMES:
        for metric in METRICS:
            treat = results.metric_map(metric, regime=regime, fraction=fraction,
                                       seeds=(seed,), scenes=scenes,
                                       select_mode=cfg.SELECT_GLOBAL)
            base = results.metric_map(metric, regime=regime, fraction=fraction,
                                      seeds=(seed,), scenes=scenes,
                                      select_mode=cfg.SELECT_TILE_RANK)
            if not treat or not base:
                continue
            result = stats.paired_summary(
                cfg.SELECT_GLOBAL, cfg.SELECT_TILE_RANK, metric,
                treatment_values=treat, baseline_values=base,
                resamples=cfg.BOOTSTRAP_RESAMPLES, seed=cfg.BOOTSTRAP_SEED)
            row = result.as_row()
            row["regime"] = regime
            summary_rows.append(row)
            for (scene, seed_value), delta in sorted(result.cell_deltas.items()):
                scene_rows.append({
                    "regime": regime, "metric": metric, "scene": scene,
                    "dataset_family": cfg.scene_family(scene),
                    "seed": seed_value, "delta": delta,
                    "global_value": treat.get((scene, seed_value)),
                    "tile_rank_value": base.get((scene, seed_value)),
                })
    return summary_rows, scene_rows


def _figure(delta_rows: list[dict], figures_dir: Path) -> None:
    plt = figs.pyplot()
    if plt is None:
        return
    panels = [("psnr_srgb", "$\\Delta$ PSNR (dB)"),
              ("lpips", "$\\Delta$ LPIPS"),
              ("train_active_s", "$\\Delta$ active train (s)")]
    # Rotated tick labels on three panels each with its own y-label; the extra
    # width is what lets tight_layout resolve the decorations instead of warning.
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(figs.FIG_WIDTH_DOUBLE * 1.15, 2.6))
    regimes = list(cfg.SPARSE_REGIMES)
    for ax, (metric, ylabel) in zip(axes, panels):
        drew = False
        for index, regime in enumerate(regimes):
            row = next((r for r in delta_rows if r["regime"] == regime
                        and r["metric"] == metric), None)
            if row is None or row.get("mean_delta") is None:
                continue
            drew = True
            style = figs.style_for(regime)
            ax.bar([index], [row["mean_delta"]], 0.6, color=style["color"],
                   label=style["label"])
            low, high = row.get("ci_low"), row.get("ci_high")
            if low is not None and high is not None:
                ax.plot([index, index], [low, high], color="#222222",
                        linewidth=1.4, solid_capstyle="butt", zorder=4)
        ax.axhline(0.0, color="#000000", linewidth=0.7, linestyle=":")
        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels([figs.style_for(r)["label"] for r in regimes],
                           rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        if not drew:
            figs.annotate_empty(ax)
    axes[0].set_title("Experiment S: global slice threshold $-$ per-tile rank "
                      "(bar = mean over scenes, line = 95% bootstrap CI)",
                      loc="left")
    fig.tight_layout()
    figs.save(fig, figures_dir, "expS_stratification_ablation")


def _latex(summary_rows: list[dict], delta_rows: list[dict],
           latex_dir: Path, fraction: float) -> None:
    rows: list[list[str]] = []
    for regime in cfg.SPARSE_REGIMES:
        for mode in (cfg.SELECT_GLOBAL, cfg.SELECT_TILE_RANK):
            row = next((r for r in summary_rows if r["regime"] == regime
                        and r["select_mode"] == mode), None)
            if row is None:
                continue
            rows.append([
                (cfg.REGIME_LABELS[regime] if mode == cfg.SELECT_GLOBAL else ""),
                cfg.SELECT_MODE_LABELS[mode],
                tables.num(row.get("psnr_srgb_mean"), 3),
                tables.num(row.get("ssim_srgb_mean"), 4),
                tables.num(row.get("lpips_mean"), 4),
                tables.num(row.get("train_active_s_mean"), 1),
            ])
        delta = next((r for r in delta_rows if r["regime"] == regime
                      and r["metric"] == "psnr_srgb"), None)
        if delta is not None:
            rows.append([
                "", r"\quad$\Delta$ (global $-$ per-tile)",
                tables.num(delta.get("mean_delta"), 3, plus=True),
                tables.ci_cell(delta, 3),
                f"{delta.get('wins', 0)}/{delta.get('n_cells', 0)}",
                tables.num(delta.get("sign_test_p"), 3),
            ])
    body = tables.latex_table(
        caption=(rf"Selection-mode ablation at $\rho={fraction}$, seed "
                 rf"{cfg.PRIMARY_SEED}. A global slice threshold keeps the "
                 rf"lowest $\rho$ fraction of each whole $128{{\times}}128$ "
                 rf"slice — the point set the asset was optimized for — while "
                 rf"per-tile ranking keeps exactly $k$ of every "
                 rf"$8{{\times}}8$ execution tile, imposing a stratification the "
                 rf"asset does not have. On the mask alone, per-tile ranking "
                 rf"costs $\sim$30\% of the blue mask's low-frequency "
                 rf"suppression and makes the white control $\sim$24\% bluer "
                 rf"than true white."),
        label="tab:stbn_stratification",
        column_spec="llrrrr",
        header=["Supervision", "Selection", "PSNR", "SSIM", "LPIPS",
                "Active (s)"],
        rows=rows,
        midrule_before=[i for i in range(3, len(rows), 3)],
        notes=(r"The $\Delta$ rows re-use the SSIM/LPIPS columns for the 95\% "
               r"bootstrap CI, the win count and the sign $p$ of the PSNR "
               r"difference; per-metric deltas are in "
               r"\texttt{tables/stratification\_deltas.csv}."),
        source="tables/stratification_summary.csv")
    tables.write_latex(latex_dir / "table_stratification_ablation.tex", body)


def run(results: ResultSet, out: dict, *, scenes=cfg.SCENES,
        fraction: float = cfg.PRIMARY_FRACTION,
        seed: int = cfg.PRIMARY_SEED) -> dict:
    tables_dir, figures_dir = out["tables"], out["figures"]

    cells = [r for r in results.select(seed=seed, fraction=fraction,
                                       select_mode=None)
             if r.cell.scene in set(scenes) and r.cell.regime != "full"]
    if not cells:
        return {"status": "no cells for the stratification ablation"}
    tables.write_csv(out["raw"] / "stratification_raw.csv",
                     results.rows(cells), RAW_COLUMNS)

    summary_rows = _summary_rows(results, scenes, fraction, seed)
    tables.write_csv(tables_dir / "stratification_summary.csv", summary_rows,
                     SUMMARY_COLUMNS)

    delta_rows, scene_rows = _paired(results, scenes, fraction, seed)
    if not delta_rows:
        return {"status": "only one selection mode completed; nothing to pair",
                "summary": summary_rows}
    tables.write_csv(tables_dir / "stratification_deltas.csv", delta_rows,
                     DELTA_COLUMNS)
    tables.write_csv(tables_dir / "stratification_deltas_by_scene.csv",
                     scene_rows, BY_SCENE_COLUMNS)

    _figure(delta_rows, figures_dir)
    _latex(summary_rows, delta_rows, out["latex"], fraction)

    # The predicted pattern: the blue conditions should IMPROVE under a global
    # threshold, while the white control should get WORSE (per-tile ranking was
    # stratifying it for free). Stated as a checkable claim, not an assumption.
    psnr = {r["regime"]: r for r in delta_rows if r["metric"] == "psnr_srgb"}
    pattern = {
        regime: {
            "mean_delta_psnr": (psnr.get(regime) or {}).get("mean_delta"),
            "ci_excludes_zero": (psnr.get(regime) or {}).get("ci_excludes_zero"),
            "wins": (psnr.get(regime) or {}).get("wins"),
            "n_cells": (psnr.get(regime) or {}).get("n_cells"),
        }
        for regime in cfg.SPARSE_REGIMES
    }
    blue_gain = [pattern[r]["mean_delta_psnr"] for r in
                 ("indep_blue", "stbn")
                 if pattern[r]["mean_delta_psnr"] is not None]
    white_gain = pattern.get("iid_white", {}).get("mean_delta_psnr")
    verdict = "insufficient data"
    if blue_gain and white_gain is not None:
        if all(g > 0 for g in blue_gain) and white_gain < 0:
            verdict = ("as predicted: a global threshold helps both blue "
                       "conditions and hurts the white control, i.e. per-tile "
                       "ranking was flattering the control and damaging the "
                       "blue masks")
        elif all(g > 0 for g in blue_gain):
            verdict = ("a global threshold helps the blue conditions; the white "
                       "control moved the same way, so the stratification is not "
                       "simply flattering the control")
        else:
            verdict = ("NOT as predicted: the trained-quality effect does not "
                       "follow the spectral one — report this, it bounds how much "
                       "the mask spectrum explains")
    return {
        "status": "ok",
        "fraction": fraction,
        "seed": seed,
        "n_cells": len(cells),
        "summary": summary_rows,
        "deltas": {f"{r['regime']}::{r['metric']}": r for r in delta_rows},
        "pattern": pattern,
        "verdict": verdict,
        "artifacts": {
            "raw_csv": "raw/stratification_raw.csv",
            "summary_csv": "tables/stratification_summary.csv",
            "deltas_csv": "tables/stratification_deltas.csv",
            "deltas_by_scene_csv": "tables/stratification_deltas_by_scene.csv",
            "figures": ["figures/expS_stratification_ablation.pdf"],
            "latex": ["latex/table_stratification_ablation.tex"],
        },
    }
