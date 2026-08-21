"""Experiment C — convergence: quality against ITERATIONS and against TIME.

The seed-42 rho=0.25 cells are evaluated on the held-out split at 250, 500, 750,
1000, 1500, 2000, 2500 and 3000 iterations. Two things make this measurement
trustworthy rather than circular:

**Evaluation does not count as training.** Each sweep renders and scores every
held-out view, which takes longer than the training it interrupts. The trainer
accumulates that cost and reports ``elapsed_s`` per checkpoint with it already
subtracted, so the x-axis of every time plot here is ACTIVE training seconds. A
run that was measured more often does not appear slower.

**Evaluation does not perturb training.** The sweep only renders; it advances no
optimizer step, draws no camera from the sampler, and consumes no mask slice
(the slice index follows each camera's own visit counter, which only
``_next_iteration`` touches). This was verified empirically: the same cell run
with and without the checkpoint ladder produced identical held-out PSNR and an
identical final Gaussian count.

The time-to-quality analysis targets the DENSE final PSNR minus 0.5/0.2/0.1 dB,
per scene. Two rules keep it honest: interpolation only ever happens BETWEEN
measured checkpoints, and a trajectory that never reaches a target is recorded as
``not_reached`` with its shortfall — never extrapolated to a number that would be
the single most quotable figure in the section.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as cfg
from . import figures as figs
from . import stats
from . import tables
from .results import ResultSet

CURVE_COLUMNS = (
    "scene", "dataset_family", "regime", "seed", "pixel_fraction", "iter",
    "active_train_s", "psnr_srgb", "ssim_srgb", "lpips", "n_gaussians",
    "n_eval_cams", "eval_dt_s", "cell_key",
)

AGGREGATE_COLUMNS = (
    "regime", "iter", "n_scenes", "scenes_expected", "complete",
    "active_train_s_mean", "active_train_s_std",
    "psnr_srgb_mean", "psnr_srgb_std",
    "ssim_srgb_mean", "lpips_mean", "lpips_std",
    "n_gaussians_mean", "n_gaussians_std",
)

TTQ_COLUMNS = (
    "scene", "dataset_family", "regime", "target_db_below_dense",
    "target_psnr", "dense_final_psnr", "status", "seconds", "iteration",
    "interpolated", "best_value", "shortfall_db",
)

TTQ_SUMMARY_COLUMNS = (
    "regime", "target_db_below_dense", "n_scenes_reached", "n_scenes_total",
    "reach_rate", "seconds_mean", "seconds_std", "seconds_min", "seconds_max",
    "speedup_vs_dense_mean", "not_reached_scenes",
)


def _curve_rows(results: ResultSet, scenes, fraction, seed) -> list[dict]:
    """Flatten every checkpoint of every convergence cell.

    Only the ladder points the experiment asked for are emitted, so a run that
    also happened to evaluate elsewhere cannot quietly add points to a curve that
    other regimes lack.
    """
    wanted = set(cfg.CONVERGENCE_ITERS)
    rows: list[dict] = []
    for regime in cfg.REGIMES:
        use_fraction = 1.0 if regime == "full" else fraction
        for result in results.select(regime=regime, fraction=use_fraction,
                                     seed=seed, convergence=True):
            if result.cell.scene not in set(scenes):
                continue
            for point in result.curve:
                iteration = int(point.get("iter", 0))
                if iteration not in wanted:
                    continue
                rows.append({
                    "scene": result.cell.scene,
                    "dataset_family": result.cell.family,
                    "regime": regime,
                    "seed": seed,
                    "pixel_fraction": use_fraction,
                    "iter": iteration,
                    # Already eval-excluded by the trainer.
                    "active_train_s": point.get("elapsed_s"),
                    "psnr_srgb": point.get("psnr_srgb"),
                    "ssim_srgb": point.get("ssim_srgb"),
                    "lpips": point.get("lpips"),
                    "n_gaussians": point.get("n_gaussians"),
                    "n_eval_cams": point.get("n_eval_cams"),
                    "eval_dt_s": point.get("eval_dt_s"),
                    "cell_key": result.cell.key,
                })
    return rows


def _aggregate(curve_rows: list[dict], scenes) -> list[dict]:
    """Mean trajectory per regime, averaged across scenes at each checkpoint."""
    out: list[dict] = []
    for regime in cfg.REGIMES:
        for iteration in cfg.CONVERGENCE_ITERS:
            picked = [r for r in curve_rows
                      if r["regime"] == regime and r["iter"] == iteration]
            if not picked:
                continue
            present = {r["scene"] for r in picked}
            row: dict = {
                "regime": regime, "iter": iteration,
                "n_scenes": len(present), "scenes_expected": len(scenes),
                "complete": len(present) == len(scenes),
            }
            for key in ("active_train_s", "psnr_srgb", "ssim_srgb",
                        "lpips", "n_gaussians"):
                summary = stats.mean_std([r.get(key) for r in picked])
                row[f"{key}_mean"] = summary["mean"]
                row[f"{key}_std"] = summary["std"]
            out.append(row)
    return out


def _time_to_quality(results: ResultSet, curve_rows: list[dict], scenes,
                     fraction, seed) -> tuple[list[dict], list[dict]]:
    """Per-scene time to reach the dense final PSNR minus each target offset."""
    dense_final: dict[str, float] = {}
    for result in results.select(regime="full", fraction=1.0, seed=seed,
                                 convergence=True):
        value = result.metric("psnr_srgb")
        if value is not None:
            dense_final[result.cell.scene] = float(value)

    rows: list[dict] = []
    for scene in scenes:
        reference = dense_final.get(scene)
        if reference is None:
            continue
        for regime in cfg.REGIMES:
            trajectory = sorted(
                (r for r in curve_rows
                 if r["scene"] == scene and r["regime"] == regime),
                key=lambda r: r["iter"])
            if not trajectory:
                continue
            for offset in cfg.QUALITY_TARGETS_DB:
                target = reference - offset
                outcome = stats.time_to_target(
                    [r["iter"] for r in trajectory],
                    [r["active_train_s"] for r in trajectory],
                    [r["psnr_srgb"] for r in trajectory],
                    target, higher_is_better=True)
                rows.append({
                    "scene": scene,
                    "dataset_family": cfg.scene_family(scene),
                    "regime": regime,
                    "target_db_below_dense": offset,
                    "target_psnr": target,
                    "dense_final_psnr": reference,
                    "status": outcome.get("status"),
                    "seconds": outcome.get("seconds"),
                    "iteration": outcome.get("iteration"),
                    "interpolated": outcome.get("interpolated"),
                    "best_value": outcome.get("best_value"),
                    "shortfall_db": outcome.get("shortfall"),
                })

    summary: list[dict] = []
    for regime in cfg.REGIMES:
        for offset in cfg.QUALITY_TARGETS_DB:
            picked = [r for r in rows if r["regime"] == regime
                      and abs(r["target_db_below_dense"] - offset) < 1e-9]
            if not picked:
                continue
            reached = [r for r in picked
                       if r["status"] in ("reached", "reached_at_first_checkpoint")]
            seconds = stats.mean_std([r["seconds"] for r in reached])
            # Speedup only over scenes where BOTH this regime and dense reached
            # the target — a ratio against a scene dense never reached would be
            # undefined, and dropping it silently would bias the mean.
            dense_reached = {
                r["scene"]: r["seconds"] for r in rows
                if r["regime"] == "full"
                and abs(r["target_db_below_dense"] - offset) < 1e-9
                and r["status"] in ("reached", "reached_at_first_checkpoint")
                and r["seconds"]}
            ratios = [dense_reached[r["scene"]] / r["seconds"]
                      for r in reached
                      if r["scene"] in dense_reached and r["seconds"]]
            summary.append({
                "regime": regime,
                "target_db_below_dense": offset,
                "n_scenes_reached": len(reached),
                "n_scenes_total": len(picked),
                "reach_rate": (len(reached) / len(picked)) if picked else None,
                "seconds_mean": seconds["mean"],
                "seconds_std": seconds["std"],
                "seconds_min": seconds["min"],
                "seconds_max": seconds["max"],
                "speedup_vs_dense_mean": stats.mean_std(ratios)["mean"],
                "not_reached_scenes": ";".join(sorted(
                    r["scene"] for r in picked if r["status"] == "not_reached")),
            })
    return rows, summary


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _figure_aggregate(aggregate: list[dict], figures_dir: Path,
                      fraction: float) -> None:
    plt = figs.pyplot()
    if plt is None:
        return
    panels = [("active_train_s_mean", "psnr_srgb_mean",
               "active train time (s)", "held-out PSNR (dB)"),
              ("active_train_s_mean", "lpips_mean",
               "active train time (s)", "held-out LPIPS"),
              ("iter", "psnr_srgb_mean", "iteration", "held-out PSNR (dB)"),
              ("active_train_s_mean", "n_gaussians_mean",
               "active train time (s)", "Gaussians")]
    # Four panels with independent y-labels and scientific-notation ticks; the
    # extra width is what lets tight_layout actually resolve the decorations.
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(figs.FIG_WIDTH_DOUBLE * 1.25, 2.3))
    for ax, (xkey, ykey, xlabel, ylabel) in zip(axes, panels):
        drew = False
        for regime in cfg.REGIMES:
            style = figs.style_for(regime)
            points = sorted((r for r in aggregate if r["regime"] == regime),
                            key=lambda r: r["iter"])
            xs = [r.get(xkey) for r in points]
            ys = [r.get(ykey) for r in points]
            pairs = [(x, y) for x, y in zip(xs, ys)
                     if x is not None and y is not None]
            if not pairs:
                continue
            drew = True
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], linewidth=1.2, markersize=3.5,
                    label=style["label"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ykey == "n_gaussians_mean":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        if not drew:
            figs.annotate_empty(ax)
    axes[0].set_title(f"Experiment C: convergence at $\\rho$={fraction}, "
                      f"seed {cfg.PRIMARY_SEED}, mean over scenes "
                      f"(x-axis excludes evaluation time)", loc="left")
    axes[0].legend()
    fig.tight_layout()
    figs.save(fig, figures_dir, "expC_convergence_aggregate")


def _figure_per_scene(curve_rows: list[dict], scenes, figures_dir: Path) -> None:
    """A small-multiples grid: one panel per scene, PSNR against active time."""
    plt = figs.pyplot()
    if plt is None:
        return
    present = [s for s in scenes if any(r["scene"] == s for r in curve_rows)]
    if not present:
        return
    columns = 5
    rows_n = int(np.ceil(len(present) / columns))
    fig, axes = plt.subplots(rows_n, columns,
                             figsize=(figs.FIG_WIDTH_DOUBLE,
                                      1.75 * rows_n), squeeze=False)
    for index, scene in enumerate(present):
        ax = axes[index // columns][index % columns]
        finite: list[float] = []
        for regime in cfg.REGIMES:
            style = figs.style_for(regime)
            points = sorted((r for r in curve_rows if r["scene"] == scene
                             and r["regime"] == regime),
                            key=lambda r: r["iter"])
            pairs = [(r["active_train_s"], r["psnr_srgb"]) for r in points
                     if r["active_train_s"] is not None
                     and r["psnr_srgb"] is not None]
            if not pairs:
                continue
            finite += [p[1] for p in pairs]
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs],
                    color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], linewidth=1.0, markersize=2.5,
                    label=style["label"])
        ax.set_title(scene, fontsize=7)
        figs.zoom_to_data(ax, finite, min_pad=0.1)
        ax.tick_params(labelsize=6)
    for index in range(len(present), rows_n * columns):
        axes[index // columns][index % columns].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4,
                   bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("PSNR (dB) vs active training seconds, per scene", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    figs.save(fig, figures_dir, "expC_convergence_per_scene")


def _figure_time_to_quality(summary: list[dict], figures_dir: Path) -> None:
    plt = figs.pyplot()
    if plt is None:
        return
    fig, axes = plt.subplots(1, len(cfg.QUALITY_TARGETS_DB),
                             figsize=(figs.FIG_WIDTH_DOUBLE * 0.8, 2.2),
                             squeeze=False)
    for column, offset in enumerate(cfg.QUALITY_TARGETS_DB):
        ax = axes[0][column]
        labels, values, errors, colors, reach = [], [], [], [], []
        for regime in cfg.REGIMES:
            row = next((r for r in summary if r["regime"] == regime
                        and abs(r["target_db_below_dense"] - offset) < 1e-9), None)
            if row is None:
                continue
            labels.append(figs.style_for(regime)["label"])
            values.append(row.get("seconds_mean") or np.nan)
            errors.append(row.get("seconds_std") or 0.0)
            colors.append(figs.style_for(regime)["color"])
            reach.append(row.get("reach_rate"))
        if not labels:
            figs.annotate_empty(ax)
            continue
        positions = np.arange(len(labels))
        ax.bar(positions, values, 0.6, yerr=errors, capsize=2, color=colors,
               error_kw={"linewidth": 0.6})
        # Reach rate printed on each bar: a mean over the scenes that made it is
        # meaningless without knowing how many did.
        for pos, value, rate in zip(positions, values, reach):
            if rate is not None and value == value:
                ax.text(pos, value, f"{rate:.0%}", ha="center", va="bottom",
                        fontsize=6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("active seconds to target" if column == 0 else "")
        ax.set_title(f"dense final $-$ {offset} dB", fontsize=8)
    fig.suptitle("Time to reach dense-final quality (label = fraction of scenes "
                 "that reached it)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    figs.save(fig, figures_dir, "expC_time_to_quality")


def run(results: ResultSet, out: dict, *, scenes=cfg.SCENES,
        fraction: float = cfg.PRIMARY_FRACTION,
        seed: int = cfg.PRIMARY_SEED) -> dict:
    tables_dir, figures_dir = out["tables"], out["figures"]

    curve_rows = _curve_rows(results, scenes, fraction, seed)
    tables.write_csv(out["raw"] / "convergence_curves.csv", curve_rows,
                     CURVE_COLUMNS)

    aggregate = _aggregate(curve_rows, scenes)
    tables.write_csv(tables_dir / "convergence_aggregate.csv", aggregate,
                     AGGREGATE_COLUMNS)

    ttq_rows, ttq_summary = _time_to_quality(results, curve_rows, scenes,
                                             fraction, seed)
    tables.write_csv(tables_dir / "convergence_time_to_quality.csv", ttq_rows,
                     TTQ_COLUMNS)
    tables.write_csv(tables_dir / "convergence_time_to_quality_summary.csv",
                     ttq_summary, TTQ_SUMMARY_COLUMNS)

    _figure_aggregate(aggregate, figures_dir, fraction)
    _figure_per_scene(curve_rows, scenes, figures_dir)
    _figure_time_to_quality(ttq_summary, figures_dir)

    not_reached = {
        f"{r['regime']}@{r['target_db_below_dense']}": r["not_reached_scenes"]
        for r in ttq_summary if r["not_reached_scenes"]}
    return {
        "n_checkpoints": len(curve_rows),
        "checkpoint_iters": list(cfg.CONVERGENCE_ITERS),
        "n_scenes": len({r["scene"] for r in curve_rows}),
        "aggregate": aggregate,
        "time_to_quality": ttq_summary,
        "not_reached": not_reached,
        "artifacts": {
            "curves_csv": "raw/convergence_curves.csv",
            "aggregate_csv": "tables/convergence_aggregate.csv",
            "time_to_quality_csv": "tables/convergence_time_to_quality.csv",
            "time_to_quality_summary_csv":
                "tables/convergence_time_to_quality_summary.csv",
            "figures": ["figures/expC_convergence_aggregate.pdf",
                        "figures/expC_convergence_per_scene.pdf",
                        "figures/expC_time_to_quality.pdf"],
        },
    }
