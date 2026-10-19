# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Turn the manifest into the training tables and figures the report cites.

Pure aggregation — reads ``manifest.json`` and the run directories, writes CSVs and
figures, never touches the GPU. Safe to re-run at any point; a stage that has not
finished simply produces fewer rows, and every aggregate carries its ``n`` so a
mean over three scenes can never be mistaken for a mean over five.

Two rules run through all of it.

**Missing is not zero.** An absent metric is an empty CSV field, never a 0.0. A
zero in a PSNR column reads as a catastrophic result rather than as no result.

**Paired comparisons, not pooled means.** Scenes differ by more than the regimes
do (bicycle and kitchen are 2 dB apart), so every contrast is computed as a
per-scene DELTA and then averaged. Averaging the two regimes' absolute PSNRs across
scenes and subtracting would let a single missing scene move the answer by more
than the effect.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from paper_suite import figures as figs
from paper_suite import tables

from . import config as cfg
from .manifest import Manifest

# Section 13's required per-run report, in order.
CELL_COLUMNS = (
    "scene", "dataset_family", "regime", "seed", "pixel_fraction", "adaptive",
    "alpha", "sigma", "importance_downsample", "uniform_mode", "unbiased",
    "iters", "n_gaussians",
    "psnr_srgb", "ssim_srgb", "lpips",
    "backward_raster_ms", "loss_ms", "adaptive_sampling_ms",
    "forward_raster_ms", "stage_total_ms",
    "train_active_s", "train_wall_s", "process_wall_s",
    "selected_fraction", "p_mean", "p_min", "p_max", "saturated_fraction",
    "lambda_mean", "budget_error_max",
    "gpu_device_peak_mb", "gpu_alloc_capacity_mb",
    "width", "height", "n_train", "n_test", "status", "config_hash",
)
SUMMARY_COLUMNS = (
    "regime", "pixel_fraction", "alpha", "sigma", "n_scenes", "complete",
    "psnr_mean", "psnr_std", "ssim_mean", "lpips_mean",
    "backward_raster_ms_mean", "loss_ms_mean", "adaptive_sampling_ms_mean",
    "train_active_s_mean", "n_gaussians_mean", "selected_fraction_mean",
    "p_min_min", "p_max_max", "saturated_fraction_mean",
    "gpu_device_peak_mb_mean",
    "backward_speedup_vs_dense", "psnr_delta_vs_dense",
)
CONTRAST_COLUMNS = (
    "treatment", "control", "axis", "pixel_fraction", "n_scenes", "scenes",
    "psnr_delta_mean", "psnr_delta_std", "psnr_delta_min", "psnr_delta_max",
    "ssim_delta_mean", "lpips_delta_mean", "backward_raster_ms_delta_mean",
    "loss_ms_delta_mean", "train_active_s_delta_mean", "wins", "losses",
)
HYPER_COLUMNS = (
    "parameter", "value", "scene", "psnr_srgb", "ssim_srgb", "lpips",
    "backward_raster_ms", "loss_ms", "adaptive_sampling_ms", "n_gaussians",
    "selected_fraction", "p_min", "p_max", "saturated_fraction",
    "psnr_delta_vs_uniform_stbn",
)
FRACTION_COLUMNS = (
    "regime", "pixel_fraction", "rho_enforced", "n_scenes", "psnr_mean",
    "psnr_std", "ssim_mean", "lpips_mean", "backward_raster_ms_mean",
    "backward_speedup_vs_dense", "train_active_s_mean", "n_gaussians_mean",
)


def _enforced(fraction: float) -> float:
    from trainer.backend.render.stbn import pixels_per_tile
    from trainer.contracts.codegen import TILE_AREA
    if fraction >= 1.0:
        return 1.0
    return pixels_per_tile(fraction, TILE_AREA) / TILE_AREA


def _row(record) -> dict:
    return {c: getattr(record, c, None) for c in CELL_COLUMNS}


def _mean(values) -> float | None:
    clean = [v for v in values if v is not None and v == v]
    return float(np.mean(clean)) if clean else None


def _std(values) -> float | None:
    clean = [v for v in values if v is not None and v == v]
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else (
        0.0 if clean else None)


def _select(records, **filters):
    out = []
    for record in records:
        if all(getattr(record, key, None) == value
               for key, value in filters.items()):
            out.append(record)
    return out


def _matches(record, regime: str, fraction: float, *, alpha: float | None = None,
             sigma: float | None = None) -> bool:
    if record.regime != regime:
        return False
    if abs((record.pixel_fraction or 0.0) - fraction) > 1e-9:
        return False
    if record.adaptive:
        if alpha is not None and abs((record.alpha or 0.0) - alpha) > 1e-9:
            return False
        if sigma is not None and abs((record.sigma or 0.0) - sigma) > 1e-9:
            return False
    return True


def collect(manifest: Manifest) -> list:
    """Every usable cell record, newest manifest state."""
    return [r for r in manifest.records.values() if r.usable]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def pilot_tables(paths: dict, records: list, scenes: tuple[str, ...]) -> dict:
    """Section 13: the per-cell table and the per-regime aggregate."""
    out = paths["training"]
    fraction = cfg.PRIMARY_FRACTION

    def pick(regime: str):
        target = 1.0 if regime == cfg.REGIME_FULL else fraction
        return {r.scene: r for r in records
                if _matches(r, regime, target, alpha=cfg.PRIMARY_ALPHA,
                            sigma=cfg.PRIMARY_SIGMA) and r.scene in scenes}

    by_regime = {regime: pick(regime) for regime in cfg.ALL_REGIMES}
    cell_rows = [_row(r) for regime in cfg.ALL_REGIMES
                 for _, r in sorted(by_regime[regime].items())]
    tables.write_csv(out / "pilot_cells.csv", cell_rows, CELL_COLUMNS)

    dense = by_regime[cfg.REGIME_FULL]
    summary_rows: list[dict] = []
    for regime in cfg.ALL_REGIMES:
        found = by_regime[regime]
        if not found:
            continue
        cells = [found[s] for s in scenes if s in found]
        # Speedup and PSNR delta against dense are PAIRED per scene, then averaged;
        # a ratio of cross-scene means would be dominated by the largest scene.
        speedups, deltas = [], []
        for cell in cells:
            reference = dense.get(cell.scene)
            if reference is None:
                continue
            if reference.backward_raster_ms and cell.backward_raster_ms:
                speedups.append(reference.backward_raster_ms
                                / cell.backward_raster_ms)
            if reference.psnr_srgb is not None and cell.psnr_srgb is not None:
                deltas.append(cell.psnr_srgb - reference.psnr_srgb)
        summary_rows.append({
            "regime": regime,
            "pixel_fraction": (1.0 if regime == cfg.REGIME_FULL else fraction),
            "alpha": (cfg.PRIMARY_ALPHA if regime.startswith("adaptive") else None),
            "sigma": (cfg.PRIMARY_SIGMA if regime.startswith("adaptive") else None),
            "n_scenes": len(cells), "complete": len(cells) == len(scenes),
            "psnr_mean": _mean([c.psnr_srgb for c in cells]),
            "psnr_std": _std([c.psnr_srgb for c in cells]),
            "ssim_mean": _mean([c.ssim_srgb for c in cells]),
            "lpips_mean": _mean([c.lpips for c in cells]),
            "backward_raster_ms_mean": _mean([c.backward_raster_ms
                                              for c in cells]),
            "loss_ms_mean": _mean([c.loss_ms for c in cells]),
            "adaptive_sampling_ms_mean": _mean([c.adaptive_sampling_ms
                                                for c in cells]),
            "train_active_s_mean": _mean([c.train_active_s for c in cells]),
            "n_gaussians_mean": _mean([c.n_gaussians for c in cells]),
            "selected_fraction_mean": _mean([c.selected_fraction for c in cells]),
            "p_min_min": (min((c.p_min for c in cells if c.p_min is not None),
                              default=None)),
            "p_max_max": (max((c.p_max for c in cells if c.p_max is not None),
                              default=None)),
            "saturated_fraction_mean": _mean([c.saturated_fraction
                                              for c in cells]),
            "gpu_device_peak_mb_mean": _mean([c.gpu_device_peak_mb
                                              for c in cells]),
            "backward_speedup_vs_dense": _mean(speedups),
            "psnr_delta_vs_dense": _mean(deltas),
        })
    tables.write_csv(out / "pilot_summary.csv", summary_rows, SUMMARY_COLUMNS)

    contrast_rows = contrasts(by_regime, scenes, fraction)
    tables.write_csv(out / "pilot_contrasts.csv", contrast_rows, CONTRAST_COLUMNS)
    return {"cells": cell_rows, "summary": summary_rows,
            "contrasts": contrast_rows, "by_regime": by_regime}


def contrasts(by_regime: dict, scenes: tuple[str, ...],
              fraction: float) -> list[dict]:
    """Per-scene paired deltas for each controlled pair, then averaged.

    ``wins``/``losses`` count the scenes rather than only reporting a mean: five
    scenes is too few for a mean to be persuasive on its own, and "improved on 2 of
    5" is the honest summary of a mean that happens to be positive.
    """
    rows: list[dict] = []
    for treatment, control, axis in cfg.CONTRASTS:
        left, right = by_regime.get(treatment, {}), by_regime.get(control, {})
        shared = [s for s in scenes if s in left and s in right]
        if not shared:
            continue

        def deltas(attribute: str) -> list[float]:
            out = []
            for scene in shared:
                a = getattr(left[scene], attribute, None)
                b = getattr(right[scene], attribute, None)
                if a is not None and b is not None:
                    out.append(a - b)
            return out

        psnr = deltas("psnr_srgb")
        rows.append({
            "treatment": treatment, "control": control, "axis": axis,
            "pixel_fraction": fraction, "n_scenes": len(shared),
            "scenes": " ".join(shared),
            "psnr_delta_mean": _mean(psnr), "psnr_delta_std": _std(psnr),
            "psnr_delta_min": (min(psnr) if psnr else None),
            "psnr_delta_max": (max(psnr) if psnr else None),
            "ssim_delta_mean": _mean(deltas("ssim_srgb")),
            # LPIPS is a distance: NEGATIVE is better, unlike the other two.
            "lpips_delta_mean": _mean(deltas("lpips")),
            "backward_raster_ms_delta_mean": _mean(deltas("backward_raster_ms")),
            "loss_ms_delta_mean": _mean(deltas("loss_ms")),
            "train_active_s_delta_mean": _mean(deltas("train_active_s")),
            "wins": sum(1 for d in psnr if d > 0),
            "losses": sum(1 for d in psnr if d < 0),
        })
    return rows


def hyper_tables(paths: dict, records: list) -> dict:
    """Section 14: alpha and sigma robustness on two scenes only.

    Each point is reported as a DELTA against the uniform STBN cell of the same
    scene as well as in absolute terms — the two scenes are 4 dB apart, so an
    absolute column alone hides which direction the parameter moved things.
    """
    out = paths["training"]
    scenes = cfg.HYPER_SCENES
    baseline = {r.scene: r for r in records
                if _matches(r, "stbn", cfg.PRIMARY_FRACTION)}

    rows: list[dict] = []
    for parameter, values, fixed in (
            ("alpha", cfg.HYPER_ALPHAS, {"sigma": cfg.PRIMARY_SIGMA}),
            ("sigma", cfg.HYPER_SIGMAS, {"alpha": cfg.PRIMARY_ALPHA})):
        for value in values:
            kwargs = {parameter: value, **fixed}
            for scene in scenes:
                found = [r for r in records if r.scene == scene
                         and _matches(r, "adaptive_stbn", cfg.PRIMARY_FRACTION,
                                      **kwargs)]
                if not found:
                    continue
                cell = found[0]
                reference = baseline.get(scene)
                rows.append({
                    "parameter": parameter, "value": value, "scene": scene,
                    "psnr_srgb": cell.psnr_srgb, "ssim_srgb": cell.ssim_srgb,
                    "lpips": cell.lpips,
                    "backward_raster_ms": cell.backward_raster_ms,
                    "loss_ms": cell.loss_ms,
                    "adaptive_sampling_ms": cell.adaptive_sampling_ms,
                    "n_gaussians": cell.n_gaussians,
                    "selected_fraction": cell.selected_fraction,
                    "p_min": cell.p_min, "p_max": cell.p_max,
                    "saturated_fraction": cell.saturated_fraction,
                    "psnr_delta_vs_uniform_stbn": (
                        cell.psnr_srgb - reference.psnr_srgb
                        if reference is not None and reference.psnr_srgb is not None
                        and cell.psnr_srgb is not None else None),
                })
    tables.write_csv(out / "hyperparameters.csv", rows, HYPER_COLUMNS)
    return {"rows": rows}


def fraction_tables(paths: dict, records: list,
                    scenes: tuple[str, ...]) -> dict:
    """Section 17: the pixel-budget comparison, uniform STBN vs adaptive STBN."""
    out = paths["training"]
    dense = {r.scene: r for r in records
             if _matches(r, cfg.REGIME_FULL, 1.0)}
    rows: list[dict] = []
    for regime in cfg.FRACTION_REGIMES:
        for fraction in cfg.FRACTION_SWEEP:
            cells = [r for r in records if r.scene in scenes
                     and _matches(r, regime, fraction, alpha=cfg.PRIMARY_ALPHA,
                                  sigma=cfg.PRIMARY_SIGMA)]
            if not cells:
                continue
            speedups = []
            for cell in cells:
                reference = dense.get(cell.scene)
                if (reference is not None and reference.backward_raster_ms
                        and cell.backward_raster_ms):
                    speedups.append(reference.backward_raster_ms
                                    / cell.backward_raster_ms)
            rows.append({
                "regime": regime, "pixel_fraction": fraction,
                "rho_enforced": _enforced(fraction), "n_scenes": len(cells),
                "psnr_mean": _mean([c.psnr_srgb for c in cells]),
                "psnr_std": _std([c.psnr_srgb for c in cells]),
                "ssim_mean": _mean([c.ssim_srgb for c in cells]),
                "lpips_mean": _mean([c.lpips for c in cells]),
                "backward_raster_ms_mean": _mean([c.backward_raster_ms
                                                  for c in cells]),
                "backward_speedup_vs_dense": _mean(speedups),
                "train_active_s_mean": _mean([c.train_active_s for c in cells]),
                "n_gaussians_mean": _mean([c.n_gaussians for c in cells]),
            })
    tables.write_csv(out / "fraction_sweep.csv", rows, FRACTION_COLUMNS)
    return {"rows": rows, "equal_quality": _equal_quality(records, scenes)}


def _equal_quality(records: list, scenes: tuple[str, ...]) -> dict:
    """Does adaptive STBN at a LOWER rho match uniform STBN at 0.25?

    The interesting form of the result (design section 17): converted allocation,
    not just a higher number at the same budget. Reported per scene as well as
    pooled, because "on average yes" over five scenes where it fails on two is not
    the same finding.
    """
    reference = {r.scene: r for r in records
                 if _matches(r, "stbn", cfg.PRIMARY_FRACTION)}
    out: dict = {"target": "uniform STBN at rho=0.25", "per_scene": {}}
    for scene in scenes:
        target = reference.get(scene)
        if target is None or target.psnr_srgb is None:
            continue
        matches = []
        for fraction in sorted(cfg.FRACTION_SWEEP):
            cells = [r for r in records if r.scene == scene
                     and _matches(r, "adaptive_stbn", fraction,
                                  alpha=cfg.PRIMARY_ALPHA,
                                  sigma=cfg.PRIMARY_SIGMA)]
            if not cells or cells[0].psnr_srgb is None:
                continue
            matches.append({
                "pixel_fraction": fraction,
                "psnr_delta": cells[0].psnr_srgb - target.psnr_srgb,
                "backward_raster_ms": cells[0].backward_raster_ms,
            })
        cheapest = next((m for m in matches if m["psnr_delta"] >= -0.05), None)
        out["per_scene"][scene] = {
            "uniform_psnr_at_025": target.psnr_srgb,
            "uniform_backward_ms_at_025": target.backward_raster_ms,
            "adaptive_points": matches,
            "cheapest_matching_fraction": (cheapest or {}).get("pixel_fraction"),
            "criterion": "adaptive PSNR within 0.05 dB of uniform STBN at 0.25",
        }
    achieved = [v["cheapest_matching_fraction"]
                for v in out["per_scene"].values()
                if v["cheapest_matching_fraction"] is not None]
    out["n_scenes_matched_below_025"] = sum(1 for f in achieved if f < 0.25)
    out["n_scenes_evaluated"] = len(out["per_scene"])
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

REGIME_STYLE = {
    "full": {"color": "#000000", "marker": "o", "linestyle": "-"},
    "iid_white": {"color": "#E69F00", "marker": "s", "linestyle": "--"},
    "indep_blue": {"color": "#56B4E9", "marker": "^", "linestyle": "-."},
    "stbn": {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    "adaptive_iid": {"color": "#D55E00", "marker": "v", "linestyle": ":"},
    "adaptive_indep_blue": {"color": "#CC79A7", "marker": "P", "linestyle": "-."},
    "adaptive_stbn": {"color": "#009E73", "marker": "D", "linestyle": "-"},
    "adaptive_stbn_biased": {"color": "#666666", "marker": "X", "linestyle": ":"},
}


def figures(paths: dict, pilot: dict, hyper: dict, fractions: dict,
            scenes: tuple[str, ...]) -> list[Path]:
    plt = figs.pyplot()
    if plt is None:
        return []
    out = paths["figures"]
    written: list[Path] = []
    by_regime = pilot["by_regime"]

    # 1. Per-scene PSNR, grouped by regime. Per scene rather than pooled: the
    #    scene-to-scene spread is an order of magnitude larger than the effect, so
    #    a single pooled bar would say nothing.
    present = [r for r in cfg.ALL_REGIMES if by_regime.get(r)]
    if present and scenes:
        fig, ax = plt.subplots(figsize=(figs.FIG_WIDTH_DOUBLE, 2.6))
        width = 0.8 / max(len(present), 1)
        for index, regime in enumerate(present):
            values = [(by_regime[regime].get(s).psnr_srgb
                       if by_regime[regime].get(s) else np.nan) for s in scenes]
            offsets = np.arange(len(scenes)) + (index - len(present) / 2) * width
            ax.bar(offsets, values, width=width * 0.92,
                   color=REGIME_STYLE[regime]["color"],
                   label=cfg.REGIME_LABELS[regime])
        ax.set_xticks(np.arange(len(scenes)))
        ax.set_xticklabels(scenes)
        ax.set_ylabel("held-out PSNR (dB)")
        finite = [v for regime in present for v in
                  [(by_regime[regime].get(s).psnr_srgb
                    if by_regime[regime].get(s) else None) for s in scenes]
                  if v is not None]
        figs.zoom_to_data(ax, finite, pad_fraction=0.15, min_pad=0.3)
        figs.legend_below(fig, ax, ncol=4)
        written += figs.save(fig, out, "training_psnr_by_scene")

    # 2. The Pareto view: quality against the PRIMARY performance metric. One
    #    point per (regime, scene), so a regime that wins on average but loses on a
    #    scene is visible rather than averaged away.
    fig, ax = plt.subplots(figsize=(figs.FIG_WIDTH_DOUBLE * 0.6, 2.6))
    for regime in present:
        xs, ys = [], []
        for scene in scenes:
            cell = by_regime[regime].get(scene)
            if cell and cell.backward_raster_ms and cell.psnr_srgb is not None:
                xs.append(cell.backward_raster_ms)
                ys.append(cell.psnr_srgb)
        if xs:
            ax.scatter(xs, ys, s=18, label=cfg.REGIME_LABELS[regime],
                       color=REGIME_STYLE[regime]["color"],
                       marker=REGIME_STYLE[regime]["marker"])
    ax.set_xlabel("backward raster (ms/iter)")
    ax.set_ylabel("held-out PSNR (dB)")
    figs.legend_below(fig, ax, ncol=3)
    written += figs.save(fig, out, "training_quality_vs_backward_time")

    # 3. alpha and sigma sweeps, as deltas against uniform STBN on the same scene.
    rows = hyper.get("rows") or []
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(figs.FIG_WIDTH_DOUBLE, 2.4))
        for ax, parameter in zip(axes, ("alpha", "sigma")):
            for scene in cfg.HYPER_SCENES:
                picked = sorted([r for r in rows if r["parameter"] == parameter
                                 and r["scene"] == scene],
                                key=lambda r: r["value"])
                if not picked:
                    continue
                ax.plot([r["value"] for r in picked],
                        [r["psnr_delta_vs_uniform_stbn"] for r in picked],
                        marker="o", markersize=3, linewidth=1.0, label=scene)
            ax.axhline(0.0, color="#000000", linewidth=0.7)
            ax.set_xlabel(rf"$\{parameter}$" if parameter == "alpha"
                          else r"$\sigma$ (output pixels)")
            ax.set_ylabel("PSNR vs uniform STBN (dB)")
        axes[0].legend(fontsize=6)
        written += figs.save(fig, out, "training_hyperparameters")

    # 4. The budget sweep: quality against rho, uniform vs adaptive, with the
    #    uniform rho=0.25 level drawn in — the line the "same quality at a lower
    #    budget" claim has to cross.
    sweep = fractions.get("rows") or []
    if sweep:
        fig, axes = plt.subplots(1, 2, figsize=(figs.FIG_WIDTH_DOUBLE, 2.4))
        for regime in cfg.FRACTION_REGIMES:
            picked = sorted([r for r in sweep if r["regime"] == regime],
                            key=lambda r: r["pixel_fraction"])
            if not picked:
                continue
            style = REGIME_STYLE[regime]
            axes[0].errorbar([r["rho_enforced"] for r in picked],
                             [r["psnr_mean"] for r in picked],
                             yerr=[r["psnr_std"] for r in picked],
                             label=cfg.REGIME_LABELS[regime], capsize=2,
                             linewidth=1.0, markersize=3, **style)
            axes[1].plot([r["rho_enforced"] for r in picked],
                         [r["backward_raster_ms_mean"] for r in picked],
                         label=cfg.REGIME_LABELS[regime], linewidth=1.0,
                         markersize=3, **style)
        reference = next((r for r in sweep if r["regime"] == "stbn"
                          and abs(r["pixel_fraction"] - 0.25) < 1e-9), None)
        if reference and reference["psnr_mean"] is not None:
            axes[0].axhline(reference["psnr_mean"], color="#999999",
                            linewidth=0.7, linestyle=":",
                            label=r"uniform STBN @ $\rho$=0.25")
        for ax, label in zip(axes, ("held-out PSNR (dB)",
                                    "backward raster (ms/iter)")):
            ax.set_xlabel(r"enforced $\rho$")
            ax.set_ylabel(label)
        figs.legend_below(fig, axes[0], ncol=3)
        written += figs.save(fig, out, "training_fraction_sweep")
    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(paths: dict, manifest: Manifest,
        scenes: tuple[str, ...] | None = None) -> dict:
    records = collect(manifest)
    scenes = tuple(scenes or cfg.PILOT_SCENES)
    pilot = pilot_tables(paths, records, scenes)
    hyper = hyper_tables(paths, records)
    fractions = fraction_tables(paths, records, scenes)
    scaling = backward_scaling(paths, records, scenes)
    written = figures(paths, pilot, hyper, fractions, scenes)
    record = {
        "status": "ok" if pilot["summary"] else "empty",
        "n_usable_cells": len(records),
        "scenes": list(scenes),
        "regimes_present": sorted({r.regime for r in records}),
        "pilot_summary": pilot["summary"],
        "contrasts": pilot["contrasts"],
        "hyperparameters": hyper["rows"],
        "fraction_sweep": fractions["rows"],
        "equal_quality": fractions["equal_quality"],
        "backward_scaling": scaling,
        "outputs": {
            "cells_csv": "training/pilot_cells.csv",
            "summary_csv": "training/pilot_summary.csv",
            "contrasts_csv": "training/pilot_contrasts.csv",
            "hyperparameters_csv": "training/hyperparameters.csv",
            "fraction_sweep_csv": "training/fraction_sweep.csv",
            "backward_scaling_csv": "training/backward_scaling.csv",
            "figures": [str(p.relative_to(paths["root"])) for p in written],
        },
    }
    (paths["training"] / "training_results.json").write_text(
        json.dumps(record, indent=2, default=str))
    return record


# ---------------------------------------------------------------------------
# How much of the backward can allocation reach at all?
# ---------------------------------------------------------------------------

BACKWARD_SCALING_COLUMNS = (
    "scene", "n_points", "dense_backward_ms", "backward_ms_at_025",
    "rho_independent_ms", "rho_dependent_ms_at_025",
    "rho_independent_share_at_025", "rho_independent_share_at_010",
    "addressable_ms_at_025", "step_total_ms", "addressable_share_of_step",
)


def backward_scaling(paths: dict, records: list,
                     scenes: tuple[str, ...]) -> dict:
    """Decompose the backward raster into its rho-dependent and -independent parts.

    This bounds the whole approach, and it is the number the study's recommendation
    rests on, so it is measured rather than inferred. Fitting
    ``bwd(rho) = a + b*rho`` over the uniform STBN arm separates the per-pixel loop
    — the only part any sample allocation can touch — from the per-splat projection,
    the int64 atomics, the bucket setup and the checkpoint staging, which run
    whatever the mask says.

    ``addressable_share_of_step`` is the punchline: what fraction of a training step
    a PERFECT allocation could remove by cutting the sampled fraction. If that is
    small, an improved importance proxy has nowhere to convert its improvement into
    time, and the honest conclusion is about the ceiling and not about the proxy.
    """
    import json as _json

    out = paths["training"]
    rows: list[dict] = []
    for scene in scenes:
        points = [(f, r) for f in cfg.FRACTION_SWEEP
                  for r in records if _matches(r, "stbn", f)]
        points = [(f, r) for f, r in points if r.backward_raster_ms]
        dense = next((r for r in records if _matches(r, cfg.REGIME_FULL, 1.0)
                      and r.scene == scene), None)
        points = [(f, r) for f, r in points if r.scene == scene]
        if len(points) < 2:
            continue
        x = np.asarray([f for f, _ in points], dtype=float)
        y = np.asarray([r.backward_raster_ms for _, r in points], dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        at_025 = next((r.backward_raster_ms for f, r in points
                       if abs(f - 0.25) < 1e-9), None)
        step = None
        reference = next((r for f, r in points if abs(f - 0.25) < 1e-9), None)
        if reference is not None:
            try:
                timing = _json.loads(
                    (Path(reference.run_dir) / "stage_timing.json").read_text())
                step = (timing.get("whole_run") or {}).get("stage_total_ms")
            except (OSError, ValueError):
                step = None
        dependent = float(slope * 0.25)
        rows.append({
            "scene": scene, "n_points": len(points),
            "dense_backward_ms": (dense.backward_raster_ms if dense else None),
            "backward_ms_at_025": at_025,
            "rho_independent_ms": float(intercept),
            "rho_dependent_ms_at_025": dependent,
            "rho_independent_share_at_025": (
                float(intercept / (intercept + dependent))
                if intercept + dependent > 0 else None),
            "rho_independent_share_at_010": (
                float(intercept / (intercept + slope * 0.10))
                if intercept + slope * 0.10 > 0 else None),
            # What a PERFECT allocation could remove: the whole rho-dependent part.
            "addressable_ms_at_025": dependent,
            "step_total_ms": step,
            "addressable_share_of_step": (dependent / step if step else None),
        })
    if not rows:
        return {"status": "missing"}
    tables.write_csv(out / "backward_scaling.csv", rows,
                     BACKWARD_SCALING_COLUMNS)
    return {
        "status": "ok",
        "rows": rows,
        "rho_independent_share_at_025": _mean(
            [r["rho_independent_share_at_025"] for r in rows]),
        "rho_independent_share_at_010": _mean(
            [r["rho_independent_share_at_010"] for r in rows]),
        "addressable_share_of_step": _mean(
            [r["addressable_share_of_step"] for r in rows]),
        "method": ("linear fit of backward_raster_ms against rho over the uniform "
                   "STBN arm at rho in {0.10, 0.15, 0.25}; the intercept is the "
                   "work that runs whatever the mask says"),
        "outputs": {"csv": "training/backward_scaling.csv"},
    }
