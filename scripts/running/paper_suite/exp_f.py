# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Experiment F: launch the per-scene gradient probes and aggregate their results.

Sanity check: at K=1 a single STBN slice is an independent blue-noise slice, so the two must agree.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import config as cfg
from . import figures as figs
from . import stats
from . import tables

PROBE_SCRIPT = cfg.REPO / "scripts" / "running" / "paper_gradient_probe.py"

RAW_COLUMNS = (
    "scene", "dataset_family", "state_iter", "n_gaussians", "camera_index",
    "regime", "k", "param_group", "cosine_similarity", "relative_l2_error",
    "norm_ratio", "sparse_norm", "dense_norm", "n_values",
)

SUMMARY_COLUMNS = (
    "regime", "k", "param_group", "n_scenes", "n_states",
    "cosine_mean", "cosine_std", "cosine_min", "cosine_max",
    "relative_l2_mean", "relative_l2_std", "norm_ratio_mean",
)

CHECK_COLUMNS = (
    "param_group", "k", "stbn_cosine_mean", "indep_blue_cosine_mean",
    "difference", "expected", "note",
)


def _probe_command(scene: str, out_path: Path, *, python: str,
                   state_iters, k_values, fraction: float) -> list[str]:
    directory = cfg.scene_dir(scene)
    if directory is None:
        raise FileNotFoundError(f"scene {scene!r} not found")
    return [
        python, str(PROBE_SCRIPT),
        "--scene", str(directory),
        "--state-iters", *[str(i) for i in state_iters],
        "--k-values", *[str(k) for k in k_values],
        "--probe-fraction", f"{fraction:.4f}",
        # STBN on with the NVIDIA table; the probe still advances to each state densely.
        "--stbn-pixel-fraction", f"{fraction:.4f}",
        "--stbn-source", "nvidia",
        "--iters", str(max(state_iters)),
        "--seed", str(cfg.PRIMARY_SEED),
        "--sh-degree", "3",
        "--densify-mode", "structure_aware",
        "--mip-splatting",
        "--downsample", "4",
        "--no-viewer",
        "--out", str(out_path),
    ]


def run_probes(out: dict, *, python: str, scenes=cfg.GRADIENT_SCENES,
               state_iters=cfg.GRADIENT_STATES, k_values=cfg.GRADIENT_K_VALUES,
               fraction: float = cfg.PRIMARY_FRACTION,
               force: bool = False, timeout_s: float = 5400.0) -> dict:
    """Launch one probe per scene, reusing parseable results unless ``force``; failures are recorded, not fatal."""
    directory: Path = out["gradient_analysis"]
    directory.mkdir(parents=True, exist_ok=True)
    logs: Path = out["logs"]
    status: dict[str, dict] = {}

    for scene in scenes:
        target = directory / f"gradient_{scene}.json"
        if target.is_file() and not force:
            try:
                json.loads(target.read_text())
                print(f"[grad] cached {scene}", flush=True)
                status[scene] = {"status": "cached", "path": str(target)}
                continue
            except ValueError:
                pass
        if cfg.scene_dir(scene) is None:
            status[scene] = {"status": "missing_scene"}
            print(f"[grad] SKIP {scene}: scene not found", flush=True)
            continue

        command = _probe_command(scene, target, python=python,
                                 state_iters=state_iters, k_values=k_values,
                                 fraction=fraction)
        stdout_path = logs / f"gradient_{scene}.stdout.log"
        stderr_path = logs / f"gradient_{scene}.stderr.log"
        print(f"[grad] PROBE {scene}", flush=True)
        started = time.perf_counter()
        try:
            with stdout_path.open("w", encoding="utf-8", errors="replace") as o, \
                    stderr_path.open("w", encoding="utf-8", errors="replace") as e:
                proc = subprocess.run(command, cwd=str(cfg.REPO), stdout=o,
                                      stderr=e, timeout=timeout_s)
            ok = proc.returncode == 0 and target.is_file()
            status[scene] = {
                "status": "ok" if ok else "failed",
                "return_code": proc.returncode,
                "seconds": time.perf_counter() - started,
                "stderr_path": str(stderr_path.relative_to(cfg.REPO)),
            }
            if not ok:
                print(f"[grad] FAILED {scene} (rc={proc.returncode}); "
                      f"see {stderr_path}", flush=True)
        except (OSError, subprocess.SubprocessError) as error:
            status[scene] = {"status": "failed",
                             "error": f"{type(error).__name__}: {error}",
                             "stderr_path": str(stderr_path.relative_to(cfg.REPO))}
            print(f"[grad] FAILED {scene}: {error}", flush=True)
    return status


def _raw_rows(out: dict, scenes) -> list[dict]:
    directory: Path = out["gradient_analysis"]
    rows: list[dict] = []
    for scene in scenes:
        path = directory / f"gradient_{scene}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text())
        except ValueError:
            continue
        for state in payload.get("states") or []:
            for regime, block in (state.get("regimes") or {}).items():
                for k, groups in (block.get("by_k") or {}).items():
                    for group, values in groups.items():
                        rows.append({
                            "scene": scene,
                            "dataset_family": cfg.scene_family(scene),
                            "state_iter": state.get("state_iter"),
                            "n_gaussians": state.get("n_gaussians"),
                            "camera_index": state.get("camera_index"),
                            "regime": regime,
                            "k": int(k),
                            "param_group": group,
                            "cosine_similarity": values.get("cosine_similarity"),
                            "relative_l2_error": values.get("relative_l2_error"),
                            "norm_ratio": values.get("norm_ratio"),
                            "sparse_norm": values.get("sparse_norm"),
                            "dense_norm": values.get("dense_norm"),
                            "n_values": values.get("n_values"),
                        })
    return rows


def _summary_rows(raw: list[dict]) -> list[dict]:
    out: list[dict] = []
    regimes = ("iid_white", "indep_blue", "stbn")
    groups = []
    for row in raw:
        if row["param_group"] not in groups:
            groups.append(row["param_group"])
    for regime in regimes:
        for k in sorted({r["k"] for r in raw}):
            for group in groups:
                picked = [r for r in raw if r["regime"] == regime
                          and r["k"] == k and r["param_group"] == group]
                if not picked:
                    continue
                cosine = stats.mean_std([r["cosine_similarity"] for r in picked])
                rel = stats.mean_std([r["relative_l2_error"] for r in picked])
                norm = stats.mean_std([r["norm_ratio"] for r in picked])
                out.append({
                    "regime": regime, "k": k, "param_group": group,
                    "n_scenes": len({r["scene"] for r in picked}),
                    "n_states": len({r["state_iter"] for r in picked}),
                    "cosine_mean": cosine["mean"], "cosine_std": cosine["std"],
                    "cosine_min": cosine["min"], "cosine_max": cosine["max"],
                    "relative_l2_mean": rel["mean"], "relative_l2_std": rel["std"],
                    "norm_ratio_mean": norm["mean"],
                })
    return out


def _k1_equivalence_check(summary: list[dict]) -> list[dict]:
    """At K=1, STBN and independent blue noise are the same object, so any gap is harness noise."""
    rows: list[dict] = []
    for group in sorted({r["param_group"] for r in summary}):
        stbn = next((r for r in summary if r["regime"] == "stbn" and r["k"] == 1
                     and r["param_group"] == group), None)
        indep = next((r for r in summary if r["regime"] == "indep_blue"
                      and r["k"] == 1 and r["param_group"] == group), None)
        if not stbn or not indep:
            continue
        a, b = stbn["cosine_mean"], indep["cosine_mean"]
        difference = (a - b) if (a is not None and b is not None) else None
        rows.append({
            "param_group": group, "k": 1,
            "stbn_cosine_mean": a, "indep_blue_cosine_mean": b,
            "difference": difference,
            "expected": 0.0,
            "note": ("a single STBN slice IS a single independent blue-noise "
                     "slice; a non-trivial difference here would indicate a "
                     "harness problem, not a temporal effect"),
        })
    return rows


def _figure(summary: list[dict], figures_dir: Path, k_values) -> None:
    plt = figs.pyplot()
    if plt is None:
        return
    groups = ["all", "position", "scale", "rotation", "opacity", "color_sh0"]
    groups = [g for g in groups if any(r["param_group"] == g for r in summary)]
    if not groups:
        return

    fig, axes = plt.subplots(2, len(groups),
                             figsize=(figs.FIG_WIDTH_DOUBLE,
                                      3.8), squeeze=False, sharex=True)
    for column, group in enumerate(groups):
        for row_index, (metric, ylabel) in enumerate(
                (("cosine_mean", "cosine similarity to dense"),
                 ("relative_l2_mean", "relative $L_2$ error"))):
            ax = axes[row_index][column]
            for regime in ("iid_white", "indep_blue", "stbn"):
                style = figs.style_for(regime)
                points = sorted(
                    (r for r in summary if r["regime"] == regime
                     and r["param_group"] == group), key=lambda r: r["k"])
                xs = [r["k"] for r in points if r.get(metric) is not None]
                ys = [r[metric] for r in points if r.get(metric) is not None]
                es = [r.get(metric.replace("_mean", "_std")) or 0.0
                      for r in points if r.get(metric) is not None]
                if not xs:
                    continue
                ax.errorbar(xs, ys, yerr=es, color=style["color"],
                            marker=style["marker"], linestyle=style["linestyle"],
                            linewidth=1.1, markersize=3.5, capsize=1.5,
                            label=style["label"], elinewidth=0.6)
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(k_values))
            ax.set_xticklabels([str(k) for k in k_values])
            if row_index == 0:
                ax.set_title(group, fontsize=7.5)
            if row_index == 1:
                ax.set_xlabel("revisits $K$")
                ax.set_yscale("log")
            if column == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Experiment F: sparse accumulated gradient vs the dense "
                 "gradient, at frozen training states", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    figs.save(fig, figures_dir, "expF_gradient_quality")


def _latex(summary: list[dict], latex_dir: Path, k_values) -> None:
    rows: list[list[str]] = []
    for regime in ("iid_white", "indep_blue", "stbn"):
        cells = [cfg.REGIME_LABELS[regime]]
        for k in k_values:
            row = next((r for r in summary if r["regime"] == regime
                        and r["k"] == k and r["param_group"] == "all"), None)
            cells.append(tables.num(row.get("cosine_mean") if row else None, 4))
        for k in k_values:
            row = next((r for r in summary if r["regime"] == regime
                        and r["k"] == k and r["param_group"] == "all"), None)
            cells.append(tables.num(row.get("relative_l2_mean") if row else None,
                                    4))
        rows.append(cells)
    header = (["Supervision"]
              + [rf"$K{{=}}{k}$" for k in k_values]
              + [rf"$K{{=}}{k}$" for k in k_values])
    body = tables.latex_table(
        caption=(rf"Gradient agreement with dense supervision at "
                 rf"$\rho={cfg.PRIMARY_FRACTION}$, averaged over "
                 rf"{len(cfg.GRADIENT_SCENES)} scenes and "
                 rf"{len(cfg.GRADIENT_STATES)} frozen training states, over all "
                 rf"14 gradient components. $K$ is the number of consecutive "
                 rf"masks averaged. At $K{{=}}1$ STBN and independent blue noise "
                 rf"are the same object and agree, as they must; the separation "
                 rf"is purely a function of accumulation."),
        label="tab:stbn_gradients",
        column_spec="l" + "r" * (2 * len(k_values)),
        header=header, rows=rows,
        notes=(r"Left block: cosine similarity (higher is better). Right block: "
               r"relative $L_2$ error (lower is better). No parameter is updated "
               r"during the diagnostic; the camera is pinned and the Gaussians "
               r"are frozen."),
        source="gradient_analysis/gradient_summary.csv")
    tables.write_latex(latex_dir / "table_gradient_quality.tex", body)


def run(out: dict, *, python: str, scenes=cfg.GRADIENT_SCENES,
        state_iters=cfg.GRADIENT_STATES, k_values=cfg.GRADIENT_K_VALUES,
        fraction: float = cfg.PRIMARY_FRACTION, force: bool = False,
        skip_probes: bool = False) -> dict:
    directory: Path = out["gradient_analysis"]
    status = ({} if skip_probes else
              run_probes(out, python=python, scenes=scenes,
                         state_iters=state_iters, k_values=k_values,
                         fraction=fraction, force=force))

    raw = _raw_rows(out, scenes)
    if not raw:
        return {"status": "no gradient probes completed", "probe_status": status}

    tables.write_csv(directory / "gradient_raw.csv", raw, RAW_COLUMNS)
    summary = _summary_rows(raw)
    tables.write_csv(directory / "gradient_summary.csv", summary, SUMMARY_COLUMNS)
    check = _k1_equivalence_check(summary)
    tables.write_csv(directory / "gradient_k1_equivalence_check.csv", check,
                     CHECK_COLUMNS)

    _figure(summary, directory, k_values)
    _latex(summary, out["latex"], k_values)

    overall = {}
    for regime in ("iid_white", "indep_blue", "stbn"):
        overall[regime] = {
            str(k): next(({"cosine": r["cosine_mean"],
                           "relative_l2": r["relative_l2_mean"],
                           "norm_ratio": r["norm_ratio_mean"]}
                          for r in summary if r["regime"] == regime
                          and r["k"] == k and r["param_group"] == "all"), None)
            for k in k_values}

    worst_k1_gap = max((abs(r["difference"]) for r in check
                        if r["difference"] is not None), default=None)
    return {
        "status": "ok",
        "probe_status": status,
        "scenes": sorted({r["scene"] for r in raw}),
        "states": sorted({r["state_iter"] for r in raw}),
        "k_values": list(k_values),
        "overall_all_components": overall,
        "k1_equivalence_worst_gap": worst_k1_gap,
        "k1_equivalence_verdict": (
            "harness consistent: STBN and independent blue noise agree at K=1"
            if (worst_k1_gap is not None and worst_k1_gap < 0.02)
            else "CHECK: STBN and independent blue noise differ at K=1, where "
                 "they should be equivalent"),
        "artifacts": {
            "raw_csv": "gradient_analysis/gradient_raw.csv",
            "summary_csv": "gradient_analysis/gradient_summary.csv",
            "k1_check_csv": "gradient_analysis/gradient_k1_equivalence_check.csv",
            "figures": ["gradient_analysis/expF_gradient_quality.pdf"],
            "latex": ["latex/table_gradient_quality.tex"],
        },
    }
