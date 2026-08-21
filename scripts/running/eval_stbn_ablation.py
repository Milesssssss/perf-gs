"""STBN sparse-supervision PAPER SUITE: train the grid, then produce every table,
figure and number the study needs, resumably and unattended.

This grew out of a four-cell ablation (dense / white / independent blue /
spatiotemporal blue on one scene set) and is now the whole study. One long run
fills ``outputs/paper_stbn/`` with the quantitative data and paper-ready figures.

    # the whole thing, unattended (hours; run it on an IDLE GPU)
    .venv/Scripts/python.exe scripts/running/eval_stbn_ablation.py

    # what would run, and the projected cost, without training anything
    .venv/Scripts/python.exe scripts/running/eval_stbn_ablation.py --dry-run

    # smoke test: one scene, 100 iterations, every stage exercised end to end
    .venv/Scripts/python.exe scripts/running/eval_stbn_ablation.py --smoke

    # re-derive every table/figure from completed runs, training nothing
    .venv/Scripts/python.exe scripts/running/eval_stbn_ablation.py --analyze-only

WHAT IT RUNS

    A  primary 25% ablation      13 scenes x 4 regimes x 3 seeds
    B  pixel-budget sweep        rho in {0.10, 0.25, 0.50}, seed 42, reusing A
    C  convergence               8 checkpoints on the seed-42 25% cells
    D  runtime breakdown         per-stage GPU timing, 4 representative scenes
    E  sampling diagnostics      the masks themselves, no training
    F  gradient diagnostic       sparse-vs-dense gradient at frozen states
    G  qualitative              contact sheets, error maps, ranked crops

The four regimes, identical in everything but the mask:

    full        100% of pixels                --stbn-pixel-fraction 1.0
    iid_white   25%, fresh white mask/step    --stbn-source white  --stbn-independent
    indep_blue  25%, fresh blue mask/step     --stbn-source nvidia --stbn-independent
    stbn        25%, the per-camera walk      --stbn-source nvidia

and the two contrasts everything rests on: ``iid_white -> indep_blue`` is the
spatial arrangement, ``indep_blue -> stbn`` is the temporal ordering.

RESUMABILITY. Every cell is a fresh subprocess, and every cell is identified by a
hash of its exact invocation. A cell whose outputs exist AND whose recorded hash
matches is reused; anything else re-runs with the reason logged. So an interrupted
suite resumes for free, and a changed flag invalidates exactly the cells it
affects rather than silently mixing two configurations into one table.

Prerequisite for indep_blue + stbn: the NVIDIA asset (NON-COMMERCIAL licence) --
    .venv/Scripts/python.exe scripts/data/download_stbn.py

RUN IT ON AN IDLE GPU. Timings are wall-clock, so a profiler, a second trainer or
a live viewer sharing the device silently inflates every one of them. The
device-wide memory poll assumes the same thing and reports its baseline so a
violation is visible.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from paper_suite import config as cfg
from paper_suite import env as env_module
from paper_suite import (exp_a, exp_b, exp_c, exp_d, exp_e, exp_f, exp_f2, exp_g,
                         exp_s, report)
from paper_suite import output_layout
from paper_suite.manifest import Manifest
from paper_suite.results import ResultSet
from paper_suite.runner import DEFAULT_CELL_TIMEOUT_S, run_plan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="STBN sparse-supervision paper suite (train + analyse).")
    parser.add_argument("--out", default=str(cfg.OUTPUT_ROOT),
                       help="output root (default outputs/paper_stbn)")
    parser.add_argument("--scenes", default=",".join(cfg.SCENES),
                       help="comma-separated scene subset")
    parser.add_argument("--seeds", default=",".join(str(s) for s in cfg.SEEDS))
    parser.add_argument("--fractions",
                       default=",".join(str(f) for f in cfg.SWEEP_FRACTIONS))
    parser.add_argument("--iters", type=int, default=cfg.ITERS)
    parser.add_argument("--python", default=sys.executable,
                       help="interpreter used for every child process")
    parser.add_argument("--force", action="store_true",
                       help="retrain every cell, ignoring cached results")
    parser.add_argument("--dry-run", action="store_true",
                       help="print the plan and each cell's argv; train nothing")
    parser.add_argument("--analyze-only", action="store_true",
                       help="skip training entirely and rebuild every table, "
                            "figure and report from cells already on disk")
    parser.add_argument("--skip-training", action="store_true",
                       help="alias for --analyze-only")
    parser.add_argument("--no-timing", action="store_true",
                       help="omit Experiment D's dedicated stage-timing cells")
    parser.add_argument("--no-stratification", action="store_true",
                       help="omit Experiment S: the per-tile-rank arm that "
                            "measures whether forcing exactly k pixels per 8x8 "
                            "execution tile (instead of thresholding the whole "
                            "slice) changes trained quality")
    parser.add_argument("--no-gradients", action="store_true",
                       help="omit Experiment F's gradient probes")
    parser.add_argument("--no-qualitative", action="store_true",
                       help="omit Experiment G (contact sheets are the bulk of "
                            "the output's file count)")
    parser.add_argument("--gradient-study", action="store_true",
                       help="Run ONLY Experiment F2, the controlled "
                            "gradient-fidelity study: freeze real training "
                            "states and compare each sparse strategy's gradient "
                            "against the dense one on identical Gaussians, "
                            "camera, prediction and loss. Trains nothing and "
                            "takes no optimizer step. Kept off the default path "
                            "because it is hours of measurement in its own right.")
    parser.add_argument("--gradient-study-scenes",
                       default=",".join(exp_f2.STUDY_SCENES))
    parser.add_argument("--gradient-study-cameras", type=int, default=8)
    parser.add_argument("--gradient-study-single-masks", type=int, default=4)
    parser.add_argument("--gradient-study-sequences", type=int, default=2)
    parser.add_argument("--gradient-study-states", type=int, nargs="*",
                       default=list(exp_f2.STUDY_STATES))
    parser.add_argument("--gradient-study-k", type=int, nargs="*",
                       default=list(exp_f2.STUDY_K))
    parser.add_argument("--cell-timeout-s", type=float,
                       default=DEFAULT_CELL_TIMEOUT_S)
    parser.add_argument("--max-views-per-scene", type=int, default=None,
                       help="cap Experiment G's contact sheets per scene "
                            "(diagnostic; the default is every held-out view)")
    parser.add_argument("--smoke", action="store_true",
                       help="one scene, 100 iterations, 1 seed, every stage "
                            "exercised: validates metrics, timers, plots, "
                            "caching and the output tree before the real run")
    return parser.parse_args()


def _apply_smoke(args: argparse.Namespace) -> None:
    """A cheap end-to-end rehearsal of every moving part.

    Deliberately still runs all four regimes, the convergence ladder, the stage
    timer, the gradient probe and the qualitative pass — a smoke test that skipped
    stages would not tell us the thing we want to know before committing to a
    multi-hour run. Only the scene count, seed count and iteration count shrink.
    """
    args.scenes = "bicycle"
    args.seeds = str(cfg.PRIMARY_SEED)
    args.iters = 100
    args.fractions = "0.25,0.50"
    args.out = str(cfg.REPO / "outputs" / "paper_stbn_smoke")
    args.max_views_per_scene = args.max_views_per_scene or 3
    cfg.CONVERGENCE_ITERS = (25, 50, 75, 100)
    cfg.GRADIENT_SCENES = ("bicycle",)
    cfg.GRADIENT_STATES = (50, 100)
    cfg.GRADIENT_K_VALUES = (1, 2, 4)
    cfg.QUALITY_TARGETS_DB = (0.5, 0.2, 0.1)
    cfg.TIMING_SCENES = ("bicycle",)
    cfg.RENDER_FPS_FRAMES = 10
    cfg.BOOTSTRAP_RESAMPLES = 2000


def main() -> int:
    args = _parse_args()
    if args.smoke:
        _apply_smoke(args)
    analyze_only = args.analyze_only or args.skip_training

    scenes = tuple(s.strip() for s in args.scenes.split(",") if s.strip())
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    fractions = tuple(float(f) for f in args.fractions.split(",") if f.strip())

    out = output_layout(Path(args.out))
    environment = env_module.collect_environment()
    env_module.write_environment(out["root"] / "environment.json", environment)

    plan = cfg.build_plan(scenes=scenes, seeds=seeds, fractions=fractions,
                          iters=args.iters, with_timing=not args.no_timing,
                          timing_scenes=cfg.TIMING_SCENES,
                          with_stratification=not args.no_stratification)
    summary = cfg.plan_summary(plan)

    git = environment.get("git") or {}
    gpu = environment.get("gpu") or {}
    print(f"[suite] {summary['total_cells']} cells "
          f"({summary['by_experiment']})")
    print(f"[suite] {len(scenes)} scenes, seeds {list(seeds)}, "
          f"fractions {list(fractions)}, {args.iters} iters -> {out['root']}")
    print(f"[suite] commit {git.get('commit_short')} "
          f"(dirty={git.get('dirty')}), GPU {gpu.get('name')} "
          f"driver {gpu.get('driver_version')}")
    if git.get("dirty") and git.get("dirty_source_files"):
        print(f"[suite] WARNING: working tree has modified source files "
              f"({len(git['dirty_source_files'])}); the commit hash does not "
              f"identify the code producing these numbers.")
    if not (environment.get("stbn_asset") or {}).get("present"):
        print("[suite] WARNING: the NVIDIA STBN asset is absent — indep_blue and "
              "stbn cells will fail. Run scripts/data/download_stbn.py first.")

    manifest = Manifest.load(out["root"] / "manifest.json", environment)
    manifest.environment = environment
    manifest.suite = {
        "suite_schema": cfg.SUITE_SCHEMA,
        "scenes": list(scenes),
        "seeds": list(seeds),
        "fractions": list(fractions),
        "iters": args.iters,
        "select_mode": cfg.PRIMARY_SELECT_MODE,
        "stratification_arm": (not args.no_stratification),
        "plan": summary,
        "convergence_iters": list(cfg.CONVERGENCE_ITERS),
        "timing_scenes": list(cfg.TIMING_SCENES),
        "gradient_scenes": list(cfg.GRADIENT_SCENES),
        "last_invocation": " ".join(sys.argv),
        "last_started_at": env_module.utc_now(),
    }

    if args.gradient_study:
        # Measurement-only mode: no training grid, no manifest cells. Writes into
        # gradient_analysis/ and findings_gradient_study.json so it cannot
        # overwrite a completed suite's report.
        started = time.perf_counter()
        study_scenes = tuple(s.strip() for s in
                             args.gradient_study_scenes.split(",") if s.strip())
        print(f"[suite] gradient study only: scenes={list(study_scenes)} "
              f"states={args.gradient_study_states} "
              f"K<={max(args.gradient_study_k)} "
              f"cameras={args.gradient_study_cameras}")
        finding = exp_f2.run(
            out, python=args.python, scenes=study_scenes,
            states=tuple(args.gradient_study_states),
            k_values=tuple(sorted(args.gradient_study_k)),
            cameras=args.gradient_study_cameras,
            single_masks=args.gradient_study_single_masks,
            sequences=args.gradient_study_sequences,
            fraction=cfg.PRIMARY_FRACTION, force=args.force,
            skip_runs=analyze_only)
        (out["root"] / "findings_gradient_study.json").write_text(
            json.dumps(finding, indent=2, default=str))
        path = report.build_gradient_study(
            finding, environment=environment, out_root=out["root"])
        print(f"\n[suite] gradient study done in "
              f"{(time.perf_counter() - started) / 3600:.2f}h")
        print(f"[suite] report: {path}")
        return 0

    started = time.perf_counter()
    if analyze_only:
        print("[suite] --analyze-only: training nothing; rebuilding outputs "
              "from cells already on disk.")
        counts = {"ok": 0, "cached": 0, "failed": 0, "skipped": 0,
                  "wall_hours": 0.0}
        # Re-fold every cell whose artifacts are on disk, so the manifest rows
        # are re-derived from what actually exists rather than trusted from a
        # previous session. run_cell reuses (never retrains) when the hash
        # matches, and reports the reason when it does not.
        from paper_suite.runner import run_cell
        for cell in plan:
            if not manifest.outputs_present(cell):
                continue
            run_cell(cell, manifest, python=args.python, logs_dir=out["logs"],
                     force=False, dry_run=True, timeout_s=args.cell_timeout_s)
        manifest.save()
    else:
        counts = run_plan(plan, manifest, python=args.python,
                          logs_dir=out["logs"], force=args.force,
                          timeout_s=args.cell_timeout_s, dry_run=args.dry_run)

    if args.dry_run:
        integrity = manifest.integrity_report(plan)
        print(f"\n[suite] dry run: {summary['total_cells']} cells planned, "
              f"{integrity['ok']} already usable, "
              f"{summary['total_cells'] - integrity['ok']} would train.")
        print(f"[suite] plan written to {out['root'] / 'plan.json'}")
        (out["root"] / "plan.json").write_text(json.dumps(
            {"summary": summary,
             "cells": [{"key": c.key, "consumers": list(c.consumers),
                        "argv": c.command(args.python)} for c in plan]},
            indent=2))
        return 0

    integrity = manifest.integrity_report(plan)
    print(f"\n[suite] training accounting: {counts}")
    print(f"[suite] integrity: {integrity['ok']}/{integrity['planned']} usable, "
          f"{integrity['failed']} failed, {integrity['missing']} missing")

    # ---- analysis --------------------------------------------------------
    results = ResultSet(plan, manifest)
    findings: dict[str, dict] = {}

    def stage(name: str, fn) -> None:
        """Run one analysis stage; a stage that raises must not lose the others.

        The training hours are the expensive part and they are already on disk by
        now, so an analysis bug should cost one section of the report, not the run.
        """
        print(f"[suite] analysis: experiment {name}", flush=True)
        try:
            findings[name] = fn()
        except Exception as error:                         # noqa: BLE001
            import traceback
            findings[name] = {"status": f"analysis failed: "
                                        f"{type(error).__name__}: {error}"}
            print(f"[suite] experiment {name} analysis FAILED: {error}",
                  flush=True)
            traceback.print_exc()

    stage("A", lambda: exp_a.run(results, out, scenes=scenes, seeds=seeds,
                                 fraction=cfg.PRIMARY_FRACTION))
    stage("B", lambda: exp_b.run(results, out, scenes=scenes,
                                 fractions=fractions, seed=cfg.PRIMARY_SEED))
    stage("C", lambda: exp_c.run(results, out, scenes=scenes,
                                 fraction=cfg.PRIMARY_FRACTION,
                                 seed=cfg.PRIMARY_SEED))
    if not args.no_timing:
        stage("D", lambda: exp_d.run(results, out,
                                     fraction=cfg.PRIMARY_FRACTION))
    stage("E", lambda: exp_e.run(out, fraction=cfg.PRIMARY_FRACTION,
                                 visits=cfg.SAMPLING_VISITS))
    if not args.no_stratification:
        stage("S", lambda: exp_s.run(results, out, scenes=scenes,
                                     fraction=cfg.PRIMARY_FRACTION,
                                     seed=cfg.PRIMARY_SEED))
    if not args.no_gradients:
        stage("F", lambda: exp_f.run(out, python=args.python,
                                     scenes=cfg.GRADIENT_SCENES,
                                     state_iters=cfg.GRADIENT_STATES,
                                     k_values=cfg.GRADIENT_K_VALUES,
                                     fraction=cfg.PRIMARY_FRACTION,
                                     force=args.force,
                                     skip_probes=analyze_only))
    if not args.no_qualitative:
        stage("G", lambda: exp_g.run(results, out, scenes=scenes,
                                     fraction=cfg.PRIMARY_FRACTION,
                                     seed=cfg.PRIMARY_SEED,
                                     max_views_per_scene=args.max_views_per_scene))

    manifest.suite["last_finished_at"] = env_module.utc_now()
    manifest.suite["integrity"] = integrity
    manifest.save()

    (out["root"] / "findings.json").write_text(json.dumps(findings, indent=2,
                                                          default=str))
    path = report.build(findings, environment=environment, integrity=integrity,
                        counts=counts, plan_summary=summary,
                        out_root=out["root"], suite=manifest.suite)

    elapsed = (time.perf_counter() - started) / 3600.0
    print(f"\n[suite] done in {elapsed:.2f}h -> {out['root']}")
    print(f"[suite] report: {path}")
    if not integrity["complete"]:
        print(f"[suite] NOTE: {integrity['failed']} failed and "
              f"{integrity['missing']} missing cells; see report.md section 2 "
              f"and manifest.json. Aggregates over incomplete groups are "
              f"flagged, not silently averaged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
