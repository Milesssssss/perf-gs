# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Suite entry point.

    .venv/Scripts/python.exe -m adaptive_stbn --stages sampling
    .venv/Scripts/python.exe -m adaptive_stbn --stages gradients
    .venv/Scripts/python.exe -m adaptive_stbn --stages training hyper fractions
    .venv/Scripts/python.exe -m adaptive_stbn --stages report

Stages are independent and resumable, and each records its completion in
``manifest.json`` — so ``report`` can state what it is missing rather than
silently omitting it. Run from the repo root (``scripts/running`` must be on
sys.path; ``scripts/running/adaptive_stbn/run.py`` does that for you).

``--stages all`` runs them in dependency order: sampling and gradients are GATES.
If the estimator is not calibrated and not unbiased, the training numbers are
uninterpretable and the GPU hours are wasted, so they come first and the report
says so if they failed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import config as cfg
from .manifest import Manifest

STAGES = ("sampling", "visuals", "gradients", "training", "hyper", "fractions",
          "biased", "report")
TRAINING_STAGES = ("training", "hyper", "fractions", "biased")
DEFAULT_ORDER = ("sampling", "gradients", "visuals", "training", "hyper",
                 "fractions", "biased", "report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adaptive_stbn",
        description="Importance-guided sparse backward sampling: experiments.")
    parser.add_argument("--stages", nargs="+", default=["all"],
                        help=f"one or more of {list(STAGES)}, or 'all'.")
    parser.add_argument("--out", type=str, default=str(cfg.OUTPUT_ROOT))
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="interpreter used for the training subprocesses.")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="override the pilot scene list (training stage).")
    parser.add_argument("--force", action="store_true",
                        help="retrain every cell even if it is cached.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the training argv without running anything.")
    parser.add_argument("--cell-timeout-s", type=float, default=3600.0)
    parser.add_argument("--iters", type=int, default=cfg.ITERS,
                        help="training iterations per cell (default 3000).")
    parser.add_argument("--gradient-scenes", nargs="*", default=None)
    parser.add_argument("--gradient-cameras", type=int,
                        default=cfg.GRADIENT_CAMERAS)
    parser.add_argument("--gradient-force", action="store_true",
                        help="re-run the gradient study even if its per-scene "
                             "payload is already on disk.")
    parser.add_argument("--visual-scene", type=str, default="bicycle")
    parser.add_argument("--visuals-in-process", action="store_true",
                        help="INTERNAL. Render the visuals in THIS process rather "
                             "than spawning one. Set only by the subprocess the "
                             "visuals stage launches; running the stage this way "
                             "from the top level would leave ~15 GB of device "
                             "allocation resident for whatever stage came next.")
    return parser


def resolve_stages(requested: list[str]) -> list[str]:
    if "all" in requested:
        return list(DEFAULT_ORDER)
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; expected {list(STAGES)}")
    # Keep the dependency order regardless of the order they were typed in.
    return [s for s in DEFAULT_ORDER if s in requested]


def main(argv: list[str] | None = None) -> int:
    from paper_suite.env import collect_environment

    args = build_parser().parse_args(argv)
    stages = resolve_stages(args.stages)
    paths = cfg.output_paths(Path(args.out))
    environment = collect_environment()
    manifest = Manifest.load(paths["manifest"], environment)
    manifest.environment = environment

    print(f"[suite] adaptive_stbn {cfg.SUITE_SCHEMA}", flush=True)
    print(f"[suite] stages: {stages}", flush=True)
    print(f"[suite] out: {paths['root']}", flush=True)

    started = time.perf_counter()
    for stage in stages:
        stage_started = time.perf_counter()
        print(f"\n[suite] ===== stage {stage} =====", flush=True)
        try:
            record = _run_stage(stage, args, paths, manifest)
        except Exception as error:                             # noqa: BLE001
            import traceback
            traceback.print_exc()
            record = {"status": "error",
                      "error": f"{type(error).__name__}: {error}"}
        record = dict(record or {})
        record["wall_s"] = round(time.perf_counter() - stage_started, 2)
        manifest.stages[stage] = record
        manifest.save()
        print(f"[suite] stage {stage}: {record.get('status')} "
              f"in {record['wall_s'] / 60:.1f} min", flush=True)

    print(f"\n[suite] total {(time.perf_counter() - started) / 60:.1f} min",
          flush=True)
    return 0


def _run_stage(stage: str, args, paths: dict, manifest: Manifest) -> dict:
    if stage == "sampling":
        from . import sampling
        return sampling.run(paths)
    if stage == "visuals":
        from . import visuals
        if args.visuals_in_process:
            return visuals.render(paths, scene=args.visual_scene)
        return visuals.run(paths, python=args.python,
                           scene=args.visual_scene, logs_dir=paths["logs"])
    if stage == "gradients":
        from . import gradients
        return gradients.run(paths, python=args.python,
                             scenes=tuple(args.gradient_scenes)
                             if args.gradient_scenes else cfg.GRADIENT_SCENES,
                             cameras=args.gradient_cameras,
                             logs_dir=paths["logs"], force=args.gradient_force)
    if stage in TRAINING_STAGES:
        return _run_training_stage(stage, args, paths, manifest)
    if stage == "report":
        from . import report
        return report.run(paths, manifest)
    raise SystemExit(f"unhandled stage {stage!r}")


def _run_training_stage(stage: str, args, paths: dict,
                        manifest: Manifest) -> dict:
    from dataclasses import replace

    from . import runner

    scenes = tuple(args.scenes) if args.scenes else None
    cells = cfg.build_cells(scenes=scenes, stages=(stage,))
    if args.iters != cfg.ITERS:
        cells = [replace(cell, iters=int(args.iters)) for cell in cells]
    summary = cfg.plan_summary(cells)
    manifest.suite = {**manifest.suite, f"plan_{stage}": summary}
    print(f"[suite] {stage}: {json.dumps(summary)}", flush=True)

    counts = runner.run_plan(cells, manifest, python=args.python,
                             logs_dir=paths["logs"], force=args.force,
                             timeout_s=args.cell_timeout_s,
                             dry_run=args.dry_run)
    integrity = manifest.integrity_report(cells)
    print(f"[suite] {stage}: {counts}  integrity={integrity['ok']}/"
          f"{integrity['planned']} ok", flush=True)
    return {"status": ("ok" if integrity["complete"] else "incomplete"),
            "counts": counts, "integrity": integrity, "plan": summary,
            "cells": [c.key for c in cells]}


if __name__ == "__main__":
    raise SystemExit(main())
