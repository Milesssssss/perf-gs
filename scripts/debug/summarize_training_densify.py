# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Verify matched full-training benchmark protocols and report wall speedup."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


_MATCHED_FIELDS = (
    "reference_revision", "scene", "backend", "iters", "seed", "sh_degree", "capacity",
    "resolution", "camera_count", "stbn_pixel_fraction", "stbn_source", "stbn_adaptive",
    "settings", "source_fingerprint",
)


def summarize(paths: list[Path]) -> dict:
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise ValueError("no runs supplied")
    for field in _MATCHED_FIELDS:
        if any(run[field] != runs[0][field] for run in runs[1:]):
            raise ValueError(f"unmatched benchmark field: {field}")
    groups = {name: [run for run in runs if run["variant"] == name]
              for name in ("reference", "optimized")}
    if not all(groups.values()):
        raise ValueError("both reference and optimized trials are required")
    summaries = {}
    for variant, picked in groups.items():
        wall = [run["train_wall_s"] for run in picked]
        active = [run["train_active_s"] for run in picked]
        for run, w, a in zip(picked, wall, active):
            settings = run["settings"]
            if any(settings.get(name, 0) for name in (
                "log_every", "preview_every", "eval_every", "psnr_curve",
                "stage_timing_every", "profile_run")):
                raise ValueError("timed training includes observation/profiling passes")
            if not 0 < a <= w:
                raise ValueError("invalid wall/active training durations")
        # Even with every rendering/evaluation observation off, the frontend
        # times CPU throughput reporting and event checks (~tens of ms/run).
        # Exclude that measured CPU cost using the production active-time field;
        # preserve raw wall time alongside it rather than treating it as an eval.
        quality = [run["quality_outside_timing"] for run in picked]
        psnr = [q["mean_sample_psnr_srgb"] for q in quality if "mean_sample_psnr_srgb" in q]
        summaries[variant] = {
            "runs": len(picked), "wall_seconds": wall, "active_seconds": active,
            "mean_wall_seconds": statistics.mean(wall),
            "mean_seconds": statistics.mean(active), "median_seconds": statistics.median(active),
            "min_seconds": min(active), "max_seconds": max(active),
            "cpu_observation_seconds": [w - a for w, a in zip(wall, active)],
            "final_populations": [q["n_gaussians"] for q in quality],
            "sample_psnr_srgb": psnr,
            "mean_sample_psnr_srgb": statistics.mean(psnr) if psnr else None,
            "overbudget_ticks": [sum(t["overbudget"] for t in run["densification_ticks"])
                                 for run in picked],
            "host_densify_seconds": [sum(t["host_ms"] for t in run["densification_ticks"]) / 1000
                                     for run in picked],
        }
    old = summaries["reference"]["mean_seconds"]
    new = summaries["optimized"]["mean_seconds"]
    quality_delta = None
    if all(s["mean_sample_psnr_srgb"] is not None for s in summaries.values()):
        quality_delta = (summaries["optimized"]["mean_sample_psnr_srgb"] -
                         summaries["reference"]["mean_sample_psnr_srgb"])
    return {
        "protocol_verified": True,
        "protocol": {field: runs[0][field] for field in _MATCHED_FIELDS
                     if field != "source_fingerprint"},
        "order": [run["variant"] for run in runs], "variants": summaries,
        "speedup": old / new, "seconds_saved": old - new,
        "time_reduction_percent": 100 * (1 - new / old),
        "throughput_increase_percent": 100 * (old / new - 1),
        "sample_psnr_delta_db": quality_delta,
        "note": "End-to-end ACTIVE training-loop wall time including densification, Morton and Mip. "
            "Measured CPU observation/reporting overhead is subtracted; raw loop wall time is also reported. "
            "Setup/evaluation/export excluded. Quality is a fixed-view sample, not full held-out PSNR. "
                "Host densification time excludes asynchronous GPU completion and is not additive "
                "as a complete densification GPU cost. Two repetitions per variant do not establish "
                "statistical significance; consult run spread and GPU snapshots.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = summarize(args.runs)
    if args.json is not None:
        if args.json.exists():
            parser.error("summary already exists; choose a new path")
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()