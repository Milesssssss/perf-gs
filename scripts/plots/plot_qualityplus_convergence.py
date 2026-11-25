# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Plot Quality+ training-view convergence and its held-out PSNR change from fixed Quality."""
import argparse
import csv
import json
import math
from pathlib import Path

from matplotlib.figure import Figure

from plot_preset_results import HELDOUT_VIEWS

ROOT = Path(__file__).resolve().parents[2]


def scene_rows(path, preset):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["preset"] == preset]
    by_scene = {row["scene"]: row for row in rows}
    if len(rows) != len(HELDOUT_VIEWS) or {
        scene: int(row["heldout_views"]) for scene, row in by_scene.items()
    } != HELDOUT_VIEWS:
        raise ValueError(f"{path} needs one {preset} run for each standard held-out scene")
    if any(not math.isfinite(float(row["psnr_db"])) for row in rows):
        raise ValueError(f"Missing or invalid held-out PSNR in {path}")
    return by_scene


def probe_trace(run, row, start):
    report = json.loads((run / "run_report.json").read_text(encoding="utf-8"))
    with (run / "log.jsonl").open(encoding="utf-8") as stream:
        probes = [event for event in map(json.loads, stream) if event["event"] == "convergence_probe"]
    steps = int(row["steps"])
    if (report["stop_reason"] != "plateau" or report["iters_completed"] != steps
            or report["device"]["adapter_name"] != row["gpu"] or report["cli_args"]["preset"] != "quality+"
            or report["cli_args"]["seed"] != int(row["seed"]) or report["scene"]["n_test"] != int(row["heldout_views"])
            or not math.isclose(report["timing"]["train_wall_s"], float(row["train_loop_s"]))):
        raise ValueError(f"{run} does not match its scored Quality+ run")
    scene = report["scene"]
    train_count = scene["n_train"]
    expected_steps = range(start, steps + 1, max(100, 4 * train_count))
    if not probes or [p["iter"] for p in probes] != list(expected_steps) or probes[-1]["iter"] != steps:
        raise ValueError(f"Missing convergence checks in {run}")
    cameras = probes[0]["train_indices"]
    scores = [p["train_psnr_srgb"] for p in probes]
    if (not all(map(math.isfinite, scores)) or probes[-1]["stale_checks"] != 3
            or len(set(cameras)) != min(8, train_count) or set(cameras).intersection(scene["test_indices"])
            or any(p["train_indices"] != cameras for p in probes)):
        raise ValueError(f"Invalid training-view convergence trace in {run}")
    return [(step - start) / train_count for step in expected_steps], [score - scores[0] for score in scores]


def plot_convergence(presets_path, plus_path, sweep, output):
    fixed = scene_rows(presets_path, "quality")
    plus = scene_rows(plus_path, "quality+")
    cards = {(row["gpu"], row["seed"]) for row in (*fixed.values(), *plus.values())}
    starts = {int(row["steps"]) for row in fixed.values()}
    if len(cards) != 1 or len(starts) != 1:
        raise ValueError("Both CSVs must use one GPU and seed, and Quality must have a fixed step count")
    gpu, seed = cards.pop()
    start = starts.pop()
    for scene in fixed:
        for field in ("train_width", "train_height", "eval_width", "eval_height", "metric_source"):
            if plus[scene][field] != fixed[scene][field]:
                raise ValueError(f"Mismatched {field} for {scene}")

    fig = Figure(figsize=(11.5, 4.8), layout="constrained")
    trace, heldout = fig.subplots(1, 2)
    fig.suptitle(f"Where Quality+ stops on Mip-NeRF 360 · {gpu.removeprefix('NVIDIA GeForce ')}",
                 weight="bold", fontsize=17)
    trace.set_title("Training views used for stopping")
    for scene, color in (("bicycle", "#196b9b"), ("room", "#b85524")):
        run = ROOT / "outputs/train" / f"sweep_{sweep}_{scene}_{seed}"
        passes, gain = probe_trace(run, plus[scene], start)
        trace.plot(passes, gain, color=color, label=scene.title(), lw=2)
        trace.scatter(passes[-1], gain[-1], color=color, s=30, zorder=3)
    trace.set_xlabel(f"Camera passes after step {start:,}")
    trace.set_ylabel(f"Training PSNR gain since step {start:,} (dB)")
    trace.legend(frameon=False)
    trace.grid(axis="y", color="#eeeeee")

    deltas = {scene: float(plus[scene]["psnr_db"]) - float(row["psnr_db"]) for scene, row in fixed.items()}
    scenes = sorted(deltas, key=deltas.get)
    heldout.barh([s.title() for s in scenes], [deltas[s] for s in scenes],
                 color=["#21725c" if deltas[s] >= 0 else "#af5046" for s in scenes])
    heldout.axvline(0, color="#303940", lw=1)
    heldout.set_title("All official held-out views")
    heldout.set_xlabel("Quality+ minus fixed Quality PSNR (dB)")
    heldout.grid(axis="x", color="#eeeeee")
    for axis in (trace, heldout):
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=160, facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", type=Path, required=True, help="Fixed-budget preset CSV")
    parser.add_argument("--quality-plus", type=Path, required=True, help="Quality+ CSV for the same GPU and seed")
    parser.add_argument("--sweep", required=True, help="Quality+ sweep name under outputs/eval_sweep")
    parser.add_argument("--output", type=Path, required=True, help="Destination figure, e.g. a .png file")
    args = parser.parse_args()
    plot_convergence(args.presets, args.quality_plus, args.sweep, args.output)
