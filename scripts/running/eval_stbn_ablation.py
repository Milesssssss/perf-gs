"""STBN sparse-supervision ablation: 4 pixel-sampling regimes, PSNR + wall time.

Unlike ``eval_sweep.py`` (one flag, many values), this benchmarks four fixed
FLAG COMBINATIONS that a single swept axis can't express, chosen to isolate one
effect at a time:

    full        100% pixels (dense baseline)          --stbn-pixel-fraction 1.0
    iid_white   25%, fresh white mask each step       --stbn-source white  --stbn-independent
    indep_blue  25%, fresh blue mask each step         --stbn-source nvidia --stbn-independent
    stbn        25%, our spatiotemporal blue noise     --stbn-source nvidia

What each contrast measures:
    full vs rest       dense vs sparse (the only pair whose TIMING differs; the
                       three sparse cells share the compacted backward and time
                       alike, so read them for QUALITY, not speed).
    iid_white vs indep_blue   white vs blue arrangement, both drawn independently.
    indep_blue vs stbn        the SAME 64 blue slices, independent order vs the
                              per-visit temporal order — i.e. exactly what the
                              spatiotemporal (temporal) axis buys.

Everything else is held at the published eval config (quarter res, structure-aware
densify, mip-splatting, held-out paper protocol, seed 42) plus --sh-degree 3.
Density is FLAT (no warmup) so the arrangement/temporal contrast isn't blurred by
a dense opening.

    # the full grid (9 scenes x 4 regimes, ~3k iters each)
    .venv/Scripts/python.exe scripts/running/eval_stbn_ablation.py

    # quick smoke on one scene
    .venv/Scripts/python.exe scripts/running/eval_stbn_ablation.py \
        --scenes bicycle --iters 500

Prerequisite for indep_blue + stbn: the NVIDIA asset (NON-COMMERCIAL license) --
    .venv/Scripts/python.exe scripts/data/download_stbn.py

Each cell is a fresh subprocess (its own GPU device + clean densify state), and a
cell whose ``test_psnr.json`` already exists is reused unless ``--force`` -- so an
interrupted grid resumes for free. Results, the CSV and the plot are rewritten
after every cell, so the run is readable while it is still going.

RUN IT ON AN IDLE GPU: ``train_wall_s`` is wall-clock, so a profiler, a second
trainer or a live viewer sharing the device silently inflates every timing.

Outputs, under ``outputs/eval_stbn_ablation/<name>/``:
    results.json           one record per (scene, regime) cell
    ablation_summary.csv   scene x regime table (PSNR + wall time), with averages
    ablation_compare.png   PSNR and wall time per scene, grouped by regime
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

# Canonical Mip-NeRF 360 order: outdoor then indoor. flowers/treehill are the
# "extra" scenes in a separate folder (see _scene_dir).
DEFAULT_SCENES = ["bicycle", "flowers", "garden", "stump", "treehill",
                  "bonsai", "counter", "kitchen", "room"]
DATASET_ROOTS = ["360_v2", "360_extra_scenes"]

# The published eval config, plus SH degree 3 for this study. --evaluation is
# required because it vetoes --final-test-psnr.
BASE_ARGS = [
    "--sh-degree", "3",
    "--densify-mode", "structure_aware",
    "--mip-splatting",
    "--seed", "42",
    "--no-viewer",
    "--log-every", "0",
    "--evaluation",
    "--final-test-psnr",
]

# Column order for the CSV / plot / console table.
REGIMES = ["full", "iid_white", "indep_blue", "stbn"]


def build_conditions(fraction: str) -> list[tuple[str, list[str]]]:
    """The four (label, trainer-flags) cells at the given sparse ``fraction``.

    All three sparse cells share the per-tile count and the compacted backward;
    they differ only in the mask's ARRANGEMENT (white vs blue) and its temporal
    ORDER (independent-per-step vs the STBN per-visit walk).
    """
    f = str(fraction)
    return [
        ("full",       ["--stbn-pixel-fraction", "1.0"]),
        ("iid_white",  ["--stbn-pixel-fraction", f,
                        "--stbn-source", "white", "--stbn-independent"]),
        ("indep_blue", ["--stbn-pixel-fraction", f,
                        "--stbn-source", "nvidia", "--stbn-independent"]),
        ("stbn",       ["--stbn-pixel-fraction", f, "--stbn-source", "nvidia"]),
    ]


def _scene_dir(scene: str) -> Path | None:
    for root in DATASET_ROOTS:
        candidate = REPO / "datasets" / root / scene
        if candidate.is_dir():
            return candidate
    return None


def run_cell(scene: str, label: str, flags: list[str], *, iters: int,
             force: bool, extra: list[str], sweep_name: str) -> dict | None:
    scene_dir = _scene_dir(scene)
    if scene_dir is None:
        print(f"[abl] SKIP {scene}: not found under "
              f"{', '.join('datasets/' + r for r in DATASET_ROOTS)}", flush=True)
        return None

    # sweep_name keys the run dir so two invocations (e.g. a quick smoke and the
    # real 3k grid) never share a test_psnr.json -- the cache is per name.
    run_name = f"{sweep_name}_{scene}_{label}"
    result_json = REPO / "outputs" / "train" / run_name / "test_psnr.json"

    if result_json.exists() and not force:
        print(f"[abl] cached {scene} / {label}", flush=True)
    else:
        cmd = [
            sys.executable, str(REPO / "scripts" / "running" / "v2_train.py"),
            "--scene", str(scene_dir),
            "--iters", str(iters),
            *BASE_ARGS,
            *flags,
            *extra,
            "--run-name", run_name,
        ]
        print(f"[abl] TRAIN {scene} / {label} ...", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO))
        if proc.returncode != 0 or not result_json.exists():
            print(f"[abl] FAILED {scene} / {label} (rc={proc.returncode})",
                  flush=True)
            return None

    data = json.loads(result_json.read_text())
    return {
        "scene": scene,
        "regime": label,
        "psnr_srgb": data.get("psnr_srgb"),
        "train_wall_s": data.get("train_wall_s"),
        "n_gaussians": data.get("n_gaussians"),
        "iters": data.get("iters"),
    }


def _by_cell(results: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["scene"], r["regime"]): r for r in results}


def _write_csv(out_dir: Path, results: list[dict], scenes: list[str]) -> None:
    """Scene x regime table: PSNR then wall time per regime, with an AVERAGE row.

    The pivot (one row per scene, a column per regime) is the view that makes the
    four regimes directly comparable at a glance; results.json keeps the raw
    long-form records.
    """
    cell = _by_cell(results)
    header = (["scene"]
              + [f"{r}_psnr" for r in REGIMES]
              + [f"{r}_wall_s" for r in REGIMES])

    def _row(scene: str) -> list:
        row: list = [scene]
        row += [cell.get((scene, r), {}).get("psnr_srgb") for r in REGIMES]
        row += [cell.get((scene, r), {}).get("train_wall_s") for r in REGIMES]
        return row

    present = [s for s in scenes if any((s, r) in cell for r in REGIMES)]
    rows = [_row(s) for s in present]

    def _avg(col_idx: int) -> float | None:
        vals = [r[col_idx] for r in rows if r[col_idx] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    avg_row = ["AVERAGE"] + [_avg(i) for i in range(1, len(header))]

    with (out_dir / "ablation_summary.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
        if rows:
            writer.writerow(avg_row)


def _plot(out_dir: Path, results: list[dict], scenes: list[str]) -> None:
    """Grouped PSNR + wall-time bars per scene. Soft-fails: a missing/broken
    matplotlib must never cost the (already persisted) numbers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as error:                         # noqa: BLE001
        print(f"[abl] plotting skipped ({error}); numbers are in the CSV",
              flush=True)
        return

    cell = _by_cell(results)
    present = [s for s in scenes if any((s, r) in cell for r in REGIMES)]
    if not present:
        return
    x = np.arange(len(present))
    width = 0.8 / len(REGIMES)
    fig, (ax_p, ax_t) = plt.subplots(
        2, 1, figsize=(max(8, 1.4 * len(present)), 8))
    for i, regime in enumerate(REGIMES):
        off = (i - (len(REGIMES) - 1) / 2) * width
        psnr = [cell.get((s, regime), {}).get("psnr_srgb") or np.nan
                for s in present]
        wall = [cell.get((s, regime), {}).get("train_wall_s") or np.nan
                for s in present]
        ax_p.bar(x + off, psnr, width, label=regime)
        ax_t.bar(x + off, wall, width, label=regime)
    for ax, ylabel in ((ax_p, "held-out test PSNR (sRGB, dB)"),
                       (ax_t, "train wall time (s)")):
        ax.set_xticks(x)
        ax.set_xticklabels(present, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.legend(ncol=len(REGIMES), fontsize=8)
    # Zoom the PSNR panel to the data: the regimes differ by tenths of a dB, which
    # a 0-based axis would flatten into one indistinguishable band.
    finite = [v for s in present for r in REGIMES
              for v in [cell.get((s, r), {}).get("psnr_srgb")] if v is not None]
    if finite:
        pad = max(0.2, (max(finite) - min(finite)) * 0.1)
        ax_p.set_ylim(min(finite) - pad, max(finite) + pad)
    ax_p.set_title("STBN sampling ablation (SH3, 3k iters)")
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_compare.png", dpi=120)
    plt.close(fig)


def _print_summary(results: list[dict]) -> None:
    """Per-regime mean PSNR / wall time + speedup vs the dense baseline."""
    cell = _by_cell(results)
    scenes = sorted({s for s, _ in cell})

    def _mean(regime: str, key: str) -> float | None:
        vals = [cell[(s, regime)][key] for s in scenes
                if (s, regime) in cell and cell[(s, regime)].get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    base = _mean("full", "train_wall_s")
    print(f"\n=== STBN ablation over {len(scenes)} scene(s) ===")
    print(f"{'regime':<12} {'PSNR(dB)':>9} {'wall(s)':>9} {'speedup':>8}")
    for regime in REGIMES:
        psnr, wall = _mean(regime, "psnr_srgb"), _mean(regime, "train_wall_s")
        speed = (base / wall) if base and wall else None
        psnr_s = f"{psnr:>9.3f}" if psnr is not None else f"{'--':>9}"
        wall_s = f"{wall:>9.1f}" if wall is not None else f"{'--':>9}"
        speed_s = f"{speed:>7.2f}x" if speed is not None else f"{'--':>8}"
        print(f"{regime:<12} {psnr_s} {wall_s} {speed_s}")


def _publish(out_dir: Path, results: list[dict], scenes: list[str],
             plot: bool) -> None:
    """Persist results + CSV (+ plot) after every cell, so a long or interrupted
    grid is always readable from what has completed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    if not results:
        return
    _write_csv(out_dir, results, scenes)
    if plot:
        _plot(out_dir, results, scenes)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="STBN sampling ablation over the Mip-NeRF 360 scenes (SH3).")
    ap.add_argument("--scenes", default=",".join(DEFAULT_SCENES),
                    help="comma-separated scene names")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--fraction", default="0.25",
                    help="sparse pixel fraction for the three sparse regimes")
    ap.add_argument("--name", default="stbn_ablation",
                    help="output folder under outputs/eval_stbn_ablation's parent")
    ap.add_argument("--force", action="store_true",
                    help="retrain cells that already have results")
    ap.add_argument("--no-plot", action="store_true",
                    help="collect results + CSV only; skip the chart")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is passed to the trainer "
                         "verbatim (must be LAST)")
    args = ap.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    conditions = build_conditions(args.fraction)
    sweep_name = args.name.replace("-", "_")
    out_dir = REPO / "outputs" / "eval_stbn_ablation" / sweep_name

    print(f"[abl] {len(scenes)} scenes x {len(conditions)} regimes = "
          f"{len(scenes) * len(conditions)} runs at {args.iters} iters "
          f"(fraction {args.fraction}) -> {out_dir}")

    results: list[dict] = []
    for scene in scenes:
        for label, flags in conditions:
            record = run_cell(scene, label, flags, iters=args.iters,
                              force=args.force, extra=list(args.extra),
                              sweep_name=sweep_name)
            if record is None:
                continue
            results.append(record)
            _publish(out_dir, results, scenes, plot=not args.no_plot)

    _print_summary(results)
    print(f"\n[abl] done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
